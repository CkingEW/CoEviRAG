import json
import ast
import re
import time
from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..prompts import PromptTemplateManager
from ..utils.logging_utils import get_logger
from ..utils.llm_utils import fix_broken_generated_json, filter_invalid_triples
from ..utils.misc_utils import TripleRawOutput, NerRawOutput
from ..llm.openai_gpt import CacheOpenAI

logger = get_logger(__name__)


class ChunkInfo(TypedDict):
    num_tokens: int
    content: str
    chunk_order: List[Tuple]
    full_doc_ids: List[str]


@dataclass
class LLMInput:
    chunk_id: str
    input_message: List[Dict]


def _extract_json_list(real_response, field_name):
    candidates = [fix_broken_generated_json(real_response)]
    start, end = real_response.find("{"), real_response.rfind("}")
    if start >= 0 and end > start:
        object_candidate = real_response[start:end + 1]
        candidates.extend((object_candidate, fix_broken_generated_json(object_candidate)))
    pattern = rf'\{{[^{{}}]*"{re.escape(field_name)}"\s*:\s*\[[\s\S]*\][^{{}}]*\}}'
    match = re.search(pattern, real_response)
    if match is not None and match.group() not in candidates:
        candidates.extend((match.group(), fix_broken_generated_json(match.group())))
    for candidate in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                value = parser(candidate)
                items = value.get(field_name, []) if isinstance(value, dict) else []
                return items if isinstance(items, list) else []
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue
    return []


def _extract_ner_from_response(real_response):
    return _extract_json_list(real_response, "named_entities")


def _normalize_named_entities(items):
    normalized = []

    def collect(value):
        if isinstance(value, str):
            value = value.strip()
            if value:
                normalized.append(value)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect(child)
        elif isinstance(value, dict):
            preferred = [value[key] for key in ("entity", "name", "text") if key in value]
            for child in preferred or value.values():
                collect(child)

    collect(items)
    return list(dict.fromkeys(normalized))


class OpenIE:
    def __init__(self, llm_model: CacheOpenAI):
        # Init prompt template manager
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = llm_model
        config = llm_model.global_config
        self.num_workers = max(1, int(getattr(config, "openie_num_workers", 16)))
        self.empty_retry_attempts = max(0, int(getattr(config, "openie_empty_retry_attempts", 1)))
        self.pipeline_mode = str(
            getattr(config, "openie_pipeline_mode", "staged")
        ).strip().lower()
        if self.pipeline_mode not in {"staged", "per_document"}:
            raise ValueError(f"Unsupported OpenIE pipeline mode: {self.pipeline_mode}")
        model_name = str(getattr(config, "llm_name", "")).lower()
        # GPT-OSS keeps the legacy two-stage prompt: prior held-out ablations
        # found higher extraction success and coverage than the compact profile.
        self.prompt_profile = "gemma4" if model_name.startswith("google/gemma-4-") else "legacy"

    def _prompt_name(self, base_name: str) -> str:
        if self.prompt_profile == "gemma4" and base_name in {
            "ner", "triple_extraction", "query_triple_extraction"
        }:
            return f"{base_name}_gemma4"
        return base_name

    @staticmethod
    def _correction_messages(input_messages, raw_response, field_name):
        return input_messages + [
            {"role": "assistant", "content": raw_response},
            {
                "role": "user",
                "content": (
                    f'The previous response did not contain a complete, non-empty "{field_name}" JSON list. '
                    "Return only one concise valid JSON object grounded in the passage. "
                    "Do not include reasoning or Markdown."
                ),
            },
        ]

    def ner(self, chunk_key: str, passage: str) -> NerRawOutput:
        # PREPROCESSING
        ner_input_message = self.prompt_template_manager.render(
            name=self._prompt_name('ner'), passage=passage
        )
        raw_response = ""
        metadata = {}
        try:
            # LLM INFERENCE
            retry_count = 0
            messages = ner_input_message
            while True:
                raw_response, metadata, cache_hit = self.llm_model.infer(messages=messages)
                metadata['cache_hit'] = cache_hit
                real_response = (
                    fix_broken_generated_json(raw_response)
                    if metadata['finish_reason'] == 'length'
                    else raw_response
                )
                extracted_entities = _extract_ner_from_response(real_response)
                if extracted_entities or retry_count >= self.empty_retry_attempts:
                    break
                retry_count += 1
                messages = self._correction_messages(messages, raw_response, "named_entities")
            metadata['openie_retry_count'] = retry_count
            unique_entities = _normalize_named_entities(extracted_entities)

        except Exception as e:
            # For any other unexpected exceptions, log them and return with the error message
            logger.warning(e)
            metadata.update({'error': str(e)})
            return NerRawOutput(
                chunk_id=chunk_key,
                response=raw_response,  # Store the error message in metadata
                unique_entities=[],
                metadata=metadata  # Store the error message in metadata
            )

        return NerRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            unique_entities=unique_entities,
            metadata=metadata
        )

    def triple_extraction(self, chunk_key: str, passage: str, named_entities: List[str], prompt_name: str = 'triple_extraction') -> TripleRawOutput:
        def _extract_triples_from_response(real_response):
            return _extract_json_list(real_response, "triples")

        # PREPROCESSING
        messages = self.prompt_template_manager.render(
            name=self._prompt_name(prompt_name),
            passage=passage,
            named_entity_json=json.dumps({"named_entities": named_entities})
        )

        raw_response = ""
        metadata = {}
        try:
            # LLM INFERENCE
            retry_count = 0
            inference_messages = messages
            while True:
                raw_response, metadata, cache_hit = self.llm_model.infer(messages=inference_messages)
                metadata['cache_hit'] = cache_hit
                real_response = (
                    fix_broken_generated_json(raw_response)
                    if metadata['finish_reason'] == 'length'
                    else raw_response
                )
                extracted_triples = _extract_triples_from_response(real_response)
                triplets = filter_invalid_triples(triples=extracted_triples)
                if triplets or retry_count >= self.empty_retry_attempts:
                    break
                retry_count += 1
                inference_messages = self._correction_messages(
                    inference_messages, raw_response, "triples"
                )
            metadata['openie_retry_count'] = retry_count

        except Exception as e:
            logger.warning(f"Exception for chunk {chunk_key}: {e}")
            metadata.update({'error': str(e)})
            return TripleRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                metadata=metadata,
                triples=[]
            )

        # Success
        return TripleRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            metadata=metadata,
            triples=triplets
        )

    def openie(self, chunk_key: str, passage: str) -> Dict[str, Any]:
        ner_output = self.ner(chunk_key=chunk_key, passage=passage)
        triple_output = self.triple_extraction(chunk_key=chunk_key, passage=passage, named_entities=ner_output.unique_entities)
        return {"ner": ner_output, "triplets": triple_output}

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """
        Conduct batch OpenIE synchronously using multi-threading which includes NER and triple extraction.

        Args:
            chunks (Dict[str, ChunkInfo]): chunks to be incorporated into graph. Each key is a hashed chunk 
            and the corresponding value is the chunk info to insert.

        Returns:
            Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
                - A dict with keys as the chunk ids and values as the NER result instances.
                - A dict with keys as the chunk ids and values as the triple extraction result instances.
        """

        if self.pipeline_mode == "per_document":
            return self._batch_openie_per_document(chunks)

        # Extract passages from the provided chunks
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}

        ner_results_list = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Create NER futures for each chunk
            ner_futures = {
                executor.submit(self.ner, chunk_key, passage): chunk_key
                for chunk_key, passage in chunk_passages.items()
            }

            pbar = tqdm(as_completed(ner_futures), total=len(ner_futures), desc="NER")
            for future in pbar:
                result = future.result()
                ner_results_list.append(result)
                # Update metrics based on the metadata from the result
                metadata = result.metadata
                total_prompt_tokens += metadata.get('prompt_tokens', 0)
                total_completion_tokens += metadata.get('completion_tokens', 0)
                if metadata.get('cache_hit'):
                    num_cache_hit += 1

                pbar.set_postfix({
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'num_cache_hit': num_cache_hit
                })

        triple_results_list = []
        total_prompt_tokens, total_completion_tokens, num_cache_hit = 0, 0, 0
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Create triple extraction futures for each chunk
            re_futures = {
                executor.submit(self.triple_extraction, ner_result.chunk_id,
                                chunk_passages[ner_result.chunk_id],
                                ner_result.unique_entities): ner_result.chunk_id
                for ner_result in ner_results_list
            }
            # Collect triple extraction results with progress bar
            pbar = tqdm(as_completed(re_futures), total=len(re_futures), desc="Extracting triples")
            for future in pbar:
                result = future.result()
                triple_results_list.append(result)
                metadata = result.metadata
                total_prompt_tokens += metadata.get('prompt_tokens', 0)
                total_completion_tokens += metadata.get('completion_tokens', 0)
                if metadata.get('cache_hit'):
                    num_cache_hit += 1
                pbar.set_postfix({
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'num_cache_hit': num_cache_hit
                })

        ner_results_dict = {res.chunk_id: res for res in ner_results_list}
        triple_results_dict = {res.chunk_id: res for res in triple_results_list}

        return ner_results_dict, triple_results_dict

    def _batch_openie_per_document(
        self, chunks: Dict[str, ChunkInfo]
    ) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """Pipeline NER and triple extraction independently for each passage."""
        chunk_passages = {
            chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()
        }
        ner_results_dict = {}
        triple_results_dict = {}
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self.openie, chunk_key, passage): chunk_key
                for chunk_key, passage in chunk_passages.items()
            }
            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="OpenIE pipeline",
            )
            for future in pbar:
                result = future.result()
                ner_output = result["ner"]
                triple_output = result["triplets"]
                ner_results_dict[ner_output.chunk_id] = ner_output
                triple_results_dict[triple_output.chunk_id] = triple_output
                for metadata in (ner_output.metadata, triple_output.metadata):
                    total_prompt_tokens += metadata.get("prompt_tokens", 0)
                    total_completion_tokens += metadata.get("completion_tokens", 0)
                    if metadata.get("cache_hit"):
                        num_cache_hit += 1
                pbar.set_postfix({
                    "total_prompt_tokens": total_prompt_tokens,
                    "total_completion_tokens": total_completion_tokens,
                    "num_cache_hit": num_cache_hit,
                })

        return ner_results_dict, triple_results_dict
