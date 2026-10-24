"""Generate and evaluate answers from the final retrieved evidence."""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from coevirag.evaluation.qa_eval import QAExactMatch, QAF1Score
from coevirag.llm.openai_gpt import CacheOpenAI
from coevirag.prompts.prompt_template_manager import PromptTemplateManager
from coevirag.prompts.profiles import qa_prompt_for_model
from coevirag.utils.qa_utils import parse_qa_answer
from coevirag.utils.config_utils import BaseConfig


def _gold_answers(sample: Dict[str, Any]) -> List[str]:
    for key in ("answer", "gold_ans", "answers", "reference"):
        if key in sample:
            value = sample[key]
            if isinstance(value, list):
                return [str(item) for item in value]
            return [str(value)]
    raise KeyError("Sample does not contain a supported gold-answer field.")


def _build_messages(
    manager: PromptTemplateManager,
    question: str,
    docs: List[str],
    prompt_name: str,
) -> List[Dict[str, str]]:
    context = "".join(f"Wikipedia Title: {doc}\n\n" for doc in docs)
    prompt_user = context
    prompt_user += f"Question: {question}\nThought: "
    return manager.render(
        name=prompt_name,
        prompt_user=prompt_user,
        context=context.rstrip(),
        question=question,
    )


def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def _metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    gold = [item["gold_answers"] for item in results]
    predicted = [item["predicted_answer"] for item in results]
    em, _ = QAExactMatch().calculate_metric_scores(gold, predicted, aggregation_fn=np.max)
    f1, _ = QAF1Score().calculate_metric_scores(gold, predicted, aggregation_fn=np.max)
    return {**{k: round(float(v), 4) for k, v in em.items()},
            **{k: round(float(v), 4) for k, v in f1.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate QA from a frozen CoEviRAG retrieval JSON without rerunning retrieval."
    )
    parser.add_argument("--dataset", default="2wikimultihopqa")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--retrieval_json", required=True, type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--qa_prompt", default="auto")
    parser.add_argument("--llm_name", default="openai/gpt-oss-120b")
    parser.add_argument("--llm_base_url", default="https://integrate.api.nvidia.com/v1")
    parser.add_argument(
        "--llm_reasoning_effort",
        choices=("none", "low", "medium", "high"),
        default="low",
    )
    parser.add_argument("--qa_top_k", type=int, choices=(5,), default=5)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--failed_item_max_new_tokens", type=int, default=0,
                        help="Optional output budget for previously failed items only.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--checkpoint_every", type=int, default=20)
    parser.add_argument("--item_retry_attempts", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--max_new_items",
        type=int,
        default=0,
        help="Process at most this many unfinished samples while preserving the full evaluation target.",
    )
    args = parser.parse_args()
    if (args.llm_name != "openai/gpt-oss-120b" or args.qa_prompt not in {"auto", "rag_qa_gptoss"}
            or args.llm_reasoning_effort != "low" or args.max_new_tokens != 1024
            or args.failed_item_max_new_tokens):
        parser.error("CoEviRAG QA uses GPT-OSS-120B, Top-5, low reasoning, and 1024 completion tokens")

    with args.retrieval_json.open(encoding="utf-8") as handle:
        retrieval_payload = json.load(handle)
    retrieval_results = retrieval_payload["results"]

    dataset_path = PROJECT_ROOT / "datasets" / args.dataset / f"{args.split}.json"
    with dataset_path.open(encoding="utf-8") as handle:
        samples = json.load(handle)

    stop = args.offset + args.limit if args.limit > 0 else len(samples)
    samples = samples[args.offset:stop]
    retrieval_results = retrieval_results[args.offset:stop]
    if len(samples) != len(retrieval_results):
        raise ValueError("Dataset and retrieval result lengths do not match.")
    for sample, retrieval in zip(samples, retrieval_results):
        if " ".join(sample["question"].split()) != " ".join(retrieval["question"].split()):
            raise ValueError(f"Question mismatch: {sample['question']!r}")

    save_dir = PROJECT_ROOT / "outputs" / args.dataset / "cache/index"
    config = BaseConfig(
        save_dir=str(save_dir),
        dataset=args.dataset,
        llm_name=args.llm_name,
        llm_base_url=args.llm_base_url,
        llm_reasoning_effort=args.llm_reasoning_effort,
        max_new_tokens=args.max_new_tokens,
        max_retry_attempts=3,
        temperature=0.0,
        seed=0,
    )
    llm = CacheOpenAI.from_experiment_config(config)
    manager = PromptTemplateManager()
    if args.qa_prompt == "auto":
        args.qa_prompt = qa_prompt_for_model(args.llm_name, args.dataset, manager)

    completed: Dict[int, Dict[str, Any]] = {}
    previous_failures = set()
    if args.output_json.exists():
        with args.output_json.open(encoding="utf-8") as handle:
            old_payload = json.load(handle)
        old_meta = old_payload.get("metadata", {})
        compatible = (
            old_meta.get("retrieval_json") == str(args.retrieval_json)
            and old_meta.get("llm_name") == args.llm_name
            and old_meta.get("llm_base_url", args.llm_base_url) == args.llm_base_url
            and old_meta.get("split", "test") == args.split
            and old_meta.get("qa_prompt", "rag_qa_musique") == args.qa_prompt
            and old_meta.get("qa_top_k") == args.qa_top_k
            and old_meta.get("max_new_tokens") == args.max_new_tokens
            and old_meta.get("llm_reasoning_effort", "low") == args.llm_reasoning_effort
        )
        if not compatible:
            raise ValueError("Existing QA output uses a different retrieval/model configuration.")
        completed = {
            int(item["index"]): item
            for item in old_payload.get("results", [])
            if "index" in item
        }
        previous_failures = {int(item["index"]) for item in old_payload.get("failures", [])}

    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "retrieval_json": str(args.retrieval_json),
        "llm_name": args.llm_name,
        "llm_base_url": args.llm_base_url,
        "qa_prompt": args.qa_prompt,
        "qa_top_k": args.qa_top_k,
        "max_new_tokens": args.max_new_tokens,
        "llm_reasoning_effort": args.llm_reasoning_effort,
        "offset": args.offset,
        "limit": len(samples),
        "failed_item_max_new_tokens": args.failed_item_max_new_tokens,
    }
    result_lock = threading.Lock()

    def evaluate_one(index: int) -> Dict[str, Any]:
        sample = samples[index]
        retrieval = retrieval_results[index]
        question = sample["question"]
        absolute_index = args.offset + index
        if absolute_index in completed:
            return completed[absolute_index]
        docs = retrieval["docs"][:args.qa_top_k]
        messages = _build_messages(manager, question, docs, args.qa_prompt)
        last_error = None
        generation_overrides = {}
        if absolute_index in previous_failures and args.failed_item_max_new_tokens > 0:
            generation_overrides["max_completion_tokens"] = args.failed_item_max_new_tokens
        for attempt in range(max(1, args.item_retry_attempts)):
            try:
                response, response_metadata, cache_hit = llm.infer(messages=messages, **generation_overrides)
                return {
                    "index": absolute_index,
                    "question": question,
                    "gold_answers": _gold_answers(sample),
                    "predicted_answer": parse_qa_answer(response),
                    "raw_response": response,
                    "retrieved_doc_count": len(docs),
                    "cache_hit": bool(cache_hit),
                    "qa_retry_count": attempt,
                    "llm_metadata": response_metadata,
                    "effective_max_new_tokens": generation_overrides.get("max_completion_tokens", args.max_new_tokens),
                }
            except Exception as exc:
                last_error = exc
                if attempt + 1 < max(1, args.item_retry_attempts):
                    time.sleep(min(60, 5 * (2 ** attempt) + (index % 7)))
        return {
            "index": absolute_index,
            "question": question,
            "error": f"{type(last_error).__name__}: {last_error}",
        }

    pending = [
        index for index in range(len(samples))
        if args.offset + index not in completed
    ]
    if args.max_new_items > 0:
        pending = pending[:args.max_new_items]
    failures: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        futures = {executor.submit(evaluate_one, index): index for index in pending}
        for completed_count, future in enumerate(
            tqdm(as_completed(futures), total=len(futures), desc="QA Reading"), start=1
        ):
            item = future.result()
            with result_lock:
                if "error" in item:
                    failures.append(item)
                else:
                    completed[int(item["index"])] = item
                if completed_count % max(1, args.checkpoint_every) == 0:
                    ordered = [
                        completed[args.offset + index]
                        for index in range(len(samples))
                        if args.offset + index in completed
                    ]
                    _atomic_write(args.output_json, {
                        "metadata": metadata,
                        "metrics": _metrics(ordered),
                        "results": ordered,
                        "failures": failures,
                    })

    ordered = [
        completed[args.offset + index]
        for index in range(len(samples))
        if args.offset + index in completed
    ]
    payload = {
        "metadata": metadata,
        "metrics": _metrics(ordered),
        "results": ordered,
        "failures": failures,
        "complete": len(ordered) == len(samples),
    }
    _atomic_write(args.output_json, payload)
    if failures:
        raise RuntimeError(
            f"QA pass left {len(failures)} failed sample(s); rerun the same command to retry them."
        )
    print(json.dumps(payload["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
