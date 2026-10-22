import json
import re
from typing import Dict, List, Union, Any, Optional


from ..prompts.prompt_template_manager import PromptTemplateManager
from .logging_utils import get_logger
from ..llm.openai_gpt import CacheOpenAI

logger = get_logger(__name__)


def parse_qa_answer(response: str) -> str:
    """Parse strict JSON QA output with backward-compatible legacy fallbacks."""
    response = str(response or "").strip()
    for candidate in reversed(re.findall(r"\{[\s\S]*?\}", response)):
        try:
            value = json.loads(candidate).get("answer")
            if value is not None:
                return str(value).strip()
        except (json.JSONDecodeError, AttributeError):
            continue

    for pattern in (
        r"\bAnswer\s*:\s*(.+)",
        r"\bSo the answer is\s*:\s*(.+)",
        r"\bSo the answer is\s+(.+)",
    ):
        matches = re.findall(pattern, response, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip().splitlines()[0].strip(" `\t")
    return response.splitlines()[-1].strip() if response else ""



def merge_elements_with_same_first_line(elements, prefix='Wikipedia Title: '):
    merged_dict = {}

    # Iterate through each element in the list
    for element in elements:
        # Split the element into lines and get the first line
        lines = element.split('\n')
        first_line = lines[0]

        # Check if the first line is already a key in the dictionary
        if first_line in merged_dict:
            # Append the current element to the existing value
            merged_dict[first_line] += "\n" + element.split(first_line, 1)[1].strip('\n')
        else:
            # Add the current element as a new entry in the dictionary
            merged_dict[first_line] = prefix + element

    # Extract the merged elements from the dictionary
    merged_elements = list(merged_dict.values())
    return merged_elements


def reason_step(dataset, prompt_template_manager: PromptTemplateManager, query: str, passages: list, thoughts: list, llm_client: CacheOpenAI):
    """
    Given few-shot samples, query, previous retrieved passages, and previous thoughts, generate the next thought with OpenAI models. The generated thought is used for further retrieval step.
    :return: next thought
    """

    prompt_user = ''
    if dataset in ['hotpotqa', 'hotpotqa_train']:
        passages = merge_elements_with_same_first_line(passages)
    for passage in passages:
        prompt_user += f'{passage}\n\n'
    prompt_user += f'Question: {query}\nThought:' + ' '.join(thoughts)

    
    messages = prompt_template_manager.render(name=f'ircot_{dataset}', prompt_user=prompt_user)

    try:
        response_content, metadata, cache_hit = llm_client.infer(messages=messages)
    except Exception as e:
        logger.exception("An exception occurred while calling LLM for QA!")
        return ''
    
    return response_content
