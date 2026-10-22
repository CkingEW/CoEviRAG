# Modified for CoEviRAG: omit unused legacy token-counter dependency.
import json
import re
from string import Template
from typing import (
    Optional,
    Union,
    List,
    TypedDict
)

from openai import (
    APIConnectionError,
    RateLimitError,
    Timeout
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .config_utils import BaseConfig


class TextChatMessage(TypedDict):
    """Representation of a single text-based chat message in the chat history."""
    role: str  # Either "system", "user", or "assistant"
    content: Union[str, Template]  # The text content of the message (could also be a string.Template instance)


def convert_format_to_template(original_string: str, placeholder_mapping: Optional[dict] = None, static_values: Optional[dict] = None) -> str:
    """
    Converts a .format() style string to a Template-style string.

    Args:
        original_string (str): The original string using .format() placeholders.
        placeholder_mapping (dict, optional): Mapping from original placeholder names to new placeholder names.
        static_values (dict, optional): Mapping from original placeholders to static values to be replaced in the new template.

    Returns:
        str: The converted string in Template-style format.
    """
    # Initialize mappings
    placeholder_mapping = placeholder_mapping or {}
    static_values = static_values or {}

    # Regular expression to find .format() style placeholders
    placeholder_pattern = re.compile(r'\{(\w+)\}')

    # Substitute placeholders in the string
    def replace_placeholder(match):
        original_placeholder = match.group(1)

        # If the placeholder is in static_values, substitute its value directly
        if original_placeholder in static_values:
            return str(static_values[original_placeholder])

        # Otherwise, rename the placeholder if needed, or keep it as is
        new_placeholder = placeholder_mapping.get(original_placeholder, original_placeholder)
        return f'${{{new_placeholder}}}'

    # Replace all placeholders
    template_string = placeholder_pattern.sub(replace_placeholder, original_string)

    return template_string


def safe_unicode_decode(content: Union[bytes, str]) -> str:
    """
    Safely decodes a string or byte sequence containing Unicode escape sequences into a properly decoded Unicode string.

    Args:
        content (Union[bytes, str]): The input data, either as a byte sequence or a string, which may contain Unicode escape sequences.

    Returns:
        str: The decoded Unicode string with escape sequences replaced by their corresponding characters.
    
    Raises:
        AttributeError: If the input is neither bytes nor a string.
    """
    # Regular expression to find all Unicode escape sequences of the form \uXXXX
    unicode_escape_pattern = re.compile(r"\\u([0-9a-fA-F]{4})")

    # Function to replace the Unicode escape with the actual character
    def replace_unicode_escape(match: re.Match) -> str:
        # Convert the matched hexadecimal value into the actual Unicode character
        return chr(int(match.group(1), 16))

    # If content is bytes, decode it to a UTF-8 string
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    elif not isinstance(content, str):
        raise AttributeError("Input must be of type bytes or str.")

    # Perform the substitution
    decoded_content = unicode_escape_pattern.sub(replace_unicode_escape, content)

    return decoded_content


def dynamic_retry(experiment_config: BaseConfig):
    """
    Factory function to create a retry decorator with dynamic parameters.

    Args:
        experiment_config (BaseConfig): Configuration containing stop and wait parameters. Expected to use the global config for running all experiments. 

    Returns:
        Callable: A retry decorator with dynamically set parameters.
    """
    stop = stop_after_attempt(experiment_config.async_max_retry_attempts)
    wait = wait_exponential(
        multiplier=experiment_config.async_retry_wait_exp_multiplier,
        min=experiment_config.async_retry_min_wait_exp_time,
        max=experiment_config.async_retry_max_wait_exp_time,
    )

    return retry(
        stop=stop,
        wait=wait,
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
    )


# def fix_broken_generated_json(json_str: str) -> str:
#     """
#     Fixes a malformed JSON string by:
#     - Removing the last comma and any trailing content.
#     - Appending a closing bracket `]` and brace `}` to properly terminate the JSON.

#     Args:
#         json_str (str): The malformed JSON string to be fixed.

#     Returns:
#         str: The corrected JSON string.
#     """
#     last_comma_index = json_str.rfind(',')
#     if last_comma_index != -1:
#         json_str = json_str[:last_comma_index]

#     processed_string = json_str + ']\n}'
#     return processed_string


def fix_broken_generated_json(json_str: str) -> str:
    """
    Repair missing or mismatched JSON closers without discarding valid items.

    If the original json_str string can be successfully loaded by json.loads(), will directly return it without any modification.
    
    Args:
        json_str (str): The malformed JSON string to be fixed.

    Returns:
        str: The corrected JSON string.
    """

    try:
        # Try to load the JSON to see if it is valid
        json.loads(json_str)
        return json_str  # Return as-is if valid
    except json.JSONDecodeError as e:
        pass

    # Models occasionally close the outer object before closing an inner list,
    # e.g. {"triples": [[...]]}. Insert the missing closer at the mismatch
    # instead of dropping the final item, which can discard a valid response.
    repaired = []
    stack = []
    inside_string = False
    escape_next = False
    matching = {'}': '{', ']': '['}
    closing = {'{': '}', '[': ']'}

    for char in json_str.strip():
        if inside_string:
            repaired.append(char)
            if escape_next:
                escape_next = False
            elif char == '\\':
                escape_next = True
            elif char == '"':
                inside_string = False
            continue

        if char == '"':
            inside_string = True
            repaired.append(char)
        elif char in '{[':
            stack.append(char)
            repaired.append(char)
        elif char in '}]':
            expected = matching[char]
            while stack and stack[-1] != expected:
                repaired.append(closing[stack.pop()])
            if stack:
                stack.pop()
                repaired.append(char)
        else:
            repaired.append(char)

    if inside_string:
        repaired.append('"')
    while stack:
        stack_top = stack.pop()
        while repaired and repaired[-1].isspace():
            repaired.pop()
        if repaired and repaired[-1] == ',':
            repaired.pop()
        repaired.append(closing[stack_top])

    return ''.join(repaired)


def filter_invalid_triples(triples: List[List[str]]) -> List[List[str]]:
    """
    Filters out invalid and duplicate triples from a list of triples.

    A valid triple meets the following criteria:
    1. It contains exactly three elements.
    2. It is unique within the list (no duplicates in the output).

    The function ensures:
    - Each valid triple is converted to a list of strings.
    - The order of unique, valid triples is preserved.
    - Do not apply any text preprocessing techniques or rules within this function.
    
    Args:
        triples (List[List[str]]): 
            A list of triples (each a list of strings or elements that can be converted to strings).

    Returns:
        List[List[str]]: 
            A list of unique, valid triples, each represented as a list of strings.
    """
    unique_triples = set()
    valid_triples = []

    for triple in triples:
        if len(triple) != 3: continue  # Skip triples that do not have exactly 3 elements

        valid_triple = [str(item) for item in triple]
        if tuple(valid_triple) not in unique_triples:
            unique_triples.add(tuple(valid_triple))
            valid_triples.append(valid_triple)

    return valid_triples


PROMPT_JSON_TEMPLATE = {
    "ner": {
        "type": "object",
        "properties": {
            "named_entities": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "minItems": 0
            }
        },
        "required": ["named_entities"]
    },
    "triples": {
        "type": "object",
        "properties": {
            "triples": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "maxItems": 3,
                    "minItems": 3,
                },
                "minItems": 0
            }
        },
        "required": ["triples"]
    },
    "fact": {
        "type": "object",
        "properties": {
            "fact": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "maxItems": 3,
                    "minItems": 3,
                },
                "minItems": 0
            }
        },
        "required": ["fact"]
    },
    "json": {
        "type": "object",
    },
    "qa_cot": {
        "type": "object",
        "required": ["Thought", "Answer"],
        "properties": {
            "Thought": {
                "type": "string",
                "minLength": 1,
                "maxLength": 2000
            },
            "Answer": {
                "type": "string",
                "minLength": 1,
                "maxLength": 200
            },
        },
    }
}


