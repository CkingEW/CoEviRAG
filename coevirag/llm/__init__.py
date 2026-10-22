"""CoEviRAG release factory: remote OpenAI-compatible chat only."""
from .base import BaseLLM
from .openai_gpt import CacheOpenAI


def _get_llm_class(config):
    return CacheOpenAI.from_experiment_config(config)
