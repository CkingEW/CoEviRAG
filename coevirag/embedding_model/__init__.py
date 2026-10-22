"""CoEviRAG release factory: Qwen3 embeddings only."""
from .base import BaseEmbeddingModel, EmbeddingConfig
from .qwen3 import QwenEmbeddingModel


def _get_embedding_model_class(embedding_model_name="Qwen/Qwen3-Embedding-4B"):
    if "Qwen3" not in embedding_model_name:
        raise ValueError("This release supports Qwen3 embeddings only")
    return QwenEmbeddingModel
