"""
Embeddings package.

Provides embedding providers for RAG pipeline.
"""

from llmteam.embeddings.base import BaseEmbedding
from llmteam.embeddings.openai import OpenAIEmbedding, MODEL_DIMENSIONS
from llmteam.embeddings.cache import EmbeddingCache

__all__ = [
    "BaseEmbedding",
    "OpenAIEmbedding",
    "MODEL_DIMENSIONS",
    "EmbeddingCache",
]
