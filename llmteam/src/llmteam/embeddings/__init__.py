"""
Embeddings package.

Provides embedding providers for RAG pipeline.
"""

from llmteam.embeddings.base import BaseEmbedding
from llmteam.embeddings.openai import OpenAIEmbedding, MODEL_DIMENSIONS as OPENAI_DIMENSIONS
from llmteam.embeddings.huggingface import HuggingFaceEmbedding, MODEL_DIMENSIONS as HF_DIMENSIONS
from llmteam.embeddings.cache import EmbeddingCache

__all__ = [
    "BaseEmbedding",
    "OpenAIEmbedding",
    "HuggingFaceEmbedding",
    "OPENAI_DIMENSIONS",
    "HF_DIMENSIONS",
    "EmbeddingCache",
]
