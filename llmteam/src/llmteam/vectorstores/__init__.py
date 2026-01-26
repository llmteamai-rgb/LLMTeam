"""
Vector stores package.

Provides vector storage and similarity search for RAG pipeline.
"""

from llmteam.vectorstores.base import BaseVectorStore, SearchResult
from llmteam.vectorstores.memory import InMemoryVectorStore

__all__ = [
    "BaseVectorStore",
    "SearchResult",
    "InMemoryVectorStore",
]
