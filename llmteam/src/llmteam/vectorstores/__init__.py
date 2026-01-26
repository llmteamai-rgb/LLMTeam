"""
Vector stores package.

Provides vector storage and similarity search for RAG pipeline.
"""

from llmteam.vectorstores.base import BaseVectorStore, SearchResult
from llmteam.vectorstores.memory import InMemoryVectorStore
from llmteam.vectorstores.auto import AutoStore, select_store

# FAISS and Qdrant are lazy-loaded to avoid import errors when dependencies are missing
# Use: from llmteam.vectorstores.faiss import FAISSStore
# Use: from llmteam.vectorstores.qdrant import QdrantStore

__all__ = [
    "BaseVectorStore",
    "SearchResult",
    "InMemoryVectorStore",
    "AutoStore",
    "select_store",
]
