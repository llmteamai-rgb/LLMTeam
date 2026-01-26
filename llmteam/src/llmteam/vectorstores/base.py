"""
Base vector store protocol.

Defines the interface for all vector stores.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SearchResult:
    """
    Result from a vector similarity search.

    Attributes:
        text: The text content of the result.
        score: Similarity score (higher is more similar).
        metadata: Associated metadata.
        id: Optional unique identifier.
    """

    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None


class BaseVectorStore(ABC):
    """
    Abstract base class for vector stores.

    All vector stores must implement add and similarity_search methods
    for storing and retrieving vectors.
    """

    @abstractmethod
    async def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add texts with their embeddings to the store.

        Args:
            texts: List of text content.
            embeddings: List of embedding vectors.
            metadatas: Optional list of metadata dicts.
            ids: Optional list of IDs. Auto-generated if not provided.

        Returns:
            List of IDs for the added items.
        """
        pass

    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query vector to search for.
            k: Number of results to return.
            score_threshold: Minimum similarity score (0-1).
            filter: Optional metadata filter.

        Returns:
            List of SearchResult objects, sorted by similarity.
        """
        pass

    @abstractmethod
    async def delete(self, ids: List[str]) -> int:
        """
        Delete items by their IDs.

        Args:
            ids: List of IDs to delete.

        Returns:
            Number of items deleted.
        """
        pass

    @abstractmethod
    async def clear(self) -> int:
        """
        Clear all items from the store.

        Returns:
            Number of items cleared.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Get the number of items in the store.

        Returns:
            Number of items.
        """
        pass


__all__ = ["BaseVectorStore", "SearchResult"]
