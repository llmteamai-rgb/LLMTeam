"""
In-memory vector store using numpy.

Provides a simple vector store for development and testing.
"""

import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from llmteam.vectorstores.base import BaseVectorStore, SearchResult


class InMemoryVectorStore(BaseVectorStore):
    """
    In-memory vector store using numpy for cosine similarity.

    Suitable for development, testing, and small datasets.
    For production use with large datasets, consider external vector databases.
    """

    def __init__(self):
        """Initialize the in-memory vector store."""
        self._vectors: List[np.ndarray] = []
        self._texts: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._ids: List[str] = []

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
        if len(texts) != len(embeddings):
            raise ValueError(
                f"texts and embeddings must have same length, "
                f"got {len(texts)} and {len(embeddings)}"
            )

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        elif len(ids) != len(texts):
            raise ValueError(
                f"ids must have same length as texts, got {len(ids)} and {len(texts)}"
            )

        # Handle metadatas
        if metadatas is None:
            metadatas = [{} for _ in texts]
        elif len(metadatas) != len(texts):
            raise ValueError(
                f"metadatas must have same length as texts, "
                f"got {len(metadatas)} and {len(texts)}"
            )

        # Add to store
        for text, embedding, metadata, id_ in zip(texts, embeddings, metadatas, ids):
            self._vectors.append(np.array(embedding, dtype=np.float32))
            self._texts.append(text)
            self._metadatas.append(metadata)
            self._ids.append(id_)

        return ids

    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar vectors using cosine similarity.

        Args:
            query_embedding: Query vector to search for.
            k: Number of results to return.
            score_threshold: Minimum similarity score (0-1).
            filter: Optional metadata filter (exact match).

        Returns:
            List of SearchResult objects, sorted by similarity (highest first).
        """
        if not self._vectors:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)

        # Normalize query vector for cosine similarity
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        query_vec = query_vec / query_norm

        # Calculate similarities
        results: List[SearchResult] = []

        for i, (vec, text, metadata, id_) in enumerate(
            zip(self._vectors, self._texts, self._metadatas, self._ids)
        ):
            # Apply filter if provided
            if filter:
                if not self._matches_filter(metadata, filter):
                    continue

            # Cosine similarity
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0:
                continue
            normalized_vec = vec / vec_norm
            similarity = float(np.dot(query_vec, normalized_vec))

            # Apply score threshold
            if score_threshold is not None and similarity < score_threshold:
                continue

            results.append(
                SearchResult(
                    text=text,
                    score=similarity,
                    metadata=metadata,
                    id=id_,
                )
            )

        # Sort by score descending and take top k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    def _matches_filter(self, metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """Check if metadata matches the filter (exact match)."""
        for key, value in filter.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True

    async def delete(self, ids: List[str]) -> int:
        """
        Delete items by their IDs.

        Args:
            ids: List of IDs to delete.

        Returns:
            Number of items deleted.
        """
        ids_set = set(ids)
        indices_to_delete = [i for i, id_ in enumerate(self._ids) if id_ in ids_set]

        # Delete in reverse order to maintain indices
        for i in reversed(indices_to_delete):
            del self._vectors[i]
            del self._texts[i]
            del self._metadatas[i]
            del self._ids[i]

        return len(indices_to_delete)

    async def clear(self) -> int:
        """
        Clear all items from the store.

        Returns:
            Number of items cleared.
        """
        count = len(self._vectors)
        self._vectors.clear()
        self._texts.clear()
        self._metadatas.clear()
        self._ids.clear()
        return count

    def count(self) -> int:
        """
        Get the number of items in the store.

        Returns:
            Number of items.
        """
        return len(self._vectors)

    async def get_by_ids(self, ids: List[str]) -> List[SearchResult]:
        """
        Get items by their IDs.

        Args:
            ids: List of IDs to retrieve.

        Returns:
            List of SearchResult objects (score will be 1.0).
        """
        ids_set = set(ids)
        results = []

        for i, id_ in enumerate(self._ids):
            if id_ in ids_set:
                results.append(
                    SearchResult(
                        text=self._texts[i],
                        score=1.0,
                        metadata=self._metadatas[i],
                        id=id_,
                    )
                )

        return results


__all__ = ["InMemoryVectorStore"]
