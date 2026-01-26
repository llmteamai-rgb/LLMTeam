"""
FAISS vector store.

Provides a fast vector store using Facebook AI Similarity Search.
"""

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from llmteam.vectorstores.base import BaseVectorStore, SearchResult


class FAISSStore(BaseVectorStore):
    """
    FAISS vector store for efficient similarity search.

    Uses Facebook AI Similarity Search (FAISS) for fast vector operations.
    Suitable for medium to large datasets on a single machine.

    Requires:
        pip install faiss-cpu
        # or for GPU: pip install faiss-gpu

    Features:
        - Fast similarity search
        - Save/load to disk
        - Memory efficient for large datasets
    """

    def __init__(
        self,
        dimensions: int = 1536,
        index_type: str = "Flat",
        metric: str = "cosine",
        path: Optional[str] = None,
    ):
        """
        Initialize the FAISS store.

        Args:
            dimensions: Dimensionality of the embeddings.
            index_type: FAISS index type ("Flat", "IVF", "HNSW").
            metric: Distance metric ("cosine", "l2", "ip").
            path: Optional path for persistence.
        """
        self._dimensions = dimensions
        self._index_type = index_type
        self._metric = metric
        self._path = path

        self._index = None
        self._texts: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._ids: List[str] = []

        self._ensure_faiss()
        self._init_index()

        # Load existing data if path exists
        if path and os.path.exists(f"{path}.index"):
            self.load(path)

    def _ensure_faiss(self) -> None:
        """Ensure FAISS is available."""
        try:
            import faiss  # noqa: F401
        except ImportError:
            raise ImportError(
                "FAISS store requires 'faiss-cpu' or 'faiss-gpu'. "
                "Install with: pip install faiss-cpu"
            )

    def _init_index(self) -> None:
        """Initialize the FAISS index."""
        import faiss

        if self._metric == "cosine":
            # For cosine similarity, we normalize vectors and use inner product
            if self._index_type == "Flat":
                self._index = faiss.IndexFlatIP(self._dimensions)
            else:
                # Start with flat index for quantizer
                quantizer = faiss.IndexFlatIP(self._dimensions)
                if self._index_type == "IVF":
                    self._index = faiss.IndexIVFFlat(
                        quantizer, self._dimensions, 100, faiss.METRIC_INNER_PRODUCT
                    )
                else:
                    self._index = faiss.IndexFlatIP(self._dimensions)
        elif self._metric == "l2":
            if self._index_type == "Flat":
                self._index = faiss.IndexFlatL2(self._dimensions)
            else:
                quantizer = faiss.IndexFlatL2(self._dimensions)
                if self._index_type == "IVF":
                    self._index = faiss.IndexIVFFlat(quantizer, self._dimensions, 100)
                else:
                    self._index = faiss.IndexFlatL2(self._dimensions)
        else:  # inner product
            self._index = faiss.IndexFlatIP(self._dimensions)

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
            ids: Optional list of IDs.

        Returns:
            List of IDs for the added items.
        """
        if len(texts) != len(embeddings):
            raise ValueError(
                f"texts and embeddings must have same length, "
                f"got {len(texts)} and {len(embeddings)}"
            )

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        elif len(ids) != len(texts):
            raise ValueError(
                f"ids must have same length as texts, got {len(ids)} and {len(texts)}"
            )

        if metadatas is None:
            metadatas = [{} for _ in texts]
        elif len(metadatas) != len(texts):
            raise ValueError(
                f"metadatas must have same length as texts, "
                f"got {len(metadatas)} and {len(texts)}"
            )

        # Convert to numpy array
        vectors = np.array(embeddings, dtype=np.float32)

        # Normalize for cosine similarity
        if self._metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vectors = vectors / norms

        # Add to FAISS index
        self._index.add(vectors)

        # Store metadata
        self._texts.extend(texts)
        self._metadatas.extend(metadatas)
        self._ids.extend(ids)

        # Auto-save if path is set
        if self._path:
            self.save(self._path)

        return ids

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
            List of SearchResult objects.
        """
        if self._index.ntotal == 0:
            return []

        # Prepare query vector
        query_vec = np.array([query_embedding], dtype=np.float32)

        # Normalize for cosine similarity
        if self._metric == "cosine":
            norm = np.linalg.norm(query_vec)
            if norm > 0:
                query_vec = query_vec / norm

        # Search more if we have filter (post-filter)
        search_k = k * 10 if filter else k

        # FAISS search
        distances, indices = self._index.search(query_vec, min(search_k, self._index.ntotal))

        results: List[SearchResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Invalid index
                continue

            # Convert distance to similarity score
            if self._metric == "cosine" or self._metric == "ip":
                score = float(dist)  # Inner product is already similarity
            else:  # L2
                score = 1.0 / (1.0 + float(dist))  # Convert L2 distance to similarity

            # Apply filters
            if filter:
                if not self._matches_filter(self._metadatas[idx], filter):
                    continue

            if score_threshold is not None and score < score_threshold:
                continue

            results.append(
                SearchResult(
                    text=self._texts[idx],
                    score=score,
                    metadata=self._metadatas[idx],
                    id=self._ids[idx],
                )
            )

            if len(results) >= k:
                break

        return results

    def _matches_filter(self, metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """Check if metadata matches the filter."""
        for key, value in filter.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True

    async def delete(self, ids: List[str]) -> int:
        """
        Delete items by their IDs.

        Note: FAISS doesn't support efficient deletion, so we rebuild the index.
        """
        import faiss

        ids_set = set(ids)
        indices_to_keep = [i for i, id_ in enumerate(self._ids) if id_ not in ids_set]
        deleted_count = len(self._ids) - len(indices_to_keep)

        if deleted_count == 0:
            return 0

        # Rebuild with remaining items
        if indices_to_keep:
            # Get vectors for remaining items
            remaining_vectors = np.array(
                [self._index.reconstruct(i) for i in indices_to_keep], dtype=np.float32
            )

            # Update lists
            self._texts = [self._texts[i] for i in indices_to_keep]
            self._metadatas = [self._metadatas[i] for i in indices_to_keep]
            self._ids = [self._ids[i] for i in indices_to_keep]

            # Rebuild index
            self._init_index()
            self._index.add(remaining_vectors)
        else:
            self._texts.clear()
            self._metadatas.clear()
            self._ids.clear()
            self._init_index()

        if self._path:
            self.save(self._path)

        return deleted_count

    async def clear(self) -> int:
        """Clear all items from the store."""
        count = len(self._ids)
        self._texts.clear()
        self._metadatas.clear()
        self._ids.clear()
        self._init_index()

        if self._path:
            self.save(self._path)

        return count

    def count(self) -> int:
        """Get the number of items in the store."""
        return self._index.ntotal

    def save(self, path: str) -> None:
        """Save the index and metadata to disk."""
        import faiss

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self._index, f"{path}.index")

        # Save metadata
        metadata = {
            "texts": self._texts,
            "metadatas": self._metadatas,
            "ids": self._ids,
            "dimensions": self._dimensions,
            "index_type": self._index_type,
            "metric": self._metric,
        }
        with open(f"{path}.meta", "w") as f:
            json.dump(metadata, f)

    def load(self, path: str) -> None:
        """Load the index and metadata from disk."""
        import faiss

        # Load FAISS index
        self._index = faiss.read_index(f"{path}.index")

        # Load metadata
        with open(f"{path}.meta", "r") as f:
            metadata = json.load(f)

        self._texts = metadata["texts"]
        self._metadatas = metadata["metadatas"]
        self._ids = metadata["ids"]
        self._dimensions = metadata["dimensions"]
        self._index_type = metadata["index_type"]
        self._metric = metadata["metric"]


__all__ = ["FAISSStore"]
