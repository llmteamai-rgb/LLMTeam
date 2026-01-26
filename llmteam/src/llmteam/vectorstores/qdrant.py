"""
Qdrant vector store.

Provides a production-ready vector store using Qdrant.
"""

import uuid
from typing import Any, Dict, List, Optional

from llmteam.vectorstores.base import BaseVectorStore, SearchResult


class QdrantStore(BaseVectorStore):
    """
    Qdrant vector store for production deployments.

    Supports both local (in-memory or on-disk) and cloud deployments.

    Requires:
        pip install qdrant-client

    Features:
        - Efficient filtering
        - Horizontal scaling
        - Persistent storage
        - Cloud support (Qdrant Cloud)
    """

    def __init__(
        self,
        collection: str = "default",
        dimensions: int = 1536,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        path: Optional[str] = None,
        prefer_grpc: bool = False,
    ):
        """
        Initialize the Qdrant store.

        Args:
            collection: Name of the collection.
            dimensions: Dimensionality of the embeddings.
            url: URL of Qdrant server (None for in-memory).
            api_key: API key for Qdrant Cloud.
            path: Path for local persistent storage.
            prefer_grpc: Use gRPC instead of REST (faster).
        """
        self._collection = collection
        self._dimensions = dimensions
        self._url = url
        self._api_key = api_key
        self._path = path
        self._prefer_grpc = prefer_grpc
        self._client = None

        self._ensure_client()
        self._ensure_collection()

    def _ensure_client(self) -> None:
        """Ensure Qdrant client is initialized."""
        if self._client is not None:
            return

        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError(
                "Qdrant store requires 'qdrant-client'. "
                "Install with: pip install qdrant-client"
            )

        if self._url:
            # Remote server
            self._client = QdrantClient(
                url=self._url,
                api_key=self._api_key,
                prefer_grpc=self._prefer_grpc,
            )
        elif self._path:
            # Local persistent storage
            self._client = QdrantClient(path=self._path)
        else:
            # In-memory
            self._client = QdrantClient(":memory:")

    def _ensure_collection(self) -> None:
        """Ensure collection exists."""
        from qdrant_client.models import Distance, VectorParams

        collections = self._client.get_collections().collections
        exists = any(c.name == self._collection for c in collections)

        if not exists:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._dimensions,
                    distance=Distance.COSINE,
                ),
            )

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
        from qdrant_client.models import PointStruct

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

        points = []
        for id_, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
            payload = {
                "_text": text,
                **metadata,
            }
            points.append(
                PointStruct(
                    id=id_,
                    vector=embedding,
                    payload=payload,
                )
            )

        self._client.upsert(
            collection_name=self._collection,
            points=points,
        )

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
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Build filter if provided
        qdrant_filter = None
        if filter:
            conditions = []
            for key, value in filter.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )
            qdrant_filter = Filter(must=conditions)

        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_embedding,
            limit=k,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
        )

        search_results = []
        for result in results:
            payload = result.payload or {}
            text = payload.pop("_text", "")
            search_results.append(
                SearchResult(
                    text=text,
                    score=result.score,
                    metadata=payload,
                    id=str(result.id),
                )
            )

        return search_results

    async def delete(self, ids: List[str]) -> int:
        """Delete items by their IDs."""
        from qdrant_client.models import PointIdsList

        # Get count before deletion
        count_before = self.count()

        self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=ids),
        )

        count_after = self.count()
        return count_before - count_after

    async def clear(self) -> int:
        """Clear all items from the store."""
        count = self.count()

        # Delete and recreate collection
        self._client.delete_collection(self._collection)
        self._ensure_collection()

        return count

    def count(self) -> int:
        """Get the number of items in the store."""
        info = self._client.get_collection(self._collection)
        return info.points_count


__all__ = ["QdrantStore"]
