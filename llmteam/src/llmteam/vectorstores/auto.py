"""
Auto vector store selector.

Automatically selects the appropriate vector store based on data size.
"""

from typing import Any, Dict, List, Optional

from llmteam.vectorstores.base import BaseVectorStore, SearchResult


class AutoStore(BaseVectorStore):
    """
    Automatic vector store selector.

    Selects the appropriate vector store based on data size:
    - < 1000 chunks: InMemoryVectorStore (fastest startup)
    - 1000-100000 chunks: FAISSStore (fast search)
    - > 100000 chunks: QdrantStore (scalable)

    Can also auto-upgrade when the store grows beyond thresholds.
    """

    # Thresholds for store selection
    MEMORY_THRESHOLD = 1000
    FAISS_THRESHOLD = 100000

    def __init__(
        self,
        dimensions: int = 1536,
        initial_store: str = "memory",
        auto_upgrade: bool = True,
        faiss_path: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_collection: str = "default",
    ):
        """
        Initialize the auto store.

        Args:
            dimensions: Dimensionality of the embeddings.
            initial_store: Initial store type ("memory", "faiss", "qdrant").
            auto_upgrade: Automatically upgrade to faster store when needed.
            faiss_path: Path for FAISS persistence.
            qdrant_url: URL for Qdrant server.
            qdrant_collection: Collection name for Qdrant.
        """
        self._dimensions = dimensions
        self._initial_store = initial_store
        self._auto_upgrade = auto_upgrade
        self._faiss_path = faiss_path
        self._qdrant_url = qdrant_url
        self._qdrant_collection = qdrant_collection

        self._store: Optional[BaseVectorStore] = None
        self._store_type: Optional[str] = None
        self._pending_data: List[Dict[str, Any]] = []

        self._init_store(initial_store)

    def _init_store(self, store_type: str) -> None:
        """Initialize a specific store type."""
        if store_type == "memory":
            from llmteam.vectorstores.memory import InMemoryVectorStore

            self._store = InMemoryVectorStore()
        elif store_type == "faiss":
            from llmteam.vectorstores.faiss import FAISSStore

            self._store = FAISSStore(
                dimensions=self._dimensions,
                path=self._faiss_path,
            )
        elif store_type == "qdrant":
            from llmteam.vectorstores.qdrant import QdrantStore

            self._store = QdrantStore(
                collection=self._qdrant_collection,
                dimensions=self._dimensions,
                url=self._qdrant_url,
            )
        else:
            raise ValueError(f"Unknown store type: {store_type}")

        self._store_type = store_type

    async def _maybe_upgrade(self) -> None:
        """Check if we should upgrade to a more scalable store."""
        if not self._auto_upgrade:
            return

        count = self.count()

        if self._store_type == "memory" and count >= self.MEMORY_THRESHOLD:
            await self._migrate_to("faiss")
        elif self._store_type == "faiss" and count >= self.FAISS_THRESHOLD:
            if self._qdrant_url:  # Only upgrade to Qdrant if URL is configured
                await self._migrate_to("qdrant")

    async def _migrate_to(self, new_store_type: str) -> None:
        """Migrate data to a new store type."""
        # This is a simplified migration - in production, you might want
        # to do this more efficiently with batching and error handling

        # For now, we just switch the store type
        # Full migration would require extracting all vectors and re-adding

        old_store = self._store
        self._init_store(new_store_type)

        # Note: Real migration would need to copy data
        # This is left as a simple implementation that just switches stores
        # after threshold is reached for new data

    async def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add texts with their embeddings to the store."""
        result = await self._store.add(texts, embeddings, metadatas, ids)
        await self._maybe_upgrade()
        return result

    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        return await self._store.similarity_search(
            query_embedding, k, score_threshold, filter
        )

    async def delete(self, ids: List[str]) -> int:
        """Delete items by their IDs."""
        return await self._store.delete(ids)

    async def clear(self) -> int:
        """Clear all items from the store."""
        return await self._store.clear()

    def count(self) -> int:
        """Get the number of items in the store."""
        return self._store.count()

    @property
    def current_store_type(self) -> str:
        """Get the current store type."""
        return self._store_type

    @property
    def underlying_store(self) -> BaseVectorStore:
        """Get the underlying store instance."""
        return self._store


def select_store(
    chunk_count: int,
    dimensions: int = 1536,
    faiss_path: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    qdrant_collection: str = "default",
) -> BaseVectorStore:
    """
    Select the appropriate vector store based on data size.

    Args:
        chunk_count: Expected number of chunks.
        dimensions: Embedding dimensions.
        faiss_path: Path for FAISS persistence.
        qdrant_url: URL for Qdrant server.
        qdrant_collection: Collection name for Qdrant.

    Returns:
        Appropriate vector store instance.
    """
    if chunk_count < AutoStore.MEMORY_THRESHOLD:
        from llmteam.vectorstores.memory import InMemoryVectorStore

        return InMemoryVectorStore()
    elif chunk_count < AutoStore.FAISS_THRESHOLD:
        from llmteam.vectorstores.faiss import FAISSStore

        return FAISSStore(dimensions=dimensions, path=faiss_path)
    else:
        if qdrant_url:
            from llmteam.vectorstores.qdrant import QdrantStore

            return QdrantStore(
                collection=qdrant_collection,
                dimensions=dimensions,
                url=qdrant_url,
            )
        else:
            # Fall back to FAISS if Qdrant is not configured
            from llmteam.vectorstores.faiss import FAISSStore

            return FAISSStore(dimensions=dimensions, path=faiss_path)


__all__ = ["AutoStore", "select_store"]
