"""
Tests for InMemoryVectorStore.
"""

import pytest
import numpy as np

from llmteam.vectorstores import InMemoryVectorStore, SearchResult


class TestInMemoryVectorStore:
    """Tests for InMemoryVectorStore."""

    @pytest.mark.asyncio
    async def test_add_single_item(self):
        """Test adding a single item."""
        store = InMemoryVectorStore()

        ids = await store.add(
            texts=["Hello world"],
            embeddings=[[0.1, 0.2, 0.3]],
            metadatas=[{"source": "test"}],
        )

        assert len(ids) == 1
        assert store.count() == 1

    @pytest.mark.asyncio
    async def test_add_multiple_items(self):
        """Test adding multiple items."""
        store = InMemoryVectorStore()

        ids = await store.add(
            texts=["First", "Second", "Third"],
            embeddings=[
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ],
        )

        assert len(ids) == 3
        assert store.count() == 3

    @pytest.mark.asyncio
    async def test_add_with_custom_ids(self):
        """Test adding items with custom IDs."""
        store = InMemoryVectorStore()

        ids = await store.add(
            texts=["Test"],
            embeddings=[[0.1, 0.2, 0.3]],
            ids=["custom-id-1"],
        )

        assert ids == ["custom-id-1"]

    @pytest.mark.asyncio
    async def test_similarity_search_basic(self):
        """Test basic similarity search."""
        store = InMemoryVectorStore()

        await store.add(
            texts=["apple", "banana", "cherry"],
            embeddings=[
                [1.0, 0.0, 0.0],  # apple
                [0.0, 1.0, 0.0],  # banana
                [0.0, 0.0, 1.0],  # cherry
            ],
        )

        # Search for something similar to apple
        results = await store.similarity_search(
            query_embedding=[0.9, 0.1, 0.0],
            k=2,
        )

        assert len(results) == 2
        assert results[0].text == "apple"  # Most similar
        assert results[0].score > results[1].score  # Scores are sorted

    @pytest.mark.asyncio
    async def test_similarity_search_with_threshold(self):
        """Test similarity search with score threshold."""
        store = InMemoryVectorStore()

        await store.add(
            texts=["close", "far"],
            embeddings=[
                [0.9, 0.1, 0.0],  # close to query
                [0.0, 0.0, 1.0],  # far from query
            ],
        )

        results = await store.similarity_search(
            query_embedding=[1.0, 0.0, 0.0],
            k=10,
            score_threshold=0.8,  # Only very similar
        )

        assert len(results) == 1
        assert results[0].text == "close"

    @pytest.mark.asyncio
    async def test_similarity_search_with_filter(self):
        """Test similarity search with metadata filter."""
        store = InMemoryVectorStore()

        await store.add(
            texts=["doc1", "doc2", "doc3"],
            embeddings=[
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            metadatas=[
                {"category": "A"},
                {"category": "B"},
                {"category": "A"},
            ],
        )

        results = await store.similarity_search(
            query_embedding=[1.0, 0.0, 0.0],
            k=10,
            filter={"category": "A"},
        )

        assert len(results) == 2
        for r in results:
            assert r.metadata["category"] == "A"

    @pytest.mark.asyncio
    async def test_similarity_search_k_limit(self):
        """Test that k limits results."""
        store = InMemoryVectorStore()

        await store.add(
            texts=["a", "b", "c", "d", "e"],
            embeddings=[
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
        )

        results = await store.similarity_search(
            query_embedding=[1.0, 0.0, 0.0],
            k=3,
        )

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_delete_by_ids(self):
        """Test deleting items by IDs."""
        store = InMemoryVectorStore()

        ids = await store.add(
            texts=["a", "b", "c"],
            embeddings=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )

        assert store.count() == 3

        deleted = await store.delete([ids[0], ids[1]])
        assert deleted == 2
        assert store.count() == 1

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing all items."""
        store = InMemoryVectorStore()

        await store.add(
            texts=["a", "b", "c"],
            embeddings=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )

        assert store.count() == 3

        cleared = await store.clear()
        assert cleared == 3
        assert store.count() == 0

    @pytest.mark.asyncio
    async def test_get_by_ids(self):
        """Test getting items by IDs."""
        store = InMemoryVectorStore()

        ids = await store.add(
            texts=["first", "second"],
            embeddings=[[1, 0, 0], [0, 1, 0]],
            metadatas=[{"n": 1}, {"n": 2}],
        )

        results = await store.get_by_ids([ids[0]])
        assert len(results) == 1
        assert results[0].text == "first"
        assert results[0].id == ids[0]

    @pytest.mark.asyncio
    async def test_empty_search(self):
        """Test searching empty store."""
        store = InMemoryVectorStore()

        results = await store.similarity_search(
            query_embedding=[1.0, 0.0, 0.0],
            k=5,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_mismatched_lengths_error(self):
        """Test error on mismatched lengths."""
        store = InMemoryVectorStore()

        with pytest.raises(ValueError, match="same length"):
            await store.add(
                texts=["a", "b"],
                embeddings=[[1, 0, 0]],  # Only 1 embedding for 2 texts
            )

    @pytest.mark.asyncio
    async def test_cosine_similarity_correctness(self):
        """Test that cosine similarity is calculated correctly."""
        store = InMemoryVectorStore()

        # Add a unit vector
        await store.add(
            texts=["unit"],
            embeddings=[[1.0, 0.0, 0.0]],
        )

        # Search with the same vector should give similarity ~1.0
        results = await store.similarity_search(
            query_embedding=[1.0, 0.0, 0.0],
            k=1,
        )

        assert len(results) == 1
        assert abs(results[0].score - 1.0) < 0.001  # Very close to 1.0

        # Search with orthogonal vector should give similarity ~0.0
        results = await store.similarity_search(
            query_embedding=[0.0, 1.0, 0.0],
            k=1,
        )

        assert len(results) == 1
        assert abs(results[0].score) < 0.001  # Very close to 0.0
