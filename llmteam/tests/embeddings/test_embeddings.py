"""
Tests for embeddings module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llmteam.embeddings import BaseEmbedding, OpenAIEmbedding, EmbeddingCache, MODEL_DIMENSIONS


class TestOpenAIEmbedding:
    """Tests for OpenAIEmbedding."""

    def test_dimensions(self):
        """Test dimensions property for different models."""
        # Mock the API key
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            emb_small = OpenAIEmbedding(model="text-embedding-3-small")
            assert emb_small.dimensions == 1536

            emb_large = OpenAIEmbedding(model="text-embedding-3-large")
            assert emb_large.dimensions == 3072

            emb_ada = OpenAIEmbedding(model="text-embedding-ada-002")
            assert emb_ada.dimensions == 1536

    def test_model_name(self):
        """Test model_name property."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            emb = OpenAIEmbedding(model="text-embedding-3-small")
            assert emb.model_name == "text-embedding-3-small"

    def test_invalid_model(self):
        """Test error on invalid model."""
        with pytest.raises(ValueError, match="Unknown model"):
            OpenAIEmbedding(model="invalid-model")

    @pytest.mark.asyncio
    async def test_embed_single_text(self):
        """Test embedding a single text."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            emb = OpenAIEmbedding(model="text-embedding-3-small")

            # Mock the OpenAI client
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(index=0, embedding=[0.1] * 1536)
            ]

            mock_client = AsyncMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            emb._client = mock_client

            result = await emb.embed("Hello world")

            assert len(result) == 1
            assert len(result[0]) == 1536

    @pytest.mark.asyncio
    async def test_embed_multiple_texts(self):
        """Test embedding multiple texts."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            emb = OpenAIEmbedding(model="text-embedding-3-small")

            # Mock the OpenAI client
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(index=0, embedding=[0.1] * 1536),
                MagicMock(index=1, embedding=[0.2] * 1536),
            ]

            mock_client = AsyncMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            emb._client = mock_client

            result = await emb.embed(["Hello", "World"])

            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_embed_query(self):
        """Test embed_query convenience method."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            emb = OpenAIEmbedding(model="text-embedding-3-small")

            # Mock the OpenAI client
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(index=0, embedding=[0.1] * 1536)
            ]

            mock_client = AsyncMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            emb._client = mock_client

            result = await emb.embed_query("Hello world")

            # Should return a single vector, not a list of vectors
            assert len(result) == 1536

    @pytest.mark.asyncio
    async def test_embed_empty_list(self):
        """Test embedding empty list returns empty list."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            emb = OpenAIEmbedding(model="text-embedding-3-small")
            result = await emb.embed([])
            assert result == []


class TestEmbeddingCache:
    """Tests for EmbeddingCache."""

    @pytest.mark.asyncio
    async def test_cache_stores_embeddings(self, tmp_path):
        """Test that cache stores embeddings."""
        # Create mock embedding provider
        mock_embedding = AsyncMock(spec=BaseEmbedding)
        mock_embedding.dimensions = 3
        mock_embedding.model_name = "test-model"
        mock_embedding.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        cache = EmbeddingCache(
            embedding=mock_embedding,
            cache_path=tmp_path / "cache.db",
        )

        # First call should use the underlying embedding
        result1 = await cache.embed("Hello")
        assert mock_embedding.embed.call_count == 1

        # Second call should use cache
        result2 = await cache.embed("Hello")
        assert mock_embedding.embed.call_count == 1  # Not called again

        assert result1 == result2

    @pytest.mark.asyncio
    async def test_cache_different_texts(self, tmp_path):
        """Test that different texts get different embeddings."""
        mock_embedding = AsyncMock(spec=BaseEmbedding)
        mock_embedding.dimensions = 3
        mock_embedding.model_name = "test-model"
        mock_embedding.embed = AsyncMock(side_effect=[
            [[0.1, 0.2, 0.3]],  # First call
            [[0.4, 0.5, 0.6]],  # Second call
        ])

        cache = EmbeddingCache(
            embedding=mock_embedding,
            cache_path=tmp_path / "cache.db",
        )

        result1 = await cache.embed("Hello")
        result2 = await cache.embed("World")

        assert result1 != result2
        assert mock_embedding.embed.call_count == 2

    def test_cache_stats(self, tmp_path):
        """Test cache_stats method."""
        mock_embedding = MagicMock(spec=BaseEmbedding)
        mock_embedding.model_name = "test-model"

        cache = EmbeddingCache(
            embedding=mock_embedding,
            cache_path=tmp_path / "cache.db",
        )

        stats = cache.cache_stats()
        assert "total" in stats
        assert "by_model" in stats
        assert stats["total"] == 0

    def test_clear_cache(self, tmp_path):
        """Test clear_cache method."""
        mock_embedding = MagicMock(spec=BaseEmbedding)
        mock_embedding.model_name = "test-model"

        cache = EmbeddingCache(
            embedding=mock_embedding,
            cache_path=tmp_path / "cache.db",
        )

        # Clear should return 0 for empty cache
        count = cache.clear_cache()
        assert count == 0
