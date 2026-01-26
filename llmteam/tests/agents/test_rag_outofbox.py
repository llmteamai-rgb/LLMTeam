"""
Tests for RAGAgent "out of the box" functionality (RFC-025).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from llmteam.agents import RAGAgent, RAGAgentConfig


class TestRAGAgentOutOfBox:
    """Tests for RAGAgent out-of-the-box functionality."""

    def test_uses_builtin_store_with_texts(self):
        """Test _uses_builtin_store returns True with texts."""
        config = RAGAgentConfig(
            role="rag",
            texts=["Some text content"],
        )

        # Create mock team
        mock_team = MagicMock()

        agent = RAGAgent(team=mock_team, config=config)
        assert agent._uses_builtin_store() is True

    def test_uses_builtin_store_with_documents(self):
        """Test _uses_builtin_store returns True with documents."""
        config = RAGAgentConfig(
            role="rag",
            documents=["file.txt"],
        )

        mock_team = MagicMock()
        agent = RAGAgent(team=mock_team, config=config)
        assert agent._uses_builtin_store() is True

    def test_uses_builtin_store_with_chunks(self):
        """Test _uses_builtin_store returns True with chunks."""
        config = RAGAgentConfig(
            role="rag",
            chunks=["pre-chunked text"],
        )

        mock_team = MagicMock()
        agent = RAGAgent(team=mock_team, config=config)
        assert agent._uses_builtin_store() is True

    def test_uses_builtin_store_with_directory(self):
        """Test _uses_builtin_store returns True with directory."""
        config = RAGAgentConfig(
            role="rag",
            directory="/some/path",
        )

        mock_team = MagicMock()
        agent = RAGAgent(team=mock_team, config=config)
        assert agent._uses_builtin_store() is True

    def test_uses_builtin_store_external_mode(self):
        """Test _uses_builtin_store returns False with external store."""
        config = RAGAgentConfig(
            role="rag",
            vector_store="chroma",
            collection="docs",
        )

        mock_team = MagicMock()
        agent = RAGAgent(team=mock_team, config=config)
        assert agent._uses_builtin_store() is False

    def test_document_count_before_init(self):
        """Test document_count returns 0 before initialization."""
        config = RAGAgentConfig(
            role="rag",
            texts=["Some text"],
        )

        mock_team = MagicMock()
        agent = RAGAgent(team=mock_team, config=config)
        assert agent.document_count == 0

    @pytest.mark.asyncio
    async def test_initialize_with_texts(self, tmp_path):
        """Test initialization with raw texts."""
        config = RAGAgentConfig(
            role="rag",
            texts=["Tesla was founded in 2003 by engineers."],
            chunk_size=500,
            chunk_overlap=50,
        )

        mock_team = MagicMock()
        agent = RAGAgent(team=mock_team, config=config)

        # Mock the OpenAI embedding by patching the class after import
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.AsyncOpenAI") as mock_openai:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.data = [MagicMock(index=0, embedding=[0.1] * 1536)]
                mock_client.embeddings.create = AsyncMock(return_value=mock_response)
                mock_openai.return_value = mock_client

                await agent.initialize()

        assert agent._initialized is True
        assert agent.document_count > 0

    @pytest.mark.asyncio
    async def test_initialize_with_document_file(self, tmp_path):
        """Test initialization with document files."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is test content for RAG.")

        config = RAGAgentConfig(
            role="rag",
            documents=[str(test_file)],
            chunk_size=500,
            chunk_overlap=50,
        )

        mock_team = MagicMock()
        agent = RAGAgent(team=mock_team, config=config)

        # Mock the OpenAI embedding
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.AsyncOpenAI") as mock_openai:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.data = [MagicMock(index=0, embedding=[0.1] * 1536)]
                mock_client.embeddings.create = AsyncMock(return_value=mock_response)
                mock_openai.return_value = mock_client

                await agent.initialize()

        assert agent._initialized is True
        assert agent.document_count > 0

    @pytest.mark.asyncio
    async def test_query_triggers_lazy_init(self, tmp_path):
        """Test that query() triggers lazy initialization."""
        config = RAGAgentConfig(
            role="rag",
            texts=["Test content for querying."],
        )

        mock_team = MagicMock()
        agent = RAGAgent(team=mock_team, config=config)

        assert agent._initialized is False

        # Mock the OpenAI embedding
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.AsyncOpenAI") as mock_openai:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                # Return embeddings for both the document and query
                mock_response.data = [MagicMock(index=0, embedding=[0.1] * 1536)]
                mock_client.embeddings.create = AsyncMock(return_value=mock_response)
                mock_openai.return_value = mock_client

                result = await agent.query("What is the content?")

        assert agent._initialized is True

    @pytest.mark.asyncio
    async def test_query_returns_rag_result(self, tmp_path):
        """Test that query returns RAGResult."""
        config = RAGAgentConfig(
            role="rag",
            texts=["The sky is blue."],
        )

        mock_team = MagicMock()
        agent = RAGAgent(team=mock_team, config=config)

        # Mock the OpenAI embedding
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.AsyncOpenAI") as mock_openai:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.data = [MagicMock(index=0, embedding=[0.1] * 1536)]
                mock_client.embeddings.create = AsyncMock(return_value=mock_response)
                mock_openai.return_value = mock_client

                result = await agent.query("What color is the sky?")

        assert result.success is True
        assert result.query == "What color is the sky?"
        assert isinstance(result.documents, list)

    @pytest.mark.asyncio
    async def test_add_documents_method(self, tmp_path):
        """Test add_documents method."""
        # Create test files
        file1 = tmp_path / "doc1.txt"
        file1.write_text("First document content.")

        config = RAGAgentConfig(
            role="rag",
            texts=["Initial text."],
        )

        mock_team = MagicMock()
        agent = RAGAgent(team=mock_team, config=config)

        # Mock the OpenAI embedding
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.AsyncOpenAI") as mock_openai:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.data = [MagicMock(index=0, embedding=[0.1] * 1536)]
                mock_client.embeddings.create = AsyncMock(return_value=mock_response)
                mock_openai.return_value = mock_client

                # Initialize first
                await agent.initialize()
                initial_count = agent.document_count

                # Add more documents
                added = await agent.add_documents(documents=[str(file1)])

        assert added > 0
        assert agent.document_count > initial_count

    @pytest.mark.asyncio
    async def test_add_documents_fails_for_external_store(self):
        """Test add_documents raises error for external store mode."""
        config = RAGAgentConfig(
            role="rag",
            vector_store="chroma",
        )

        mock_team = MagicMock()
        agent = RAGAgent(team=mock_team, config=config)

        with pytest.raises(ValueError, match="only works with built-in store"):
            await agent.add_documents(texts=["New text"])

    def test_config_chunk_settings(self):
        """Test that chunk settings are configurable."""
        config = RAGAgentConfig(
            role="rag",
            texts=["Test"],
            chunk_size=1000,
            chunk_overlap=100,
        )

        mock_team = MagicMock()
        agent = RAGAgent(team=mock_team, config=config)

        assert agent.chunk_size == 1000
        assert agent.chunk_overlap == 100

    def test_config_embedding_model(self):
        """Test that embedding model is configurable."""
        config = RAGAgentConfig(
            role="rag",
            texts=["Test"],
            embedding_model="text-embedding-3-large",
        )

        mock_team = MagicMock()
        agent = RAGAgent(team=mock_team, config=config)

        assert agent.embedding_model == "text-embedding-3-large"

    @pytest.mark.asyncio
    async def test_initialize_with_chunks(self):
        """Test initialization with pre-chunked texts."""
        config = RAGAgentConfig(
            role="rag",
            chunks=["Chunk 1", "Chunk 2", "Chunk 3"],
        )

        mock_team = MagicMock()
        agent = RAGAgent(team=mock_team, config=config)

        # Mock the OpenAI embedding
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.AsyncOpenAI") as mock_openai:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.data = [
                    MagicMock(index=0, embedding=[0.1] * 1536),
                    MagicMock(index=1, embedding=[0.2] * 1536),
                    MagicMock(index=2, embedding=[0.3] * 1536),
                ]
                mock_client.embeddings.create = AsyncMock(return_value=mock_response)
                mock_openai.return_value = mock_client

                await agent.initialize()

        # Pre-chunked texts should be added directly without re-chunking
        assert agent.document_count == 3

    @pytest.mark.asyncio
    async def test_query_with_custom_top_k(self):
        """Test query with custom top_k parameter."""
        config = RAGAgentConfig(
            role="rag",
            texts=["Doc 1", "Doc 2", "Doc 3"],
            top_k=5,  # Default
        )

        mock_team = MagicMock()
        agent = RAGAgent(team=mock_team, config=config)

        # Mock embeddings
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.AsyncOpenAI") as mock_openai:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.data = [
                    MagicMock(index=0, embedding=[0.1] * 1536),
                    MagicMock(index=1, embedding=[0.2] * 1536),
                    MagicMock(index=2, embedding=[0.3] * 1536),
                ]
                mock_client.embeddings.create = AsyncMock(return_value=mock_response)
                mock_openai.return_value = mock_client

                result = await agent.query("test", top_k=1)

        # Should return at most 1 result
        assert len(result.documents) <= 1
