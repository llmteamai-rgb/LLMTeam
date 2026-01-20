"""Tests for llmteam.agents.config module."""

import pytest
from llmteam.agents.types import AgentType, AgentMode
from llmteam.agents.config import (
    AgentConfig,
    LLMAgentConfig,
    RAGAgentConfig,
    KAGAgentConfig,
)


class TestAgentConfig:
    """Tests for AgentConfig base class."""

    def test_create_config(self):
        """Test creating basic AgentConfig."""
        config = AgentConfig(type=AgentType.LLM, role="writer")
        assert config.type == AgentType.LLM
        assert config.role == "writer"
        assert config.id == "writer"  # Default: id = role
        assert config.name == "Writer"  # Auto-generated

    def test_role_required(self):
        """Test role is required."""
        with pytest.raises(ValueError, match="role is required"):
            AgentConfig(type=AgentType.LLM, role="")

    def test_custom_id(self):
        """Test custom id overrides default."""
        config = AgentConfig(type=AgentType.LLM, role="writer", id="custom_id")
        assert config.id == "custom_id"
        assert config.role == "writer"

    def test_custom_name(self):
        """Test custom name overrides auto-generated."""
        config = AgentConfig(type=AgentType.LLM, role="writer", name="My Writer")
        assert config.name == "My Writer"

    def test_name_from_role(self):
        """Test name is auto-generated from role."""
        config = AgentConfig(type=AgentType.LLM, role="data_processor")
        assert config.name == "Data Processor"

    def test_metadata_and_tags(self):
        """Test metadata and tags."""
        config = AgentConfig(
            type=AgentType.LLM,
            role="writer",
            tags=["production", "v2"],
            metadata={"version": "2.0"},
        )
        assert "production" in config.tags
        assert config.metadata["version"] == "2.0"


class TestLLMAgentConfig:
    """Tests for LLMAgentConfig."""

    def test_default_values(self):
        """Test LLMAgentConfig default values."""
        config = LLMAgentConfig(role="writer", prompt="Write: {input}")
        assert config.type == AgentType.LLM
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.use_context is True

    def test_custom_model(self):
        """Test custom model setting."""
        config = LLMAgentConfig(
            role="writer",
            prompt="Write: {input}",
            model="gpt-4o",
            temperature=0.3,
        )
        assert config.model == "gpt-4o"
        assert config.temperature == 0.3

    def test_prompt_required(self):
        """Test prompt can be empty (validated elsewhere)."""
        config = LLMAgentConfig(role="writer", prompt="")
        assert config.prompt == ""

    def test_output_settings(self):
        """Test output configuration."""
        config = LLMAgentConfig(
            role="writer",
            prompt="Write: {input}",
            output_key="result",
            output_format="json",
        )
        assert config.output_key == "result"
        assert config.output_format == "json"

    def test_system_prompt(self):
        """Test system prompt."""
        config = LLMAgentConfig(
            role="writer",
            prompt="Write: {input}",
            system_prompt="You are a helpful assistant.",
        )
        assert config.system_prompt == "You are a helpful assistant."


class TestRAGAgentConfig:
    """Tests for RAGAgentConfig."""

    def test_default_values(self):
        """Test RAGAgentConfig default values."""
        config = RAGAgentConfig(role="retriever")
        assert config.type == AgentType.RAG
        assert config.mode == AgentMode.NATIVE
        assert config.collection == "default"
        assert config.top_k == 5
        assert config.score_threshold == 0.7

    def test_custom_collection(self):
        """Test custom collection."""
        config = RAGAgentConfig(role="retriever", collection="documents")
        assert config.collection == "documents"

    def test_retrieval_settings(self):
        """Test retrieval configuration."""
        config = RAGAgentConfig(
            role="retriever",
            top_k=10,
            score_threshold=0.8,
            include_sources=True,
            include_scores=True,
        )
        assert config.top_k == 10
        assert config.score_threshold == 0.8
        assert config.include_sources is True
        assert config.include_scores is True

    def test_proxy_mode(self):
        """Test proxy mode configuration."""
        config = RAGAgentConfig(
            role="retriever",
            mode=AgentMode.PROXY,
            proxy_endpoint="http://rag-service/search",
            proxy_api_key="secret",
        )
        assert config.mode == AgentMode.PROXY
        assert config.proxy_endpoint == "http://rag-service/search"

    def test_deliver_to(self):
        """Test deliver_to for NOT_SHARED mode."""
        config = RAGAgentConfig(role="retriever", deliver_to="writer")
        assert config.deliver_to == "writer"

    def test_filters(self):
        """Test metadata filters."""
        config = RAGAgentConfig(
            role="retriever",
            filters={"category": "tech", "year": 2024},
        )
        assert config.filters["category"] == "tech"


class TestKAGAgentConfig:
    """Tests for KAGAgentConfig."""

    def test_default_values(self):
        """Test KAGAgentConfig default values."""
        config = KAGAgentConfig(role="knowledge")
        assert config.type == AgentType.KAG
        assert config.mode == AgentMode.NATIVE
        assert config.max_hops == 2
        assert config.max_entities == 10

    def test_graph_settings(self):
        """Test graph configuration."""
        config = KAGAgentConfig(
            role="knowledge",
            graph_store="neo4j",
            graph_uri="bolt://localhost:7687",
            graph_user="neo4j",
            graph_password="password",
        )
        assert config.graph_store == "neo4j"
        assert config.graph_uri == "bolt://localhost:7687"

    def test_query_settings(self):
        """Test query configuration."""
        config = KAGAgentConfig(
            role="knowledge",
            max_hops=3,
            max_entities=20,
            include_relations=True,
        )
        assert config.max_hops == 3
        assert config.max_entities == 20
        assert config.include_relations is True

    def test_entity_extraction(self):
        """Test entity extraction settings."""
        config = KAGAgentConfig(
            role="knowledge",
            extract_entities=True,
            entity_types=["Person", "Organization", "Location"],
        )
        assert config.extract_entities is True
        assert "Person" in config.entity_types

    def test_proxy_mode(self):
        """Test proxy mode configuration."""
        config = KAGAgentConfig(
            role="knowledge",
            mode=AgentMode.PROXY,
            proxy_endpoint="http://kag-service/query",
        )
        assert config.mode == AgentMode.PROXY
        assert config.proxy_endpoint == "http://kag-service/query"
