"""Tests for llmteam.agents.presets module."""

import pytest
from llmteam.agents.presets import (
    create_orchestrator_config,
    create_group_orchestrator_config,
    create_summarizer_config,
    create_reviewer_config,
    create_rag_config,
    create_kag_config,
)


class TestOrchestratorConfig:
    """Tests for orchestrator preset."""

    def test_create_orchestrator_config(self):
        """Test creating orchestrator config."""
        config = create_orchestrator_config(
            available_agents=["writer", "reviewer", "editor"],
            model="gpt-4o",
        )

        assert config["type"] == "llm"
        assert config["role"] == "_orchestrator"
        assert config["model"] == "gpt-4o"
        assert "writer" in config["metadata"]["available_agents"]
        assert "reviewer" in config["metadata"]["available_agents"]
        assert "editor" in config["metadata"]["available_agents"]

    def test_orchestrator_default_model(self):
        """Test orchestrator uses default model."""
        config = create_orchestrator_config(["agent1"])
        assert config["model"] == "gpt-4o-mini"

    def test_orchestrator_prompt_format(self):
        """Test orchestrator prompt includes JSON format."""
        config = create_orchestrator_config(["agent1", "agent2"])
        assert "JSON" in config["prompt"] or "json" in config["prompt"].lower()
        assert "next_agent" in config["prompt"]


class TestGroupOrchestratorConfig:
    """Tests for group orchestrator preset."""

    def test_create_group_orchestrator_config(self):
        """Test creating group orchestrator config."""
        config = create_group_orchestrator_config(
            available_teams=["support", "billing", "technical"],
            model="gpt-4o",
        )

        assert config["type"] == "llm"
        assert config["role"] == "group_orchestrator"  # v4.1: no underscore prefix
        assert config["model"] == "gpt-4o"
        assert "support" in config["metadata"]["available_teams"]
        assert "billing" in config["metadata"]["available_teams"]

    def test_group_orchestrator_output_format(self):
        """Test group orchestrator prompt includes routing fields."""
        config = create_group_orchestrator_config(["team1", "team2"])
        assert "next_team" in config["prompt"]
        assert "should_continue" in config["prompt"]


class TestSummarizerConfig:
    """Tests for summarizer preset."""

    def test_create_summarizer_config(self):
        """Test creating summarizer config."""
        config = create_summarizer_config()

        assert config["type"] == "llm"
        assert config["role"] == "summarizer"
        assert "summarize" in config["prompt"].lower() or "summary" in config["prompt"].lower()

    def test_summarizer_custom_model(self):
        """Test summarizer with custom model."""
        config = create_summarizer_config(model="gpt-4o")
        assert config["model"] == "gpt-4o"


class TestReviewerConfig:
    """Tests for reviewer preset."""

    def test_create_reviewer_config(self):
        """Test creating reviewer config."""
        config = create_reviewer_config()

        assert config["type"] == "llm"
        assert config["role"] == "reviewer"
        assert "review" in config["prompt"].lower()

    def test_reviewer_custom_model(self):
        """Test reviewer with custom model."""
        config = create_reviewer_config(model="gpt-4o")
        assert config["model"] == "gpt-4o"


class TestRAGConfig:
    """Tests for RAG preset."""

    def test_create_rag_config(self):
        """Test creating RAG config."""
        config = create_rag_config(collection="documents")

        assert config["type"] == "rag"
        assert config["collection"] == "documents"

    def test_rag_default_collection(self):
        """Test RAG uses default collection."""
        config = create_rag_config()
        assert config["collection"] == "default"

    def test_rag_custom_settings(self):
        """Test RAG with custom settings."""
        config = create_rag_config(
            collection="tech_docs",
            top_k=10,
            score_threshold=0.8,
        )
        assert config["collection"] == "tech_docs"
        assert config["top_k"] == 10
        assert config["score_threshold"] == 0.8


class TestKAGConfig:
    """Tests for KAG preset."""

    def test_create_kag_config(self):
        """Test creating KAG config."""
        config = create_kag_config()

        assert config["type"] == "kag"

    def test_kag_custom_settings(self):
        """Test KAG with custom settings."""
        config = create_kag_config(
            max_hops=3,
            max_entities=20,
        )
        assert config["max_hops"] == 3
        assert config["max_entities"] == 20
