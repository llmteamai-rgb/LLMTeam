"""Tests for llmteam.agents.result module."""

import pytest
from llmteam.agents.result import AgentResult, RAGResult, KAGResult
from llmteam.agents.types import AgentType


class TestAgentResult:
    """Tests for AgentResult dataclass."""

    def test_create_result(self):
        """Test creating AgentResult."""
        result = AgentResult(output={"response": "Hello"})
        assert result.output["response"] == "Hello"
        assert result.success is True
        assert result.should_escalate is False

    def test_default_values(self):
        """Test default values."""
        result = AgentResult(output={})
        assert result.success is True
        assert result.should_escalate is False
        assert result.escalation_reason is None
        assert result.error is None
        assert result.context_payload is None
        assert result.tokens_used == 0
        assert result.latency_ms == 0

    def test_failed_result(self):
        """Test creating failed result."""
        result = AgentResult(
            output={},
            success=False,
            error="Processing failed",
        )
        assert result.success is False
        assert result.error == "Processing failed"

    def test_escalation(self):
        """Test escalation flags."""
        result = AgentResult(
            output={"response": "Need help"},
            should_escalate=True,
            escalation_reason="Confidence too low",
        )
        assert result.should_escalate is True
        assert result.escalation_reason == "Confidence too low"

    def test_context_payload(self):
        """Test context payload for sharing between agents."""
        result = AgentResult(
            output={"answer": "42"},
            context_payload={"sources": ["doc1", "doc2"]},
        )
        assert result.context_payload["sources"] == ["doc1", "doc2"]

    def test_metrics(self):
        """Test usage metrics."""
        result = AgentResult(
            output={"response": "Hello"},
            tokens_used=150,
            latency_ms=1234,
        )
        assert result.tokens_used == 150
        assert result.latency_ms == 1234

    def test_agent_type_default(self):
        """Test agent_type defaults to LLM."""
        result = AgentResult(output={})
        assert result.agent_type == AgentType.LLM

    def test_to_dict(self):
        """Test serialization to dict."""
        result = AgentResult(
            output={"answer": "42"},
            agent_id="writer",
            success=True,
            tokens_used=100,
        )
        data = result.to_dict()
        assert data["output"] == {"answer": "42"}
        assert data["agent_id"] == "writer"
        assert data["success"] is True
        assert data["tokens_used"] == 100


class TestRAGResult:
    """Tests for RAGResult dataclass."""

    def test_create_rag_result(self):
        """Test creating RAGResult."""
        result = RAGResult(
            output={"summary": "Found documents"},
            documents=[
                {"content": "doc1", "score": 0.95},
                {"content": "doc2", "score": 0.87},
            ],
            query="search query",
        )
        assert len(result.documents) == 2
        assert result.query == "search query"

    def test_auto_context_payload(self):
        """Test _rag_context is auto-added to context_payload."""
        result = RAGResult(
            output={"summary": "found docs"},
            documents=[{"content": "doc1", "score": 0.9}],
            query="test query",
        )
        # RAGResult should populate context_payload with _rag_context
        assert result.context_payload is not None
        assert "_rag_context" in result.context_payload
        assert result.context_payload["rag_query"] == "test query"
        assert result.context_payload["rag_count"] == 1

    def test_default_empty_lists(self):
        """Test default empty lists."""
        result = RAGResult(output={})
        assert result.documents == []
        assert result.sources == []
        assert result.query == ""
        assert result.total_found == 0

    def test_inherits_from_agent_result(self):
        """Test RAGResult inherits AgentResult fields."""
        result = RAGResult(
            output={"data": "test"},
            success=True,
            tokens_used=100,
            documents=[{"content": "doc"}],
        )
        assert result.success is True
        assert result.tokens_used == 100

    def test_agent_type_is_rag(self):
        """Test agent_type is RAG."""
        result = RAGResult(output={})
        assert result.agent_type == AgentType.RAG

    def test_context_payload_not_set_when_no_documents(self):
        """Test context_payload stays None when no documents."""
        result = RAGResult(output={}, documents=[])
        assert result.context_payload is None


class TestKAGResult:
    """Tests for KAGResult dataclass."""

    def test_create_kag_result(self):
        """Test creating KAGResult."""
        result = KAGResult(
            output={"knowledge": "facts"},
            entities=[{"name": "Entity1"}, {"name": "Entity2"}],
            relations=[{"from": "Entity1", "rel": "knows", "to": "Entity2"}],
            query_entities=["Entity1"],
        )
        assert len(result.entities) == 2
        assert len(result.relations) == 1
        assert result.query_entities == ["Entity1"]

    def test_auto_context_payload(self):
        """Test _kag_context is auto-added to context_payload."""
        result = KAGResult(
            output={"knowledge": "facts"},
            entities=[{"name": "Entity1"}],
            relations=[{"from": "E1", "rel": "rel", "to": "E2"}],
            query_entities=["E1"],
        )
        # KAGResult should populate context_payload with _kag_context
        assert result.context_payload is not None
        assert "_kag_context" in result.context_payload
        assert "kag_relations" in result.context_payload
        assert "kag_entities" in result.context_payload

    def test_default_empty_lists(self):
        """Test default empty lists."""
        result = KAGResult(output={})
        assert result.entities == []
        assert result.relations == []
        assert result.query_entities == []

    def test_inherits_from_agent_result(self):
        """Test KAGResult inherits AgentResult fields."""
        result = KAGResult(
            output={"data": "test"},
            success=True,
            should_escalate=True,
            entities=[{"name": "E1"}],
        )
        assert result.success is True
        assert result.should_escalate is True

    def test_agent_type_is_kag(self):
        """Test agent_type is KAG."""
        result = KAGResult(output={})
        assert result.agent_type == AgentType.KAG

    def test_context_payload_not_set_when_no_entities(self):
        """Test context_payload stays None when no entities."""
        result = KAGResult(output={}, entities=[])
        assert result.context_payload is None
