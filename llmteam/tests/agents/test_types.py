"""Tests for llmteam.agents.types module."""

import pytest
from llmteam.agents.types import AgentType, AgentMode, AgentStatus


class TestAgentType:
    """Tests for AgentType enum."""

    def test_agent_type_values(self):
        """Test AgentType enum values."""
        assert AgentType.LLM.value == "llm"
        assert AgentType.RAG.value == "rag"
        assert AgentType.KAG.value == "kag"

    def test_agent_type_from_string(self):
        """Test creating AgentType from string."""
        assert AgentType("llm") == AgentType.LLM
        assert AgentType("rag") == AgentType.RAG
        assert AgentType("kag") == AgentType.KAG

    def test_agent_type_invalid(self):
        """Test invalid AgentType raises ValueError."""
        with pytest.raises(ValueError):
            AgentType("invalid")

    def test_agent_type_count(self):
        """Test only 3 agent types exist."""
        assert len(AgentType) == 3


class TestAgentMode:
    """Tests for AgentMode enum."""

    def test_agent_mode_values(self):
        """Test AgentMode enum values."""
        assert AgentMode.NATIVE.value == "native"
        assert AgentMode.PROXY.value == "proxy"

    def test_agent_mode_from_string(self):
        """Test creating AgentMode from string."""
        assert AgentMode("native") == AgentMode.NATIVE
        assert AgentMode("proxy") == AgentMode.PROXY

    def test_agent_mode_count(self):
        """Test only 2 modes exist."""
        assert len(AgentMode) == 2


class TestAgentStatus:
    """Tests for AgentStatus enum."""

    def test_agent_status_values(self):
        """Test AgentStatus enum values."""
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.RUNNING.value == "running"
        assert AgentStatus.COMPLETED.value == "completed"
        assert AgentStatus.FAILED.value == "failed"
        assert AgentStatus.WAITING.value == "waiting"

    def test_agent_status_lifecycle(self):
        """Test all lifecycle statuses exist."""
        statuses = [s.value for s in AgentStatus]
        assert "idle" in statuses
        assert "running" in statuses
        assert "completed" in statuses
        assert "failed" in statuses
        assert "waiting" in statuses

    def test_agent_status_from_string(self):
        """Test creating AgentStatus from string."""
        assert AgentStatus("idle") == AgentStatus.IDLE
        assert AgentStatus("running") == AgentStatus.RUNNING
        assert AgentStatus("completed") == AgentStatus.COMPLETED
        assert AgentStatus("failed") == AgentStatus.FAILED
        assert AgentStatus("waiting") == AgentStatus.WAITING

    def test_agent_status_count(self):
        """Test status count."""
        assert len(AgentStatus) == 5
