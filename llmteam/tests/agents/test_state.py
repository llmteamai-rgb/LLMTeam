"""Tests for llmteam.agents.state module."""

import pytest
from datetime import datetime
from llmteam.agents.state import AgentState
from llmteam.agents.types import AgentStatus


class TestAgentState:
    """Tests for AgentState dataclass."""

    def test_create_state(self):
        """Test creating AgentState."""
        state = AgentState(
            agent_id="writer",
            run_id="run-123",
            input_data={"query": "test"},
        )
        assert state.agent_id == "writer"
        assert state.run_id == "run-123"
        assert state.input_data["query"] == "test"
        assert state.status == AgentStatus.IDLE

    def test_default_values(self):
        """Test default values."""
        state = AgentState(agent_id="writer", run_id="run-123")
        assert state.input_data == {}
        assert state.context == {}
        assert state.started_at is None
        assert state.completed_at is None
        assert state.retry_count == 0
        assert state.tokens_used == 0
        assert state.latency_ms == 0

    def test_mark_started(self):
        """Test mark_started sets status and timestamp."""
        state = AgentState(agent_id="writer", run_id="run-123")
        state.mark_started()

        assert state.status == AgentStatus.RUNNING
        assert state.started_at is not None
        assert isinstance(state.started_at, datetime)

    def test_mark_completed(self):
        """Test mark_completed sets status and timestamp."""
        state = AgentState(agent_id="writer", run_id="run-123")
        state.mark_started()
        state.mark_completed()

        assert state.status == AgentStatus.COMPLETED
        assert state.completed_at is not None
        assert isinstance(state.completed_at, datetime)

    def test_mark_failed(self):
        """Test mark_failed sets status, timestamp and error."""
        state = AgentState(agent_id="writer", run_id="run-123")
        state.mark_started()
        state.mark_failed(Exception("Something went wrong"))

        assert state.status == AgentStatus.FAILED
        assert state.completed_at is not None
        assert state.error == "Something went wrong"
        assert state.error_type == "Exception"

    def test_to_dict(self):
        """Test serialization to dict."""
        state = AgentState(
            agent_id="writer",
            run_id="run-123",
            input_data={"query": "test"},
            context={"key": "value"},
        )
        state.mark_started()

        data = state.to_dict()

        assert data["agent_id"] == "writer"
        assert data["run_id"] == "run-123"
        assert data["input_data"]["query"] == "test"
        assert data["context"]["key"] == "value"
        assert data["status"] == "running"
        assert "started_at" in data

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "agent_id": "writer",
            "run_id": "run-123",
            "input_data": {"query": "test"},
            "context": {},
            "status": "completed",
            "started_at": "2024-01-01T10:00:00",
            "completed_at": "2024-01-01T10:01:00",
            "error": None,
            "retry_count": 1,
            "tokens_used": 100,
            "latency_ms": 500,
        }

        state = AgentState.from_dict(data)

        assert state.agent_id == "writer"
        assert state.run_id == "run-123"
        assert state.status == AgentStatus.COMPLETED
        assert state.retry_count == 1
        assert state.tokens_used == 100
        assert state.latency_ms == 500

    def test_lifecycle_flow(self):
        """Test full lifecycle: idle -> running -> completed."""
        state = AgentState(agent_id="writer", run_id="run-123")

        # Initial state
        assert state.status == AgentStatus.IDLE

        # Start
        state.mark_started()
        assert state.status == AgentStatus.RUNNING
        assert state.started_at is not None

        # Complete
        state.mark_completed()
        assert state.status == AgentStatus.COMPLETED
        assert state.completed_at is not None
        assert state.completed_at >= state.started_at

    def test_is_terminal(self):
        """Test is_terminal check."""
        state = AgentState(agent_id="writer", run_id="run-123")

        # Idle is not terminal
        assert state.is_terminal() is False

        # Running is not terminal
        state.mark_started()
        assert state.is_terminal() is False

        # Completed is terminal
        state.mark_completed()
        assert state.is_terminal() is True

    def test_is_terminal_on_failed(self):
        """Test is_terminal returns True for failed state."""
        state = AgentState(agent_id="writer", run_id="run-123")
        state.mark_started()
        state.mark_failed(ValueError("Error"))

        assert state.is_terminal() is True

    def test_intermediate_results(self):
        """Test intermediate results storage."""
        state = AgentState(agent_id="writer", run_id="run-123")
        assert state.intermediate_results == []

        state.intermediate_results.append({"step": 1, "output": "result1"})
        state.intermediate_results.append({"step": 2, "output": "result2"})

        assert len(state.intermediate_results) == 2
        assert state.intermediate_results[0]["step"] == 1

    def test_latency_calculated_on_complete(self):
        """Test latency is calculated when mark_completed is called."""
        import time

        state = AgentState(agent_id="writer", run_id="run-123")
        state.mark_started()

        # Small delay to ensure measurable latency
        time.sleep(0.01)

        state.mark_completed()

        # Latency should be positive
        assert state.latency_ms > 0
