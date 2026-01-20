"""Tests for llmteam.team.result module."""

import pytest
from datetime import datetime
from llmteam.team.result import RunResult, RunStatus, ContextMode, TeamResult


class TestRunStatus:
    """Tests for RunStatus enum."""

    def test_status_values(self):
        """Test RunStatus enum values."""
        assert RunStatus.PENDING.value == "pending"
        assert RunStatus.RUNNING.value == "running"
        assert RunStatus.COMPLETED.value == "completed"
        assert RunStatus.PAUSED.value == "paused"
        assert RunStatus.CANCELLED.value == "cancelled"
        assert RunStatus.FAILED.value == "failed"
        assert RunStatus.TIMEOUT.value == "timeout"

    def test_all_statuses_exist(self):
        """Test all expected statuses exist."""
        statuses = [s.value for s in RunStatus]
        assert "pending" in statuses
        assert "running" in statuses
        assert "completed" in statuses
        assert "failed" in statuses


class TestContextMode:
    """Tests for ContextMode enum."""

    def test_context_mode_values(self):
        """Test ContextMode enum values."""
        assert ContextMode.SHARED.value == "shared"
        assert ContextMode.NOT_SHARED.value == "not_shared"

    def test_context_mode_count(self):
        """Test only 2 context modes exist."""
        assert len(ContextMode) == 2


class TestRunResult:
    """Tests for RunResult dataclass."""

    def test_create_successful_result(self):
        """Test creating successful RunResult."""
        result = RunResult(
            success=True,
            status=RunStatus.COMPLETED,
            output={"response": "Hello"},
        )
        assert result.success is True
        assert result.status == RunStatus.COMPLETED
        assert result.output["response"] == "Hello"

    def test_create_failed_result(self):
        """Test creating failed RunResult."""
        result = RunResult(
            success=False,
            status=RunStatus.FAILED,
            error="Agent crashed",
            failed_agent="writer",
        )
        assert result.success is False
        assert result.status == RunStatus.FAILED
        assert result.error == "Agent crashed"
        assert result.failed_agent == "writer"

    def test_default_values(self):
        """Test default values."""
        result = RunResult(success=True)
        assert result.status == RunStatus.COMPLETED
        assert result.output == {}
        assert result.final_output is None
        assert result.agents_called == []
        assert result.iterations == 0
        assert result.duration_ms == 0
        assert result.tokens_used == 0
        assert result.escalations == []

    def test_agents_called(self):
        """Test agents_called tracking."""
        result = RunResult(
            success=True,
            agents_called=["retriever", "writer", "reviewer"],
            iterations=3,
        )
        assert len(result.agents_called) == 3
        assert "writer" in result.agents_called
        assert result.iterations == 3

    def test_escalations(self):
        """Test escalations list."""
        result = RunResult(
            success=True,
            escalations=[
                {"level": "warning", "reason": "Confidence low"},
                {"level": "error", "reason": "Validation failed"},
            ],
        )
        assert len(result.escalations) == 2
        assert result.escalations[0]["level"] == "warning"

    def test_timing(self):
        """Test timing fields."""
        now = datetime.utcnow()
        result = RunResult(
            success=True,
            started_at=now,
            completed_at=now,
            duration_ms=1500,
        )
        assert result.started_at == now
        assert result.completed_at == now
        assert result.duration_ms == 1500

    def test_to_dict(self):
        """Test serialization to dict."""
        result = RunResult(
            success=True,
            status=RunStatus.COMPLETED,
            output={"response": "Hello"},
            agents_called=["agent1"],
            iterations=1,
        )
        data = result.to_dict()

        assert data["success"] is True
        assert data["status"] == "completed"
        assert data["output"]["response"] == "Hello"
        assert data["agents_called"] == ["agent1"]
        assert data["iterations"] == 1

    def test_to_dict_with_timestamps(self):
        """Test serialization with timestamps."""
        now = datetime.utcnow()
        result = RunResult(
            success=True,
            started_at=now,
            completed_at=now,
        )
        data = result.to_dict()

        assert data["started_at"] is not None
        assert data["completed_at"] is not None

    def test_snapshot_id(self):
        """Test snapshot_id for pause/resume."""
        result = RunResult(
            success=True,
            status=RunStatus.PAUSED,
            snapshot_id="snapshot-123",
        )
        assert result.status == RunStatus.PAUSED
        assert result.snapshot_id == "snapshot-123"


class TestTeamResult:
    """Tests for TeamResult alias."""

    def test_team_result_is_run_result(self):
        """Test TeamResult is an alias for RunResult."""
        assert TeamResult is RunResult

    def test_create_team_result(self):
        """Test creating TeamResult (using RunResult)."""
        result = TeamResult(
            success=True,
            output={"data": "test"},
        )
        assert isinstance(result, RunResult)
        assert result.success is True
