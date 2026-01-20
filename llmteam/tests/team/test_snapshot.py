"""Tests for llmteam.team.snapshot module."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from llmteam.team.snapshot import TeamSnapshot


class TestTeamSnapshot:
    """Tests for TeamSnapshot dataclass."""

    def test_create_snapshot(self):
        """Test creating TeamSnapshot."""
        snapshot = TeamSnapshot(
            snapshot_id="snap-123",
            team_id="test_team",
            run_id="run-123",
            current_agent="writer",
            input_data={"query": "test"},
            agent_outputs={"agent1": {"response": "hello"}},
        )

        assert snapshot.team_id == "test_team"
        assert snapshot.run_id == "run-123"
        assert snapshot.snapshot_id == "snap-123"
        assert snapshot.current_agent == "writer"
        assert snapshot.input_data["query"] == "test"
        assert snapshot.agent_outputs["agent1"]["response"] == "hello"

    def test_default_values(self):
        """Test default values."""
        snapshot = TeamSnapshot(
            snapshot_id="snap-1",
            team_id="test",
            run_id="run-1",
        )

        assert snapshot.input_data == {}
        assert snapshot.agent_outputs == {}
        assert snapshot.mailbox_state == {}
        assert snapshot.completed_agents == []
        assert snapshot.current_agent is None
        assert snapshot.created_at is not None

    def test_mailbox_state_storage(self):
        """Test mailbox_state storage."""
        snapshot = TeamSnapshot(
            snapshot_id="snap-1",
            team_id="test",
            run_id="run-1",
            mailbox_state={"key": "value", "agents_data": {"a": "b"}},
        )

        assert snapshot.mailbox_state["key"] == "value"
        assert "agents_data" in snapshot.mailbox_state

    def test_from_segment_snapshot(self):
        """Test creating TeamSnapshot from SegmentSnapshot."""
        # Mock SegmentSnapshot
        segment_snapshot = MagicMock()
        segment_snapshot.snapshot_id = "snap-123"
        segment_snapshot.run_id = "run-456"
        segment_snapshot.current_step = "agent_step"
        segment_snapshot.completed_steps = ["step1", "step2"]
        segment_snapshot.step_outputs = {"step1": {"output": "result"}}
        segment_snapshot.input_data = {"input": "test"}
        segment_snapshot.created_at = datetime.utcnow()
        segment_snapshot.checksum = "abc123"

        team_snapshot = TeamSnapshot.from_segment_snapshot(
            segment_snapshot,
            team_id="my_team",
        )

        assert team_snapshot.team_id == "my_team"
        assert team_snapshot.run_id == "run-456"
        assert team_snapshot.current_agent == "agent_step"
        assert team_snapshot.completed_agents == ["step1", "step2"]
        assert team_snapshot.input_data == {"input": "test"}

    def test_to_segment_snapshot(self):
        """Test converting to SegmentSnapshot."""
        snapshot = TeamSnapshot(
            snapshot_id="snap-123",
            team_id="test",
            run_id="run-123",
            current_agent="writer",
            completed_agents=["reader"],
            agent_outputs={"reader": {"data": "value"}},
            input_data={"query": "test"},
            checksum="abc",
        )

        segment_snapshot = snapshot.to_segment_snapshot()

        # Should return a SegmentSnapshot-like object
        assert segment_snapshot.segment_id == "test"
        assert segment_snapshot.run_id == "run-123"
        assert segment_snapshot.current_step == "writer"
        assert segment_snapshot.completed_steps == ["reader"]

    def test_to_dict(self):
        """Test serialization to dict."""
        now = datetime.utcnow()
        snapshot = TeamSnapshot(
            snapshot_id="snap-123",
            team_id="test",
            run_id="run-123",
            current_agent="step1",
            completed_agents=["step0"],
            agent_outputs={"step0": {"a": 1}},
            input_data={"b": 2},
            mailbox_state={"c": 3},
            created_at=now,
            checksum="checksum123",
        )

        data = snapshot.to_dict()

        assert data["team_id"] == "test"
        assert data["run_id"] == "run-123"
        assert data["snapshot_id"] == "snap-123"
        assert data["current_agent"] == "step1"
        assert data["completed_agents"] == ["step0"]
        assert data["agent_outputs"] == {"step0": {"a": 1}}
        assert data["input_data"] == {"b": 2}
        assert data["mailbox_state"] == {"c": 3}
        assert "created_at" in data
        assert data["checksum"] == "checksum123"

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "snapshot_id": "snap-999",
            "team_id": "restored",
            "run_id": "run-999",
            "current_agent": "last_step",
            "completed_agents": ["first", "second"],
            "agent_outputs": {"first": {"x": 10}, "second": {"y": 20}},
            "input_data": {"z": 30},
            "mailbox_state": {"key": "val"},
            "created_at": "2024-01-01T10:00:00",
            "checksum": "abc123",
        }

        snapshot = TeamSnapshot.from_dict(data)

        assert snapshot.team_id == "restored"
        assert snapshot.run_id == "run-999"
        assert snapshot.snapshot_id == "snap-999"
        assert snapshot.current_agent == "last_step"
        assert snapshot.completed_agents == ["first", "second"]
        assert snapshot.agent_outputs == {"first": {"x": 10}, "second": {"y": 20}}

    def test_roundtrip_serialization(self):
        """Test to_dict -> from_dict roundtrip."""
        original = TeamSnapshot(
            snapshot_id="snap-123",
            team_id="test",
            run_id="run-123",
            current_agent="step3",
            completed_agents=["step1", "step2"],
            agent_outputs={"step1": {"key": "value"}},
            input_data={"result": "success"},
            mailbox_state={"agents_called": ["a", "b"]},
        )

        data = original.to_dict()
        restored = TeamSnapshot.from_dict(data)

        assert restored.team_id == original.team_id
        assert restored.run_id == original.run_id
        assert restored.snapshot_id == original.snapshot_id
        assert restored.current_agent == original.current_agent
        assert restored.completed_agents == original.completed_agents
        assert restored.agent_outputs == original.agent_outputs
        assert restored.input_data == original.input_data
        assert restored.mailbox_state == original.mailbox_state

    def test_compute_checksum(self):
        """Test compute_checksum produces consistent results."""
        snapshot = TeamSnapshot(
            snapshot_id="snap-1",
            team_id="test",
            run_id="run-1",
            current_agent="agent1",
            completed_agents=["agent0"],
            agent_outputs={"agent0": {"data": "value"}},
        )

        checksum1 = snapshot.compute_checksum()
        checksum2 = snapshot.compute_checksum()

        assert checksum1 == checksum2
        assert len(checksum1) == 64  # SHA-256 hex digest

    def test_verify_checksum(self):
        """Test verify returns True for valid checksum."""
        snapshot = TeamSnapshot(
            snapshot_id="snap-1",
            team_id="test",
            run_id="run-1",
            current_agent="agent1",
        )
        snapshot.checksum = snapshot.compute_checksum()

        assert snapshot.verify() is True

    def test_verify_fails_on_modified_data(self):
        """Test verify returns False if data was modified."""
        snapshot = TeamSnapshot(
            snapshot_id="snap-1",
            team_id="test",
            run_id="run-1",
            current_agent="agent1",
        )
        snapshot.checksum = snapshot.compute_checksum()

        # Modify data
        snapshot.current_agent = "modified_agent"

        assert snapshot.verify() is False
