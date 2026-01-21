"""
Team snapshot for pause/resume.

Wrapper around SegmentSnapshot.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import hashlib
import json


@dataclass
class TeamSnapshot:
    """
    Snapshot for pause/resume (based on SegmentSnapshot).

    Contains all information needed to resume a paused team.
    """

    snapshot_id: str
    team_id: str
    run_id: str

    # State
    current_agent: Optional[str] = None
    completed_agents: List[str] = field(default_factory=list)
    agent_outputs: Dict[str, Any] = field(default_factory=dict)
    input_data: Dict[str, Any] = field(default_factory=dict)

    # Context
    mailbox_state: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    checksum: str = ""

    def compute_checksum(self) -> str:
        """Compute checksum for integrity verification."""
        data = {
            "run_id": self.run_id,
            "team_id": self.team_id,
            "current_agent": self.current_agent,
            "completed_agents": self.completed_agents,
            "agent_outputs": json.dumps(self.agent_outputs, sort_keys=True, default=str),
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def verify(self) -> bool:
        """Verify integrity."""
        return self.checksum == self.compute_checksum()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "team_id": self.team_id,
            "run_id": self.run_id,
            "current_agent": self.current_agent,
            "completed_agents": self.completed_agents,
            "agent_outputs": self.agent_outputs,
            "input_data": self.input_data,
            "mailbox_state": self.mailbox_state,
            "created_at": self.created_at.isoformat(),
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeamSnapshot":
        """Create from dictionary."""
        snapshot = cls(
            snapshot_id=data["snapshot_id"],
            team_id=data["team_id"],
            run_id=data["run_id"],
            current_agent=data.get("current_agent"),
            completed_agents=data.get("completed_agents", []),
            agent_outputs=data.get("agent_outputs", {}),
            input_data=data.get("input_data", {}),
            mailbox_state=data.get("mailbox_state", {}),
            checksum=data.get("checksum", ""),
        )

        if data.get("created_at"):
            snapshot.created_at = datetime.fromisoformat(data["created_at"])

        return snapshot

    @classmethod
    def from_segment_snapshot(cls, segment_snapshot, team_id: str) -> "TeamSnapshot":
        """Create from SegmentSnapshot."""
        return cls(
            snapshot_id=segment_snapshot.snapshot_id,
            team_id=team_id,
            run_id=segment_snapshot.run_id,
            current_agent=segment_snapshot.current_step,
            completed_agents=segment_snapshot.completed_steps,
            agent_outputs=segment_snapshot.step_outputs,
            input_data=segment_snapshot.input_data,
            created_at=segment_snapshot.created_at,
            checksum=segment_snapshot.checksum,
        )

    def to_segment_snapshot(self):
        """Convert to ExecutionSnapshot for ExecutionEngine."""
        from llmteam.engine.engine import ExecutionSnapshot

        return ExecutionSnapshot(
            snapshot_id=self.snapshot_id,
            run_id=self.run_id,
            workflow_id=self.team_id,
            tenant_id="default",  # Will be overridden
            current_step=self.current_agent,
            completed_steps=self.completed_agents,
            step_outputs=self.agent_outputs,
            input_data=self.input_data,
            created_at=self.created_at,
            checksum=self.checksum,
        )
