"""
Persistence - Data Models.

Provides data structures for pipeline persistence:
- SnapshotType, PipelinePhase enums
- AgentSnapshot for agent state
- PipelineSnapshot for pipeline state
- RestoreResult for restore operations
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import hashlib
import json


class SnapshotType(Enum):
    """Type of snapshot."""

    AUTO = "auto"  # Automatic (by interval)
    MANUAL = "manual"  # Manual
    CHECKPOINT = "checkpoint"  # At checkpoint
    PAUSE = "pause"  # On pause
    ERROR = "error"  # On error (for recovery)


class PipelinePhase(Enum):
    """Pipeline phase."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    WAITING_HUMAN = "waiting_human"
    WAITING_ACTION = "waiting_action"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentSnapshot:
    """Snapshot of agent state."""

    agent_name: str

    # State
    state: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

    # Messages history
    messages: List[dict] = field(default_factory=list)

    # Execution
    completed_steps: List[str] = field(default_factory=list)
    current_step: str = ""

    # Metrics
    tokens_used: int = 0
    execution_time_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_name": self.agent_name,
            "state": self.state,
            "context": self.context,
            "messages": self.messages,
            "completed_steps": self.completed_steps,
            "current_step": self.current_step,
            "tokens_used": self.tokens_used,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class PipelineSnapshot:
    """Snapshot of pipeline state."""

    snapshot_id: str
    snapshot_type: SnapshotType

    # Identity
    pipeline_id: str
    run_id: str
    tenant_id: str

    # Version (for compatibility)
    pipeline_version: str

    # State
    phase: PipelinePhase

    # Version default
    snapshot_version: str = "1.0"

    # State with defaults
    global_state: Dict[str, Any] = field(default_factory=dict)

    # Agents
    agent_snapshots: Dict[str, AgentSnapshot] = field(default_factory=dict)

    # Execution
    completed_steps: List[str] = field(default_factory=list)
    current_step: str = ""
    next_steps: List[str] = field(default_factory=list)

    # Pending
    pending_actions: List[str] = field(default_factory=list)
    pending_approvals: List[str] = field(default_factory=list)

    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now())
    started_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None

    # Metrics
    total_tokens: int = 0
    total_actions: int = 0
    total_approvals: int = 0

    # Integrity
    checksum: str = ""

    def compute_checksum(self) -> str:
        """Compute checksum for integrity verification."""
        data = {
            "pipeline_id": self.pipeline_id,
            "run_id": self.run_id,
            "phase": self.phase.value,
            "global_state": json.dumps(self.global_state, sort_keys=True),
            "completed_steps": self.completed_steps,
            "current_step": self.current_step,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def verify(self) -> bool:
        """Verify integrity."""
        return self.checksum == self.compute_checksum()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "snapshot_type": self.snapshot_type.value,
            "pipeline_id": self.pipeline_id,
            "run_id": self.run_id,
            "tenant_id": self.tenant_id,
            "pipeline_version": self.pipeline_version,
            "snapshot_version": self.snapshot_version,
            "phase": self.phase.value,
            "global_state": self.global_state,
            "agent_snapshots": {
                name: snap.to_dict() for name, snap in self.agent_snapshots.items()
            },
            "completed_steps": self.completed_steps,
            "current_step": self.current_step,
            "next_steps": self.next_steps,
            "pending_actions": self.pending_actions,
            "pending_approvals": self.pending_approvals,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "paused_at": self.paused_at.isoformat() if self.paused_at else None,
            "total_tokens": self.total_tokens,
            "total_actions": self.total_actions,
            "total_approvals": self.total_approvals,
            "checksum": self.checksum,
        }


@dataclass
class RestoreResult:
    """Result of restore operation."""

    success: bool
    snapshot_id: str
    run_id: str

    # Restored state
    phase: PipelinePhase
    current_step: str

    # Warnings
    warnings: List[str] = field(default_factory=list)

    # What was skipped
    skipped_agents: List[str] = field(default_factory=list)
    skipped_steps: List[str] = field(default_factory=list)
