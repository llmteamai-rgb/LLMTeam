"""
Agent runtime state.

Tracks agent execution state for pause/resume functionality.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from llmteam.agents.types import AgentStatus


@dataclass
class AgentState:
    """
    Runtime state of an agent.

    Created on start, updated during execution.
    Used for pause/resume.
    """

    # Identity
    agent_id: str
    run_id: str

    # Status
    status: AgentStatus = AgentStatus.IDLE

    # Input
    input_data: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)  # Context from mailbox

    # Progress
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Intermediate
    intermediate_results: List[Dict[str, Any]] = field(default_factory=list)

    # Error
    error: Optional[str] = None
    error_type: Optional[str] = None
    retry_count: int = 0

    # Metrics
    tokens_used: int = 0
    latency_ms: int = 0

    def is_terminal(self) -> bool:
        """Check if agent has finished."""
        return self.status in (AgentStatus.COMPLETED, AgentStatus.FAILED)

    def mark_started(self) -> None:
        """Mark execution start."""
        self.status = AgentStatus.RUNNING
        self.started_at = datetime.utcnow()

    def mark_completed(self) -> None:
        """Mark successful completion."""
        self.status = AgentStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        if self.started_at:
            self.latency_ms = int(
                (self.completed_at - self.started_at).total_seconds() * 1000
            )

    def mark_failed(self, error: Exception) -> None:
        """Mark failure."""
        self.status = AgentStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error = str(error)
        self.error_type = type(error).__name__

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for snapshot."""
        return {
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "status": self.status.value,
            "input_data": self.input_data,
            "context": self.context,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "intermediate_results": self.intermediate_results,
            "error": self.error,
            "error_type": self.error_type,
            "retry_count": self.retry_count,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        """Deserialize from snapshot."""
        state = cls(
            agent_id=data["agent_id"],
            run_id=data["run_id"],
        )
        state.status = AgentStatus(data["status"])
        state.input_data = data.get("input_data", {})
        state.context = data.get("context", {})
        state.intermediate_results = data.get("intermediate_results", [])
        state.error = data.get("error")
        state.error_type = data.get("error_type")
        state.retry_count = data.get("retry_count", 0)
        state.tokens_used = data.get("tokens_used", 0)
        state.latency_ms = data.get("latency_ms", 0)

        if data.get("started_at"):
            state.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            state.completed_at = datetime.fromisoformat(data["completed_at"])

        return state
