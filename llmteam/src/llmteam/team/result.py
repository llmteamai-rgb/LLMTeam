"""
Team execution result types.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class RunStatus(Enum):
    """Team run status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ContextMode(Enum):
    """Context sharing mode between agents."""

    SHARED = "shared"  # One mailbox for all LLMAgents
    NOT_SHARED = "not_shared"  # Each LLMAgent has own mailbox


@dataclass
class RunResult:
    """Result of team execution."""

    # Status
    success: bool
    status: RunStatus = RunStatus.COMPLETED

    # Output
    output: Dict[str, Any] = field(default_factory=dict)  # All agent outputs
    final_output: Any = None  # Output of last agent

    # Execution info
    agents_called: List[str] = field(default_factory=list)
    iterations: int = 0
    duration_ms: int = 0
    tokens_used: int = 0

    # Errors
    error: Optional[str] = None
    failed_agent: Optional[str] = None

    # Escalations
    escalations: List[Dict[str, Any]] = field(default_factory=list)

    # For pause/resume
    snapshot_id: Optional[str] = None

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Orchestrator report (v4.1)
    report: Optional[str] = None  # Generated report from orchestrator
    summary: Optional[Dict[str, Any]] = None  # Structured execution summary

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "status": self.status.value,
            "output": self.output,
            "final_output": self.final_output,
            "agents_called": self.agents_called,
            "iterations": self.iterations,
            "duration_ms": self.duration_ms,
            "tokens_used": self.tokens_used,
            "error": self.error,
            "failed_agent": self.failed_agent,
            "escalations": self.escalations,
            "snapshot_id": self.snapshot_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "report": self.report,
            "summary": self.summary,
        }


# Backwards compatibility alias
TeamResult = RunResult
