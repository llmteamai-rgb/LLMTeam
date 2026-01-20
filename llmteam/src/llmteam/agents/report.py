"""
Agent Report model for orchestrator communication.

Each agent generates a report after execution that is sent to the orchestrator.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class AgentReport:
    """
    Report from agent execution.

    Automatically generated after each agent process() call
    and sent to TeamOrchestrator.
    """

    # Identification
    agent_id: str
    agent_role: str
    agent_type: str  # "llm", "rag", "kag"

    # Timing
    started_at: datetime
    completed_at: datetime
    duration_ms: int = 0

    # Result (compact summaries)
    input_summary: str = ""  # First 200 chars of input
    output_summary: str = ""  # First 200 chars of output
    output_key: str = ""  # Main output key name

    # Status
    success: bool = True
    error: Optional[str] = None
    error_type: Optional[str] = None

    # Metrics
    tokens_used: int = 0
    model: Optional[str] = None

    def __post_init__(self):
        """Calculate duration if not set."""
        if self.duration_ms == 0 and self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            self.duration_ms = int(delta.total_seconds() * 1000)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_role": self.agent_role,
            "agent_type": self.agent_type,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "output_key": self.output_key,
            "success": self.success,
            "error": self.error,
            "error_type": self.error_type,
            "tokens_used": self.tokens_used,
            "model": self.model,
        }

    def to_log_line(self) -> str:
        """Generate single-line log representation."""
        status = "OK" if self.success else f"FAIL: {self.error_type}"
        return (
            f"[{self.agent_id}] {self.agent_type.upper()} "
            f"| {self.duration_ms}ms | {status}"
        )

    @classmethod
    def create(
        cls,
        agent_id: str,
        agent_role: str,
        agent_type: str,
        started_at: datetime,
        input_data: Dict[str, Any],
        output: Any,
        success: bool = True,
        error: Optional[Exception] = None,
        tokens_used: int = 0,
        model: Optional[str] = None,
    ) -> "AgentReport":
        """
        Factory method to create report from execution data.

        Args:
            agent_id: Agent identifier
            agent_role: Agent role
            agent_type: Type of agent
            started_at: Execution start time
            input_data: Input data dict
            output: Output from agent
            success: Whether execution succeeded
            error: Exception if failed
            tokens_used: Tokens consumed
            model: Model used

        Returns:
            AgentReport instance
        """
        completed_at = datetime.utcnow()

        # Create input summary
        input_str = str(input_data)[:200]
        if len(str(input_data)) > 200:
            input_str += "..."

        # Create output summary
        if isinstance(output, dict):
            output_key = next(iter(output.keys()), "")
            output_str = str(output.get(output_key, output))[:200]
        else:
            output_key = "output"
            output_str = str(output)[:200]
        if len(str(output)) > 200:
            output_str += "..."

        return cls(
            agent_id=agent_id,
            agent_role=agent_role,
            agent_type=agent_type,
            started_at=started_at,
            completed_at=completed_at,
            input_summary=input_str,
            output_summary=output_str,
            output_key=output_key,
            success=success,
            error=str(error) if error else None,
            error_type=type(error).__name__ if error else None,
            tokens_used=tokens_used,
            model=model,
        )
