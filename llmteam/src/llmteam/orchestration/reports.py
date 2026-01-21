"""
Report models for orchestration.

RFC-004: TeamReport, GroupReport, GroupResult.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class TeamReport:
    """
    Report from TeamOrchestrator.

    Collected by GroupOrchestrator after team execution.

    RFC-009: Added team_role and escalations_sent fields.
    """

    team_id: str
    run_id: str
    success: bool
    duration_ms: int
    agents_executed: List[str]
    agent_reports: List[Dict[str, Any]] = field(default_factory=list)
    output_summary: str = ""
    errors: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    # RFC-009: Group context
    team_role: Optional[str] = None
    """Team's role in the group (if in group)."""

    escalations_sent: int = 0
    """Number of escalations sent to group."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "team_id": self.team_id,
            "run_id": self.run_id,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "agents_executed": self.agents_executed,
            "agent_reports": self.agent_reports,
            "output_summary": self.output_summary,
            "errors": self.errors,
            "created_at": self.created_at.isoformat(),
            # RFC-009
            "team_role": self.team_role,
            "escalations_sent": self.escalations_sent,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeamReport":
        """Create from dictionary."""
        report = cls(
            team_id=data["team_id"],
            run_id=data["run_id"],
            success=data["success"],
            duration_ms=data["duration_ms"],
            agents_executed=data.get("agents_executed", []),
            agent_reports=data.get("agent_reports", []),
            output_summary=data.get("output_summary", ""),
            errors=data.get("errors", []),
            # RFC-009
            team_role=data.get("team_role"),
            escalations_sent=data.get("escalations_sent", 0),
        )
        if data.get("created_at"):
            report.created_at = datetime.fromisoformat(data["created_at"])
        return report


@dataclass
class GroupReport:
    """
    Aggregated report from GroupOrchestrator.

    Contains TeamReports from all teams in the group.

    RFC-009: Added run_id and escalations_handled fields.
    """

    group_id: str
    role: str  # GroupRole value
    teams_count: int
    teams_succeeded: int
    teams_failed: int
    total_duration_ms: int
    team_reports: List[TeamReport] = field(default_factory=list)
    summary: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    # RFC-009: Run tracking
    run_id: str = ""
    """Run identifier."""

    escalations_handled: int = 0
    """Number of escalations handled."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "group_id": self.group_id,
            "role": self.role,
            "teams_count": self.teams_count,
            "teams_succeeded": self.teams_succeeded,
            "teams_failed": self.teams_failed,
            "total_duration_ms": self.total_duration_ms,
            "team_reports": [r.to_dict() for r in self.team_reports],
            "summary": self.summary,
            "created_at": self.created_at.isoformat(),
            # RFC-009
            "run_id": self.run_id,
            "escalations_handled": self.escalations_handled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroupReport":
        """Create from dictionary."""
        report = cls(
            group_id=data["group_id"],
            role=data["role"],
            teams_count=data["teams_count"],
            teams_succeeded=data["teams_succeeded"],
            teams_failed=data["teams_failed"],
            total_duration_ms=data["total_duration_ms"],
            team_reports=[TeamReport.from_dict(r) for r in data.get("team_reports", [])],
            summary=data.get("summary", ""),
            # RFC-009
            run_id=data.get("run_id", ""),
            escalations_handled=data.get("escalations_handled", 0),
        )
        if data.get("created_at"):
            report.created_at = datetime.fromisoformat(data["created_at"])
        return report


@dataclass
class GroupResult:
    """
    Result of group execution.

    Returned by GroupOrchestrator.execute().

    RFC-009: Added run_id and duration_ms fields.
    """

    success: bool
    output: Dict[str, Any] = field(default_factory=dict)
    team_results: Dict[str, Any] = field(default_factory=dict)  # team_id -> RunResult
    report: Optional[GroupReport] = None
    errors: List[str] = field(default_factory=list)

    # RFC-009: Run tracking
    run_id: str = ""
    """Run identifier."""

    duration_ms: int = 0
    """Total execution time in milliseconds."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "team_results": {
                k: v.to_dict() if hasattr(v, "to_dict") else v
                for k, v in self.team_results.items()
            },
            "report": self.report.to_dict() if self.report else None,
            "errors": self.errors,
            # RFC-009
            "run_id": self.run_id,
            "duration_ms": self.duration_ms,
        }
