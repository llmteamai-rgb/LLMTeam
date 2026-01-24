"""
Models for group orchestration.

RFC-009: Group Architecture Unification
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

if TYPE_CHECKING:
    from llmteam.orchestration.group import GroupOrchestrator


class TeamRole(Enum):
    """
    Role of a team within a group.

    Determines priority and permissions of the team.
    """

    LEADER = "leader"
    """
    Group leader.
    - Called first (unless specified otherwise)
    - Receives unresolved tasks
    - Only one per group
    """

    MEMBER = "member"
    """
    Regular member.
    - Called by GroupOrchestrator decision
    - Can escalate to group
    """

    SPECIALIST = "specialist"
    """
    Specialized team.
    - Called only on explicit request
    - Usually for specific tasks
    - Can be called by other teams
    """

    FALLBACK = "fallback"
    """
    Fallback team.
    - Called on errors from other teams
    - Handles edge cases
    """


class GroupEscalationAction(Enum):
    """
    Actions for group-level escalation handling.

    Used by GroupOrchestrator to respond to team escalations.
    Different from EscalationAction in llmteam.escalation which
    is for agent/team-level handling.
    """

    RETRY = "retry"
    """Retry the failed operation."""

    SKIP = "skip"
    """Skip and continue with next step."""

    REROUTE = "reroute"
    """Reroute to another team."""

    ABORT = "abort"
    """Abort group execution."""

    CONTINUE = "continue"
    """Continue as-is."""

    HUMAN = "human"
    """Requires human intervention."""


# Backward compatibility alias
EscalationAction = GroupEscalationAction


@dataclass
class GroupContext:
    """
    Group context for a team.

    Passed to team when added to a group.
    Provides bi-directional link between group and team.

    RFC-009: Group Architecture Unification
    """

    # Identity
    group_id: str
    """ID of the group."""

    group_orchestrator: "GroupOrchestrator"
    """Reference to GroupOrchestrator."""

    # Team's role
    team_role: TeamRole
    """Role of this team in the group."""

    # Group info
    other_teams: List[str] = field(default_factory=list)
    """IDs of other teams in the group."""

    leader_team: Optional[str] = None
    """ID of the leader team."""

    # Permissions
    can_escalate: bool = True
    """Can the team escalate to group."""

    can_request_team: bool = False
    """Can the team request other teams directly."""

    visible_teams: Set[str] = field(default_factory=set)
    """Which teams are visible to this team."""

    # Shared state
    shared_context: Dict[str, Any] = field(default_factory=dict)
    """Shared group context (read-only for teams)."""

    # Callbacks
    on_escalation: Optional[Callable] = field(default=None, repr=False)
    """Callback for escalation (set by GroupOrchestrator)."""


@dataclass
class EscalationRequest:
    """Escalation request from a team."""

    source_team_id: str
    """Source team ID."""

    source_agent_id: Optional[str] = None
    """Source agent ID (if escalation from agent)."""

    reason: str = ""
    """Escalation reason."""

    error: Optional[Exception] = field(default=None, repr=False)
    """Error (if any)."""

    context: Dict[str, Any] = field(default_factory=dict)
    """Additional context."""

    suggested_action: Optional[str] = None
    """Suggested action from team."""

    created_at: datetime = field(default_factory=datetime.utcnow)
    """Creation time."""


@dataclass
class EscalationResponse:
    """Response to escalation from GroupOrchestrator."""

    action: EscalationAction
    """Action for the team."""

    reason: str = ""
    """Decision justification."""

    retry_with: Optional[Dict[str, Any]] = None
    """Data for retry (if action=RETRY)."""

    route_to_team: Optional[str] = None
    """Team for rerouting (if action=REROUTE)."""

    additional_context: Dict[str, Any] = field(default_factory=dict)
    """Additional context."""

    created_at: datetime = field(default_factory=datetime.utcnow)
    """Creation time."""


__all__ = [
    "TeamRole",
    "GroupEscalationAction",
    "EscalationAction",  # Backward compatibility alias
    "GroupContext",
    "EscalationRequest",
    "EscalationResponse",
]
