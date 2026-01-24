"""
Team lifecycle management.

RFC-014: Enhanced Configurator Mode (opt-in lifecycle).

When enforce_lifecycle=True:
- Team starts in UNCONFIGURED state
- Must transition through CONFIGURING → READY before RUNNING
- State transitions are enforced
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TeamState(str, Enum):
    """Team lifecycle states."""

    UNCONFIGURED = "unconfigured"
    CONFIGURING = "configuring"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class ProposalStatus(str, Enum):
    """Configuration proposal status."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"


# Valid state transitions
VALID_TRANSITIONS = {
    TeamState.UNCONFIGURED: {TeamState.CONFIGURING},
    TeamState.CONFIGURING: {TeamState.READY, TeamState.UNCONFIGURED},
    TeamState.READY: {TeamState.RUNNING, TeamState.CONFIGURING},
    TeamState.RUNNING: {TeamState.COMPLETED, TeamState.FAILED, TeamState.PAUSED},
    TeamState.PAUSED: {TeamState.RUNNING, TeamState.FAILED},
    TeamState.COMPLETED: {TeamState.READY, TeamState.CONFIGURING},
    TeamState.FAILED: {TeamState.READY, TeamState.CONFIGURING},
}


@dataclass
class ConfigurationProposal:
    """
    A proposed configuration change.

    Generated during CONFIGURING state, must be approved
    before team transitions to READY.

    Args:
        proposal_id: Unique proposal identifier
        changes: Dict of proposed changes
        reason: Reason for the proposal
        status: Current status
    """

    proposal_id: str
    changes: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    status: ProposalStatus = ProposalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    reviewed_at: Optional[datetime] = None

    def approve(self) -> None:
        """Approve the proposal."""
        self.status = ProposalStatus.APPROVED
        self.reviewed_at = datetime.utcnow()

    def reject(self, reason: str = "") -> None:
        """Reject the proposal."""
        self.status = ProposalStatus.REJECTED
        self.reviewed_at = datetime.utcnow()
        if reason:
            self.reason = f"{self.reason} | Rejected: {reason}"

    def mark_applied(self) -> None:
        """Mark as applied."""
        self.status = ProposalStatus.APPLIED

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "proposal_id": self.proposal_id,
            "changes": self.changes,
            "reason": self.reason,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
        }


class LifecycleError(Exception):
    """Raised when a lifecycle state transition is invalid."""

    def __init__(self, current: TeamState, target: TeamState, message: str = ""):
        self.current = current
        self.target = target
        msg = message or (
            f"Invalid state transition: {current.value} → {target.value}"
        )
        super().__init__(msg)


class TeamLifecycle:
    """
    Manages team state transitions (RFC-014).

    Only active when enforce_lifecycle=True.

    States:
        UNCONFIGURED → CONFIGURING → READY → RUNNING → COMPLETED
                                                      → FAILED
                                              → PAUSED → RUNNING
    """

    def __init__(self):
        self._state = TeamState.UNCONFIGURED
        self._proposals: List[ConfigurationProposal] = []
        self._history: List[Dict[str, Any]] = []

    @property
    def state(self) -> TeamState:
        """Current team state."""
        return self._state

    @property
    def proposals(self) -> List[ConfigurationProposal]:
        """All configuration proposals."""
        return self._proposals.copy()

    @property
    def pending_proposals(self) -> List[ConfigurationProposal]:
        """Proposals awaiting review."""
        return [p for p in self._proposals if p.status == ProposalStatus.PENDING]

    def transition_to(self, target: TeamState) -> None:
        """
        Transition to a new state.

        Args:
            target: Target state

        Raises:
            LifecycleError: If transition is invalid
        """
        valid = VALID_TRANSITIONS.get(self._state, set())
        if target not in valid:
            raise LifecycleError(self._state, target)

        # Record transition
        self._history.append({
            "from": self._state.value,
            "to": target.value,
            "timestamp": datetime.utcnow().isoformat(),
        })

        self._state = target

    def can_transition_to(self, target: TeamState) -> bool:
        """Check if transition is valid without performing it."""
        valid = VALID_TRANSITIONS.get(self._state, set())
        return target in valid

    def ensure_ready(self) -> None:
        """
        Ensure team is in READY state (or can run).

        Raises:
            LifecycleError: If team is not ready to run
        """
        if self._state != TeamState.READY:
            raise LifecycleError(
                self._state,
                TeamState.RUNNING,
                f"Team must be in READY state to run (current: {self._state.value}). "
                f"Call team.mark_ready() after configuration.",
            )

    def add_proposal(self, proposal: ConfigurationProposal) -> None:
        """Add a configuration proposal."""
        self._proposals.append(proposal)

    def approve_all(self) -> int:
        """
        Approve all pending proposals.

        Returns:
            Number of proposals approved
        """
        count = 0
        for p in self._proposals:
            if p.status == ProposalStatus.PENDING:
                p.approve()
                count += 1
        return count

    def has_unapplied_approvals(self) -> bool:
        """Check if there are approved but not applied proposals."""
        return any(p.status == ProposalStatus.APPROVED for p in self._proposals)

    @property
    def history(self) -> List[Dict[str, Any]]:
        """State transition history."""
        return self._history.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize lifecycle state."""
        return {
            "state": self._state.value,
            "proposals_count": len(self._proposals),
            "pending_count": len(self.pending_proposals),
            "history": self._history,
        }
