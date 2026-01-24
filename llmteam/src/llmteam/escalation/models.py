"""
Escalation Models.

Core models for escalation handling in LLMTeam.
"""
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime


class EscalationLevel(Enum):
    """Severity levels for escalations."""
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()
    EMERGENCY = auto()


class EscalationAction(Enum):
    """Actions that can be taken in response to an escalation."""
    ACKNOWLEDGE = "acknowledge"  # Log and continue
    RETRY = "retry"  # Retry the failed operation
    REDIRECT = "redirect"  # Route to alternative team/agent
    HUMAN_REVIEW = "human_review"  # Escalate to human
    ABORT = "abort"  # Stop execution


@dataclass
class Escalation:
    """
    Escalation event from a team or agent.

    Attributes:
        level: Severity level of the escalation.
        reason: Human-readable description of the issue.
        source_team: ID of the team that raised the escalation.
        source_agent: ID of the agent that raised the escalation (optional).
        context: Additional context data.
        created_at: Timestamp when escalation was created.
    """
    level: EscalationLevel
    reason: str
    source_team: str = ""
    source_agent: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EscalationDecision:
    """
    Decision made by an escalation handler.

    Attributes:
        action: The action to take.
        message: Human-readable explanation of the decision.
        target_team: Team to redirect to (if action is REDIRECT).
        metadata: Additional metadata about the decision.
    """
    action: EscalationAction
    message: str = ""
    target_team: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EscalationRecord:
    """
    Record of an escalation and its handling.

    Used for audit trail and analytics.
    """
    id: str
    escalation: Escalation
    decision: EscalationDecision
    handled_at: datetime = field(default_factory=datetime.utcnow)
    handler_name: str = ""
