"""
Human Interaction - Data Models.

Provides data structures for human-in-the-loop interactions:
- InteractionType, InteractionStatus, InteractionPriority enums
- InteractionRequest for creating interaction requests
- InteractionResponse for human responses
- NotificationConfig for notification settings
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional


class InteractionType(Enum):
    """Type of human interaction."""

    APPROVAL = "approval"  # Yes/No decision
    CHOICE = "choice"  # Select from options
    INPUT = "input"  # Data entry
    REVIEW = "review"  # Review and edit
    CHAT = "chat"  # Dialogue
    TASK = "task"  # Execute task


class InteractionStatus(Enum):
    """Interaction status."""

    PENDING = "pending"
    NOTIFIED = "notified"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    ESCALATED = "escalated"


class InteractionPriority(Enum):
    """Interaction priority."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class InteractionRequest:
    """Request for human interaction."""

    request_id: str
    interaction_type: InteractionType

    # Context
    run_id: str
    pipeline_id: str
    agent_name: str
    step_name: str
    tenant_id: str

    # Content
    title: str
    description: str
    context_data: Dict[str, Any] = field(default_factory=dict)

    # For CHOICE type
    options: List[Dict[str, Any]] = field(default_factory=list)

    # For INPUT type
    input_schema: Optional[dict] = None

    # Assignment
    assignee_id: Optional[str] = None
    assignee_group: Optional[str] = None

    # Priority & Timing
    priority: InteractionPriority = InteractionPriority.NORMAL
    timeout: timedelta = field(default_factory=lambda: timedelta(hours=24))
    deadline: Optional[datetime] = None

    # SLA
    sla_warning: Optional[timedelta] = None  # Warning threshold
    sla_breach: Optional[timedelta] = None  # Breach threshold

    # Status
    status: InteractionStatus = InteractionStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now())
    updated_at: Optional[datetime] = None

    # Escalation
    escalation_chain: List[str] = field(default_factory=list)
    current_escalation_level: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "interaction_type": self.interaction_type.value,
            "run_id": self.run_id,
            "pipeline_id": self.pipeline_id,
            "agent_name": self.agent_name,
            "step_name": self.step_name,
            "tenant_id": self.tenant_id,
            "title": self.title,
            "description": self.description,
            "context_data": self.context_data,
            "options": self.options,
            "input_schema": self.input_schema,
            "assignee_id": self.assignee_id,
            "assignee_group": self.assignee_group,
            "priority": self.priority.value,
            "timeout": self.timeout.total_seconds(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "sla_warning": self.sla_warning.total_seconds() if self.sla_warning else None,
            "sla_breach": self.sla_breach.total_seconds() if self.sla_breach else None,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "escalation_chain": self.escalation_chain,
            "current_escalation_level": self.current_escalation_level,
        }


@dataclass
class InteractionResponse:
    """Human response to interaction request."""

    request_id: str
    responder_id: str

    # Decision
    approved: Optional[bool] = None  # For APPROVAL
    selected_option: Optional[str] = None  # For CHOICE
    input_data: Dict[str, Any] = field(default_factory=dict)  # For INPUT
    review_changes: Dict[str, Any] = field(default_factory=dict)  # For REVIEW

    # Metadata
    response_time: datetime = field(default_factory=lambda: datetime.now())
    comment: str = ""

    # Audit
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "responder_id": self.responder_id,
            "approved": self.approved,
            "selected_option": self.selected_option,
            "input_data": self.input_data,
            "review_changes": self.review_changes,
            "response_time": self.response_time.isoformat(),
            "comment": self.comment,
            "reason": self.reason,
        }


@dataclass
class NotificationConfig:
    """Notification configuration."""

    # Channels
    email_enabled: bool = True
    slack_enabled: bool = False
    teams_enabled: bool = False
    webhook_enabled: bool = False

    # Settings
    slack_channel: str = ""
    slack_webhook_url: str = ""
    teams_webhook_url: str = ""
    custom_webhook_url: str = ""

    # Templates
    email_template: str = ""
    slack_template: str = ""
