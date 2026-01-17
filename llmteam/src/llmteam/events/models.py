"""
Worktrail Event models.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
import json


class EventType(Enum):
    """Worktrail event types."""

    # Segment lifecycle
    SEGMENT_STARTED = "segment.started"
    SEGMENT_COMPLETED = "segment.completed"
    SEGMENT_FAILED = "segment.failed"
    SEGMENT_CANCELLED = "segment.cancelled"
    SEGMENT_PAUSED = "segment.paused"
    SEGMENT_RESUMED = "segment.resumed"

    # Step lifecycle
    STEP_STARTED = "step.started"
    STEP_COMPLETED = "step.completed"
    STEP_FAILED = "step.failed"
    STEP_SKIPPED = "step.skipped"
    STEP_RETRYING = "step.retrying"

    # Human interaction
    HUMAN_TASK_CREATED = "human.task_created"
    HUMAN_TASK_ASSIGNED = "human.task_assigned"
    HUMAN_TASK_COMPLETED = "human.task_completed"
    HUMAN_TASK_ESCALATED = "human.task_escalated"

    # External actions
    ACTION_STARTED = "action.started"
    ACTION_COMPLETED = "action.completed"
    ACTION_FAILED = "action.failed"

    # Data flow
    DATA_PRODUCED = "data.produced"
    DATA_CONSUMED = "data.consumed"


class EventSeverity(Enum):
    """Event severity level for filtering."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ErrorInfo:
    """Error information."""

    error_type: str  # "ValidationError"
    error_message: str  # "Field 'email' is required"
    error_code: Optional[str] = None  # "E001"
    stack_trace: Optional[str] = None
    recoverable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "stack_trace": self.stack_trace,
            "recoverable": self.recoverable,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ErrorInfo":
        return cls(
            error_type=data["error_type"],
            error_message=data["error_message"],
            error_code=data.get("error_code"),
            stack_trace=data.get("stack_trace"),
            recoverable=data.get("recoverable", False),
        )

    @classmethod
    def from_exception(cls, exc: Exception, recoverable: bool = False) -> "ErrorInfo":
        """Create ErrorInfo from an exception."""
        import traceback

        return cls(
            error_type=type(exc).__name__,
            error_message=str(exc),
            stack_trace=traceback.format_exc(),
            recoverable=recoverable,
        )


@dataclass
class WorktrailEvent:
    """
    Standard Worktrail event.

    All fields except payload are required.
    """

    # === Identity (always) ===
    event_id: str  # Event UUID
    event_type: EventType  # Event type
    timestamp: datetime  # When it happened

    # === Context (always) ===
    tenant_id: str  # Tenant ID
    instance_id: str  # Workflow instance ID
    run_id: str  # Run ID
    segment_id: str  # Segment ID

    # === Step context (if applicable) ===
    step_id: Optional[str] = None  # Step ID
    step_type: Optional[str] = None  # Step type ("llm_agent", "http_action")

    # === Metadata ===
    severity: EventSeverity = EventSeverity.INFO
    correlation_id: Optional[str] = None  # For linking events
    parent_event_id: Optional[str] = None  # For hierarchy

    # === Payload (depends on type) ===
    payload: Dict[str, Any] = field(default_factory=dict)

    # === Error (for *_FAILED events) ===
    error: Optional[ErrorInfo] = None

    # === Timing ===
    duration_ms: Optional[int] = None  # For *_COMPLETED events

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "instance_id": self.instance_id,
            "run_id": self.run_id,
            "segment_id": self.segment_id,
            "step_id": self.step_id,
            "step_type": self.step_type,
            "severity": self.severity.value,
            "correlation_id": self.correlation_id,
            "parent_event_id": self.parent_event_id,
            "payload": self.payload,
            "error": self.error.to_dict() if self.error else None,
            "duration_ms": self.duration_ms,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorktrailEvent":
        """Deserialize from dict."""
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            tenant_id=data["tenant_id"],
            instance_id=data["instance_id"],
            run_id=data["run_id"],
            segment_id=data["segment_id"],
            step_id=data.get("step_id"),
            step_type=data.get("step_type"),
            severity=EventSeverity(data.get("severity", "info")),
            correlation_id=data.get("correlation_id"),
            parent_event_id=data.get("parent_event_id"),
            payload=data.get("payload", {}),
            error=ErrorInfo.from_dict(data["error"]) if data.get("error") else None,
            duration_ms=data.get("duration_ms"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "WorktrailEvent":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
