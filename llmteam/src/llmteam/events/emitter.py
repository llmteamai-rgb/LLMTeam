"""
Event Emitter for Worktrail events.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional
import uuid

from llmteam.events.models import (
    EventType,
    EventSeverity,
    ErrorInfo,
    WorktrailEvent,
)

if TYPE_CHECKING:
    from llmteam.runtime import RuntimeContext


class EventEmitter:
    """Emitter for Worktrail events."""

    def __init__(self, runtime: "RuntimeContext"):
        self.runtime = runtime
        self._sequence = 0

    def _make_event_id(self) -> str:
        """Generate unique event ID."""
        self._sequence += 1
        return f"{self.runtime.run_id}:{self._sequence}"

    def emit(
        self,
        event_type: EventType,
        *,
        step_id: Optional[str] = None,
        step_type: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        error: Optional[ErrorInfo] = None,
        duration_ms: Optional[int] = None,
        severity: EventSeverity = EventSeverity.INFO,
        correlation_id: Optional[str] = None,
        parent_event_id: Optional[str] = None,
    ) -> WorktrailEvent:
        """Emit an event."""
        event = WorktrailEvent(
            event_id=self._make_event_id(),
            event_type=event_type,
            timestamp=datetime.now(),
            tenant_id=self.runtime.tenant_id,
            instance_id=self.runtime.instance_id,
            run_id=self.runtime.run_id,
            segment_id=self.runtime.segment_id,
            step_id=step_id,
            step_type=step_type,
            severity=severity,
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
            payload=payload or {},
            error=error,
            duration_ms=duration_ms,
        )

        # Call hook if registered
        if self.runtime.on_event:
            self.runtime.on_event(event)

        return event

    # === Segment lifecycle ===

    def segment_started(self, payload: Optional[Dict[str, Any]] = None) -> WorktrailEvent:
        """Emit segment started event."""
        return self.emit(EventType.SEGMENT_STARTED, payload=payload)

    def segment_completed(
        self, duration_ms: int, payload: Optional[Dict[str, Any]] = None
    ) -> WorktrailEvent:
        """Emit segment completed event."""
        return self.emit(
            EventType.SEGMENT_COMPLETED, duration_ms=duration_ms, payload=payload
        )

    def segment_failed(self, error: ErrorInfo) -> WorktrailEvent:
        """Emit segment failed event."""
        return self.emit(
            EventType.SEGMENT_FAILED, error=error, severity=EventSeverity.ERROR
        )

    def segment_cancelled(
        self, payload: Optional[Dict[str, Any]] = None
    ) -> WorktrailEvent:
        """Emit segment cancelled event."""
        return self.emit(
            EventType.SEGMENT_CANCELLED, payload=payload, severity=EventSeverity.WARNING
        )

    def segment_paused(self, payload: Optional[Dict[str, Any]] = None) -> WorktrailEvent:
        """Emit segment paused event."""
        return self.emit(EventType.SEGMENT_PAUSED, payload=payload)

    def segment_resumed(
        self, payload: Optional[Dict[str, Any]] = None
    ) -> WorktrailEvent:
        """Emit segment resumed event."""
        return self.emit(EventType.SEGMENT_RESUMED, payload=payload)

    # === Step lifecycle ===

    def step_started(
        self,
        step_id: str,
        step_type: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> WorktrailEvent:
        """Emit step started event."""
        return self.emit(
            EventType.STEP_STARTED,
            step_id=step_id,
            step_type=step_type,
            payload=payload,
        )

    def step_completed(
        self,
        step_id: str,
        step_type: str,
        duration_ms: int,
        payload: Optional[Dict[str, Any]] = None,
    ) -> WorktrailEvent:
        """Emit step completed event."""
        return self.emit(
            EventType.STEP_COMPLETED,
            step_id=step_id,
            step_type=step_type,
            duration_ms=duration_ms,
            payload=payload,
        )

    def step_failed(
        self,
        step_id: str,
        step_type: str,
        error: ErrorInfo,
    ) -> WorktrailEvent:
        """Emit step failed event."""
        return self.emit(
            EventType.STEP_FAILED,
            step_id=step_id,
            step_type=step_type,
            error=error,
            severity=EventSeverity.ERROR,
        )

    def step_skipped(
        self,
        step_id: str,
        step_type: str,
        reason: str,
    ) -> WorktrailEvent:
        """Emit step skipped event."""
        return self.emit(
            EventType.STEP_SKIPPED,
            step_id=step_id,
            step_type=step_type,
            payload={"reason": reason},
        )

    def step_retrying(
        self,
        step_id: str,
        step_type: str,
        attempt: int,
        max_attempts: int,
        error: ErrorInfo,
    ) -> WorktrailEvent:
        """Emit step retrying event."""
        return self.emit(
            EventType.STEP_RETRYING,
            step_id=step_id,
            step_type=step_type,
            payload={"attempt": attempt, "max_attempts": max_attempts},
            error=error,
            severity=EventSeverity.WARNING,
        )

    # === Human interaction ===

    def human_task_created(
        self,
        step_id: str,
        task_id: str,
        task_type: str,
        assignee: Optional[str] = None,
    ) -> WorktrailEvent:
        """Emit human task created event."""
        return self.emit(
            EventType.HUMAN_TASK_CREATED,
            step_id=step_id,
            step_type="human_task",
            payload={
                "task_id": task_id,
                "task_type": task_type,
                "assignee": assignee,
            },
        )

    def human_task_completed(
        self,
        step_id: str,
        task_id: str,
        completed_by: str,
        duration_ms: int,
    ) -> WorktrailEvent:
        """Emit human task completed event."""
        return self.emit(
            EventType.HUMAN_TASK_COMPLETED,
            step_id=step_id,
            step_type="human_task",
            duration_ms=duration_ms,
            payload={
                "task_id": task_id,
                "completed_by": completed_by,
            },
        )

    # === External actions ===

    def action_started(
        self,
        step_id: str,
        action_type: str,
        endpoint: Optional[str] = None,
    ) -> WorktrailEvent:
        """Emit action started event."""
        payload = {"action_type": action_type}
        if endpoint:
            payload["endpoint"] = endpoint
        return self.emit(
            EventType.ACTION_STARTED,
            step_id=step_id,
            step_type="action",
            payload=payload,
        )

    def action_completed(
        self,
        step_id: str,
        action_type: str,
        duration_ms: int,
        status_code: Optional[int] = None,
    ) -> WorktrailEvent:
        """Emit action completed event."""
        payload: Dict[str, Any] = {"action_type": action_type}
        if status_code is not None:
            payload["status_code"] = status_code
        return self.emit(
            EventType.ACTION_COMPLETED,
            step_id=step_id,
            step_type="action",
            duration_ms=duration_ms,
            payload=payload,
        )

    def action_failed(
        self,
        step_id: str,
        action_type: str,
        error: ErrorInfo,
    ) -> WorktrailEvent:
        """Emit action failed event."""
        return self.emit(
            EventType.ACTION_FAILED,
            step_id=step_id,
            step_type="action",
            error=error,
            payload={"action_type": action_type},
            severity=EventSeverity.ERROR,
        )

    # === Data flow ===

    def data_produced(
        self,
        step_id: str,
        output_port: str,
        data_type: str,
        size_bytes: Optional[int] = None,
    ) -> WorktrailEvent:
        """Emit data produced event."""
        payload: Dict[str, Any] = {
            "output_port": output_port,
            "data_type": data_type,
        }
        if size_bytes is not None:
            payload["size_bytes"] = size_bytes
        return self.emit(
            EventType.DATA_PRODUCED,
            step_id=step_id,
            payload=payload,
            severity=EventSeverity.DEBUG,
        )

    def data_consumed(
        self,
        step_id: str,
        input_port: str,
        from_step: str,
        from_port: str,
    ) -> WorktrailEvent:
        """Emit data consumed event."""
        return self.emit(
            EventType.DATA_CONSUMED,
            step_id=step_id,
            payload={
                "input_port": input_port,
                "from_step": from_step,
                "from_port": from_port,
            },
            severity=EventSeverity.DEBUG,
        )
