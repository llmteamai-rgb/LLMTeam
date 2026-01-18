"""
Secure Data Bus (RFC-002).

Provides a secure event bus for integration with external systems like Corpos.
Features:
- Mandatory events (run.started, step.completed, etc.)
- Control commands (run.cancel, run.pause)
- Data modes (refs only vs full payload)
- Audit logging with trace_id and process_run_id
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from llmteam.observability import get_logger


logger = get_logger(__name__)


class BusEventType(Enum):
    """Mandatory bus event types."""

    # Run lifecycle
    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"
    RUN_PAUSED = "run.paused"
    RUN_RESUMED = "run.resumed"

    # Step lifecycle
    STEP_STARTED = "step.started"
    STEP_COMPLETED = "step.completed"
    STEP_FAILED = "step.failed"
    STEP_SKIPPED = "step.skipped"

    # Data events
    DATA_INPUT = "data.input"
    DATA_OUTPUT = "data.output"
    DATA_TRANSFORM = "data.transform"

    # Control events
    CONTROL_COMMAND = "control.command"
    CONTROL_ACK = "control.ack"

    # Human interaction
    HUMAN_REQUESTED = "human.requested"
    HUMAN_RESPONDED = "human.responded"
    HUMAN_TIMEOUT = "human.timeout"

    # Escalation
    ESCALATION_RAISED = "escalation.raised"
    ESCALATION_HANDLED = "escalation.handled"


class DataMode(Enum):
    """Data transmission modes."""

    REFS_ONLY = 0  # Only send references, not actual data
    FULL_PAYLOAD = 1  # Send complete payload


class ControlCommand(Enum):
    """Control commands for run management."""

    CANCEL = "cancel"
    PAUSE = "pause"
    RESUME = "resume"
    RETRY = "retry"
    SKIP = "skip"


@dataclass
class BusEvent:
    """
    Event transmitted through the secure bus.

    Attributes:
        event_id: Unique event identifier
        event_type: Type of event
        trace_id: Trace ID for distributed tracing
        process_run_id: Process run identifier
        timestamp: Event timestamp
        source: Source component/step
        data: Event data (filtered based on DataMode)
        metadata: Additional metadata
    """

    event_id: str
    event_type: BusEventType
    trace_id: str
    process_run_id: str
    timestamp: datetime
    source: str
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for transmission."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "trace_id": self.trace_id,
            "process_run_id": self.process_run_id,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "data": self.data,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BusEvent":
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=BusEventType(data["event_type"]),
            trace_id=data["trace_id"],
            process_run_id=data["process_run_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data["source"],
            data=data.get("data", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class BusConfig:
    """
    Configuration for SecureBus.

    Attributes:
        data_mode: Data transmission mode
        sensitive_fields: Fields to redact in REFS_ONLY mode
        max_payload_size: Maximum payload size in bytes
        enable_audit: Whether to enable audit logging
        buffer_size: Event buffer size
    """

    data_mode: DataMode = DataMode.REFS_ONLY
    sensitive_fields: Set[str] = field(default_factory=lambda: {
        "api_key", "password", "secret", "token", "credential",
        "ssn", "credit_card", "pii",
    })
    max_payload_size: int = 1024 * 1024  # 1MB
    enable_audit: bool = True
    buffer_size: int = 1000


EventHandler = Callable[[BusEvent], None]
AsyncEventHandler = Callable[[BusEvent], Any]


class SecureBus:
    """
    Secure Data Bus for external system integration.

    Provides:
    - Event publishing with mandatory event types
    - Control command handling
    - Data mode filtering (refs vs payload)
    - Audit trail integration
    - Async event delivery

    Example:
        bus = SecureBus(BusConfig(data_mode=DataMode.REFS_ONLY))

        # Subscribe to events
        bus.subscribe(BusEventType.STEP_COMPLETED, my_handler)

        # Publish event
        await bus.publish(
            event_type=BusEventType.STEP_COMPLETED,
            trace_id="trace-123",
            process_run_id="run-456",
            source="step_1",
            data={"result": "success"},
        )

        # Send control command
        await bus.send_command(
            command=ControlCommand.PAUSE,
            process_run_id="run-456",
        )
    """

    def __init__(self, config: Optional[BusConfig] = None) -> None:
        """
        Initialize SecureBus.

        Args:
            config: Bus configuration
        """
        self.config = config or BusConfig()
        self._subscribers: Dict[BusEventType, List[AsyncEventHandler]] = {}
        self._command_handlers: Dict[ControlCommand, List[AsyncEventHandler]] = {}
        self._event_buffer: List[BusEvent] = []
        self._audit_log: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()

        logger.debug(f"SecureBus initialized (mode={self.config.data_mode.name})")

    def subscribe(
        self,
        event_type: BusEventType,
        handler: AsyncEventHandler,
    ) -> None:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Event type to subscribe to
            handler: Async handler function
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed to {event_type.value}")

    def unsubscribe(
        self,
        event_type: BusEventType,
        handler: AsyncEventHandler,
    ) -> None:
        """
        Unsubscribe from events.

        Args:
            event_type: Event type
            handler: Handler to remove
        """
        if event_type in self._subscribers:
            self._subscribers[event_type] = [
                h for h in self._subscribers[event_type] if h != handler
            ]

    def subscribe_command(
        self,
        command: ControlCommand,
        handler: AsyncEventHandler,
    ) -> None:
        """
        Subscribe to control commands.

        Args:
            command: Command type
            handler: Async handler function
        """
        if command not in self._command_handlers:
            self._command_handlers[command] = []
        self._command_handlers[command].append(handler)
        logger.debug(f"Subscribed to command {command.value}")

    async def publish(
        self,
        event_type: BusEventType,
        trace_id: str,
        process_run_id: str,
        source: str,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BusEvent:
        """
        Publish an event to the bus.

        Args:
            event_type: Type of event
            trace_id: Trace ID for distributed tracing
            process_run_id: Process run identifier
            source: Source component/step
            data: Event data
            metadata: Additional metadata

        Returns:
            Published BusEvent
        """
        # Filter data based on mode
        filtered_data = self._filter_data(data or {})

        event = BusEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            trace_id=trace_id,
            process_run_id=process_run_id,
            timestamp=datetime.now(),
            source=source,
            data=filtered_data,
            metadata=metadata or {},
        )

        # Add to buffer
        async with self._lock:
            self._event_buffer.append(event)
            if len(self._event_buffer) > self.config.buffer_size:
                self._event_buffer.pop(0)

        # Audit log
        if self.config.enable_audit:
            self._audit_log.append({
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "trace_id": trace_id,
                "process_run_id": process_run_id,
                "source": source,
                "timestamp": event.timestamp.isoformat(),
            })

        logger.debug(f"Published event {event.event_id} ({event_type.value})")

        # Notify subscribers
        await self._notify_subscribers(event)

        return event

    async def send_command(
        self,
        command: ControlCommand,
        process_run_id: str,
        trace_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> BusEvent:
        """
        Send a control command.

        Args:
            command: Control command
            process_run_id: Target process run
            trace_id: Optional trace ID
            data: Additional command data

        Returns:
            Command event
        """
        event = await self.publish(
            event_type=BusEventType.CONTROL_COMMAND,
            trace_id=trace_id or str(uuid.uuid4()),
            process_run_id=process_run_id,
            source="bus",
            data={
                "command": command.value,
                **(data or {}),
            },
        )

        # Notify command handlers
        handlers = self._command_handlers.get(command, [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Command handler error: {e}")

        return event

    async def _notify_subscribers(self, event: BusEvent) -> None:
        """
        Notify all subscribers of an event.

        Args:
            event: Event to broadcast
        """
        handlers = self._subscribers.get(event.event_type, [])
        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    def _filter_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter data based on DataMode.

        Args:
            data: Original data

        Returns:
            Filtered data
        """
        if self.config.data_mode == DataMode.FULL_PAYLOAD:
            return data

        # REFS_ONLY mode - redact sensitive fields
        filtered = {}
        for key, value in data.items():
            # Check if key contains sensitive field name
            is_sensitive = any(
                s in key.lower() for s in self.config.sensitive_fields
            )
            if is_sensitive:
                filtered[key] = "[REDACTED]"
            elif isinstance(value, dict):
                filtered[key] = self._filter_data(value)
            else:
                filtered[key] = value

        return filtered

    def get_events(
        self,
        process_run_id: Optional[str] = None,
        event_type: Optional[BusEventType] = None,
        limit: int = 100,
    ) -> List[BusEvent]:
        """
        Get events from buffer.

        Args:
            process_run_id: Filter by process run
            event_type: Filter by event type
            limit: Maximum events to return

        Returns:
            List of events
        """
        events = self._event_buffer

        if process_run_id:
            events = [e for e in events if e.process_run_id == process_run_id]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return list(reversed(events[-limit:]))

    def get_audit_log(
        self,
        trace_id: Optional[str] = None,
        process_run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get audit log entries.

        Args:
            trace_id: Filter by trace ID
            process_run_id: Filter by process run ID

        Returns:
            List of audit log entries
        """
        entries = self._audit_log

        if trace_id:
            entries = [e for e in entries if e["trace_id"] == trace_id]

        if process_run_id:
            entries = [e for e in entries if e["process_run_id"] == process_run_id]

        return entries

    def clear_buffer(self) -> None:
        """Clear the event buffer."""
        self._event_buffer.clear()
        logger.debug("Event buffer cleared")

    # Convenience methods for mandatory events

    async def run_started(
        self,
        trace_id: str,
        process_run_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> BusEvent:
        """Emit run.started event."""
        return await self.publish(
            BusEventType.RUN_STARTED, trace_id, process_run_id, "runtime", data
        )

    async def run_completed(
        self,
        trace_id: str,
        process_run_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> BusEvent:
        """Emit run.completed event."""
        return await self.publish(
            BusEventType.RUN_COMPLETED, trace_id, process_run_id, "runtime", data
        )

    async def run_failed(
        self,
        trace_id: str,
        process_run_id: str,
        error: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> BusEvent:
        """Emit run.failed event."""
        return await self.publish(
            BusEventType.RUN_FAILED, trace_id, process_run_id, "runtime",
            {"error": error, **(data or {})}
        )

    async def step_started(
        self,
        trace_id: str,
        process_run_id: str,
        step_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> BusEvent:
        """Emit step.started event."""
        return await self.publish(
            BusEventType.STEP_STARTED, trace_id, process_run_id, step_id, data
        )

    async def step_completed(
        self,
        trace_id: str,
        process_run_id: str,
        step_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> BusEvent:
        """Emit step.completed event."""
        return await self.publish(
            BusEventType.STEP_COMPLETED, trace_id, process_run_id, step_id, data
        )

    async def step_failed(
        self,
        trace_id: str,
        process_run_id: str,
        step_id: str,
        error: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> BusEvent:
        """Emit step.failed event."""
        return await self.publish(
            BusEventType.STEP_FAILED, trace_id, process_run_id, step_id,
            {"error": error, **(data or {})}
        )
