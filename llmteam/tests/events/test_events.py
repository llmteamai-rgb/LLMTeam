"""
Tests for Worktrail Events module.
"""

import pytest
from datetime import datetime
import json

from llmteam.events import (
    EventType,
    EventSeverity,
    ErrorInfo,
    WorktrailEvent,
    EventEmitter,
    MemoryEventStore,
    EventStream,
)
from llmteam.runtime import RuntimeContext


# === Tests for Models ===


class TestErrorInfo:
    """Tests for ErrorInfo model."""

    def test_create(self) -> None:
        error = ErrorInfo(
            error_type="ValidationError",
            error_message="Field 'email' is required",
            error_code="E001",
            recoverable=True,
        )

        assert error.error_type == "ValidationError"
        assert error.error_message == "Field 'email' is required"
        assert error.error_code == "E001"
        assert error.recoverable is True

    def test_to_dict(self) -> None:
        error = ErrorInfo(
            error_type="ValidationError",
            error_message="Field 'email' is required",
        )

        data = error.to_dict()

        assert data["error_type"] == "ValidationError"
        assert data["error_message"] == "Field 'email' is required"

    def test_from_dict(self) -> None:
        data = {
            "error_type": "RuntimeError",
            "error_message": "Something went wrong",
            "error_code": "E002",
            "recoverable": True,
        }

        error = ErrorInfo.from_dict(data)

        assert error.error_type == "RuntimeError"
        assert error.error_code == "E002"

    def test_from_exception(self) -> None:
        exc = ValueError("Invalid value")
        error = ErrorInfo.from_exception(exc, recoverable=True)

        assert error.error_type == "ValueError"
        assert error.error_message == "Invalid value"
        assert error.recoverable is True
        assert error.stack_trace is not None


class TestWorktrailEvent:
    """Tests for WorktrailEvent model."""

    def test_create(self) -> None:
        event = WorktrailEvent(
            event_id="run_123:1",
            event_type=EventType.STEP_STARTED,
            timestamp=datetime.now(),
            tenant_id="acme",
            instance_id="inst_123",
            run_id="run_123",
            segment_id="pipeline_1",
            step_id="step_1",
            step_type="llm_agent",
        )

        assert event.event_id == "run_123:1"
        assert event.event_type == EventType.STEP_STARTED
        assert event.step_id == "step_1"

    def test_to_dict(self) -> None:
        event = WorktrailEvent(
            event_id="run_123:1",
            event_type=EventType.STEP_COMPLETED,
            timestamp=datetime(2025, 1, 16, 12, 0, 0),
            tenant_id="acme",
            instance_id="inst_123",
            run_id="run_123",
            segment_id="pipeline_1",
            duration_ms=1500,
        )

        data = event.to_dict()

        assert data["event_id"] == "run_123:1"
        assert data["event_type"] == "step.completed"
        assert data["duration_ms"] == 1500

    def test_to_json(self) -> None:
        event = WorktrailEvent(
            event_id="run_123:1",
            event_type=EventType.STEP_STARTED,
            timestamp=datetime(2025, 1, 16, 12, 0, 0),
            tenant_id="acme",
            instance_id="inst_123",
            run_id="run_123",
            segment_id="pipeline_1",
        )

        json_str = event.to_json()
        data = json.loads(json_str)

        assert data["event_type"] == "step.started"

    def test_from_dict(self) -> None:
        data = {
            "event_id": "run_123:1",
            "event_type": "step.completed",
            "timestamp": "2025-01-16T12:00:00",
            "tenant_id": "acme",
            "instance_id": "inst_123",
            "run_id": "run_123",
            "segment_id": "pipeline_1",
            "step_id": "step_1",
            "severity": "info",
            "duration_ms": 1500,
        }

        event = WorktrailEvent.from_dict(data)

        assert event.event_type == EventType.STEP_COMPLETED
        assert event.duration_ms == 1500

    def test_with_error(self) -> None:
        error = ErrorInfo(
            error_type="RuntimeError",
            error_message="Something went wrong",
        )

        event = WorktrailEvent(
            event_id="run_123:1",
            event_type=EventType.STEP_FAILED,
            timestamp=datetime.now(),
            tenant_id="acme",
            instance_id="inst_123",
            run_id="run_123",
            segment_id="pipeline_1",
            error=error,
            severity=EventSeverity.ERROR,
        )

        assert event.error is not None
        assert event.error.error_type == "RuntimeError"

        data = event.to_dict()
        assert data["error"]["error_type"] == "RuntimeError"


# === Tests for EventEmitter ===


class TestEventEmitter:
    """Tests for EventEmitter."""

    def _create_runtime(self) -> RuntimeContext:
        return RuntimeContext(
            tenant_id="acme",
            instance_id="inst_123",
            run_id="run_456",
            segment_id="pipeline_1",
        )

    def test_emit(self) -> None:
        runtime = self._create_runtime()
        emitter = EventEmitter(runtime)

        event = emitter.emit(EventType.STEP_STARTED, step_id="step_1", step_type="llm_agent")

        assert event.event_type == EventType.STEP_STARTED
        assert event.step_id == "step_1"
        assert event.tenant_id == "acme"
        assert event.run_id == "run_456"

    def test_emit_with_hook(self) -> None:
        events: list[WorktrailEvent] = []

        def on_event(event: WorktrailEvent) -> None:
            events.append(event)

        runtime = self._create_runtime()
        runtime.on_event = on_event

        emitter = EventEmitter(runtime)
        emitter.emit(EventType.STEP_STARTED, step_id="step_1", step_type="llm_agent")

        assert len(events) == 1
        assert events[0].step_id == "step_1"

    def test_segment_started(self) -> None:
        runtime = self._create_runtime()
        emitter = EventEmitter(runtime)

        event = emitter.segment_started(payload={"input_count": 10})

        assert event.event_type == EventType.SEGMENT_STARTED
        assert event.payload["input_count"] == 10

    def test_segment_completed(self) -> None:
        runtime = self._create_runtime()
        emitter = EventEmitter(runtime)

        event = emitter.segment_completed(duration_ms=5000)

        assert event.event_type == EventType.SEGMENT_COMPLETED
        assert event.duration_ms == 5000

    def test_segment_failed(self) -> None:
        runtime = self._create_runtime()
        emitter = EventEmitter(runtime)

        error = ErrorInfo(error_type="RuntimeError", error_message="Failed")
        event = emitter.segment_failed(error)

        assert event.event_type == EventType.SEGMENT_FAILED
        assert event.error is not None
        assert event.severity == EventSeverity.ERROR

    def test_step_started(self) -> None:
        runtime = self._create_runtime()
        emitter = EventEmitter(runtime)

        event = emitter.step_started("step_1", "llm_agent", payload={"model": "gpt-4"})

        assert event.event_type == EventType.STEP_STARTED
        assert event.step_id == "step_1"
        assert event.step_type == "llm_agent"

    def test_step_completed(self) -> None:
        runtime = self._create_runtime()
        emitter = EventEmitter(runtime)

        event = emitter.step_completed("step_1", "llm_agent", duration_ms=1500)

        assert event.event_type == EventType.STEP_COMPLETED
        assert event.duration_ms == 1500

    def test_step_failed(self) -> None:
        runtime = self._create_runtime()
        emitter = EventEmitter(runtime)

        error = ErrorInfo(error_type="ValidationError", error_message="Invalid input")
        event = emitter.step_failed("step_1", "llm_agent", error)

        assert event.event_type == EventType.STEP_FAILED
        assert event.error is not None

    def test_step_retrying(self) -> None:
        runtime = self._create_runtime()
        emitter = EventEmitter(runtime)

        error = ErrorInfo(error_type="TimeoutError", error_message="Timeout", recoverable=True)
        event = emitter.step_retrying("step_1", "http_action", attempt=2, max_attempts=3, error=error)

        assert event.event_type == EventType.STEP_RETRYING
        assert event.payload["attempt"] == 2
        assert event.payload["max_attempts"] == 3

    def test_event_id_sequence(self) -> None:
        runtime = self._create_runtime()
        emitter = EventEmitter(runtime)

        e1 = emitter.emit(EventType.STEP_STARTED, step_id="step_1", step_type="test")
        e2 = emitter.emit(EventType.STEP_COMPLETED, step_id="step_1", step_type="test")

        assert e1.event_id == "run_456:1"
        assert e2.event_id == "run_456:2"


# === Tests for MemoryEventStore ===


class TestMemoryEventStore:
    """Tests for MemoryEventStore."""

    def _create_event(self, run_id: str, step_id: str = None) -> WorktrailEvent:
        return WorktrailEvent(
            event_id=f"{run_id}:1",
            event_type=EventType.STEP_STARTED,
            timestamp=datetime.now(),
            tenant_id="acme",
            instance_id="inst_123",
            run_id=run_id,
            segment_id="pipeline_1",
            step_id=step_id,
        )

    async def test_append_and_get(self) -> None:
        store = MemoryEventStore()
        event = self._create_event("run_1", "step_1")

        await store.append(event)
        events = await store.get_by_run("run_1")

        assert len(events) == 1
        assert events[0].event_id == event.event_id

    async def test_get_by_step(self) -> None:
        store = MemoryEventStore()

        await store.append(self._create_event("run_1", "step_1"))
        await store.append(self._create_event("run_1", "step_2"))

        events = await store.get_by_step("run_1", "step_1")
        assert len(events) == 1

    async def test_get_by_type(self) -> None:
        store = MemoryEventStore()

        event1 = WorktrailEvent(
            event_id="run_1:1",
            event_type=EventType.STEP_STARTED,
            timestamp=datetime.now(),
            tenant_id="acme",
            instance_id="inst_123",
            run_id="run_1",
            segment_id="pipeline_1",
        )
        event2 = WorktrailEvent(
            event_id="run_1:2",
            event_type=EventType.STEP_COMPLETED,
            timestamp=datetime.now(),
            tenant_id="acme",
            instance_id="inst_123",
            run_id="run_1",
            segment_id="pipeline_1",
        )

        await store.append(event1)
        await store.append(event2)

        events = await store.get_by_type("run_1", EventType.STEP_STARTED)
        assert len(events) == 1

    async def test_clear(self) -> None:
        store = MemoryEventStore()

        await store.append(self._create_event("run_1"))
        await store.clear()

        events = await store.get_by_run("run_1")
        assert len(events) == 0


# === Tests for EventStream ===


class TestEventStream:
    """Tests for EventStream."""

    async def test_publish_and_get_history(self) -> None:
        stream = EventStream()

        event = WorktrailEvent(
            event_id="run_1:1",
            event_type=EventType.STEP_STARTED,
            timestamp=datetime.now(),
            tenant_id="acme",
            instance_id="inst_123",
            run_id="run_1",
            segment_id="pipeline_1",
        )

        await stream.publish(event)
        history = await stream.get_history("run_1")

        assert len(history) == 1
        assert history[0].event_id == "run_1:1"

    async def test_subscriber_count(self) -> None:
        stream = EventStream()

        assert await stream.get_subscriber_count("run_1") == 0
        assert not await stream.has_subscribers("run_1")
