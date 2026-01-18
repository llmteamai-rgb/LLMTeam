"""Tests for SecureBus."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from llmteam.transport import (
    SecureBus,
    BusEvent,
    BusEventType,
    DataMode,
    ControlCommand,
    BusConfig,
)


class TestBusConfig:
    """Tests for BusConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = BusConfig()

        assert config.data_mode == DataMode.REFS_ONLY
        assert config.enable_audit is True
        assert config.buffer_size == 1000
        assert "api_key" in config.sensitive_fields

    def test_custom_config(self):
        """Test custom configuration."""
        config = BusConfig(
            data_mode=DataMode.FULL_PAYLOAD,
            enable_audit=False,
            buffer_size=500,
        )

        assert config.data_mode == DataMode.FULL_PAYLOAD
        assert config.enable_audit is False
        assert config.buffer_size == 500


class TestBusEvent:
    """Tests for BusEvent."""

    def test_event_creation(self):
        """Test creating an event."""
        event = BusEvent(
            event_id="evt-123",
            event_type=BusEventType.STEP_COMPLETED,
            trace_id="trace-456",
            process_run_id="run-789",
            timestamp=pytest.importorskip("datetime").datetime.now(),
            source="step_1",
            data={"result": "success"},
        )

        assert event.event_id == "evt-123"
        assert event.event_type == BusEventType.STEP_COMPLETED
        assert event.source == "step_1"

    def test_event_to_dict(self):
        """Test event serialization."""
        from datetime import datetime
        ts = datetime(2024, 1, 15, 12, 0, 0)

        event = BusEvent(
            event_id="evt-123",
            event_type=BusEventType.RUN_STARTED,
            trace_id="trace-456",
            process_run_id="run-789",
            timestamp=ts,
            source="runtime",
            data={"key": "value"},
        )

        data = event.to_dict()

        assert data["event_id"] == "evt-123"
        assert data["event_type"] == "run.started"
        assert data["trace_id"] == "trace-456"
        assert data["timestamp"] == ts.isoformat()

    def test_event_from_dict(self):
        """Test event deserialization."""
        data = {
            "event_id": "evt-123",
            "event_type": "step.completed",
            "trace_id": "trace-456",
            "process_run_id": "run-789",
            "timestamp": "2024-01-15T12:00:00",
            "source": "step_1",
            "data": {"result": "ok"},
            "metadata": {},
        }

        event = BusEvent.from_dict(data)

        assert event.event_id == "evt-123"
        assert event.event_type == BusEventType.STEP_COMPLETED
        assert event.data["result"] == "ok"


class TestSecureBus:
    """Tests for SecureBus."""

    @pytest.fixture
    def bus(self):
        """Create a SecureBus instance."""
        return SecureBus()

    @pytest.fixture
    def full_payload_bus(self):
        """Create a SecureBus with full payload mode."""
        return SecureBus(BusConfig(data_mode=DataMode.FULL_PAYLOAD))

    async def test_publish_event(self, bus):
        """Test publishing an event."""
        event = await bus.publish(
            event_type=BusEventType.STEP_COMPLETED,
            trace_id="trace-123",
            process_run_id="run-456",
            source="step_1",
            data={"result": "success"},
        )

        assert event.event_type == BusEventType.STEP_COMPLETED
        assert event.trace_id == "trace-123"
        assert event.process_run_id == "run-456"
        assert event.source == "step_1"

    async def test_subscribe_and_receive(self, bus):
        """Test subscribing to events."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        bus.subscribe(BusEventType.RUN_STARTED, handler)

        await bus.publish(
            event_type=BusEventType.RUN_STARTED,
            trace_id="trace-123",
            process_run_id="run-456",
            source="runtime",
        )

        assert len(received_events) == 1
        assert received_events[0].event_type == BusEventType.RUN_STARTED

    async def test_unsubscribe(self, bus):
        """Test unsubscribing from events."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        bus.subscribe(BusEventType.STEP_STARTED, handler)
        bus.unsubscribe(BusEventType.STEP_STARTED, handler)

        await bus.publish(
            event_type=BusEventType.STEP_STARTED,
            trace_id="trace-123",
            process_run_id="run-456",
            source="step_1",
        )

        assert len(received_events) == 0

    async def test_send_command(self, bus):
        """Test sending control commands."""
        received_commands = []

        async def handler(event):
            received_commands.append(event)

        bus.subscribe_command(ControlCommand.PAUSE, handler)

        event = await bus.send_command(
            command=ControlCommand.PAUSE,
            process_run_id="run-456",
        )

        assert event.event_type == BusEventType.CONTROL_COMMAND
        assert event.data["command"] == "pause"
        assert len(received_commands) == 1


class TestSecureBusDataFiltering:
    """Tests for data filtering."""

    async def test_refs_only_mode_redacts_sensitive(self):
        """Test REFS_ONLY mode redacts sensitive fields."""
        bus = SecureBus(BusConfig(data_mode=DataMode.REFS_ONLY))

        event = await bus.publish(
            event_type=BusEventType.DATA_INPUT,
            trace_id="trace-123",
            process_run_id="run-456",
            source="step_1",
            data={
                "query": "normal data",
                "api_key": "secret-key-123",
                "password": "my-password",
            },
        )

        assert event.data["query"] == "normal data"
        assert event.data["api_key"] == "[REDACTED]"
        assert event.data["password"] == "[REDACTED]"

    async def test_full_payload_mode_preserves_all(self):
        """Test FULL_PAYLOAD mode preserves all data."""
        bus = SecureBus(BusConfig(data_mode=DataMode.FULL_PAYLOAD))

        event = await bus.publish(
            event_type=BusEventType.DATA_INPUT,
            trace_id="trace-123",
            process_run_id="run-456",
            source="step_1",
            data={
                "query": "normal data",
                "api_key": "secret-key-123",
            },
        )

        assert event.data["query"] == "normal data"
        assert event.data["api_key"] == "secret-key-123"


class TestSecureBusBuffer:
    """Tests for event buffer."""

    async def test_get_events(self):
        """Test retrieving events from buffer."""
        bus = SecureBus()

        await bus.publish(
            event_type=BusEventType.STEP_STARTED,
            trace_id="trace-1",
            process_run_id="run-1",
            source="step_1",
        )
        await bus.publish(
            event_type=BusEventType.STEP_COMPLETED,
            trace_id="trace-1",
            process_run_id="run-1",
            source="step_1",
        )

        events = bus.get_events(process_run_id="run-1")

        assert len(events) == 2

    async def test_get_events_filtered_by_type(self):
        """Test filtering events by type."""
        bus = SecureBus()

        await bus.publish(
            event_type=BusEventType.STEP_STARTED,
            trace_id="trace-1",
            process_run_id="run-1",
            source="step_1",
        )
        await bus.publish(
            event_type=BusEventType.STEP_COMPLETED,
            trace_id="trace-1",
            process_run_id="run-1",
            source="step_1",
        )

        events = bus.get_events(event_type=BusEventType.STEP_COMPLETED)

        assert len(events) == 1
        assert events[0].event_type == BusEventType.STEP_COMPLETED

    async def test_clear_buffer(self):
        """Test clearing the event buffer."""
        bus = SecureBus()

        await bus.publish(
            event_type=BusEventType.RUN_STARTED,
            trace_id="trace-1",
            process_run_id="run-1",
            source="runtime",
        )

        bus.clear_buffer()
        events = bus.get_events()

        assert len(events) == 0


class TestSecureBusAudit:
    """Tests for audit logging."""

    async def test_audit_log_enabled(self):
        """Test audit logging when enabled."""
        bus = SecureBus(BusConfig(enable_audit=True))

        await bus.publish(
            event_type=BusEventType.RUN_STARTED,
            trace_id="trace-123",
            process_run_id="run-456",
            source="runtime",
        )

        audit_log = bus.get_audit_log()

        assert len(audit_log) == 1
        assert audit_log[0]["trace_id"] == "trace-123"
        assert audit_log[0]["process_run_id"] == "run-456"

    async def test_audit_log_filtered_by_trace(self):
        """Test filtering audit log by trace ID."""
        bus = SecureBus()

        await bus.publish(
            event_type=BusEventType.RUN_STARTED,
            trace_id="trace-1",
            process_run_id="run-1",
            source="runtime",
        )
        await bus.publish(
            event_type=BusEventType.RUN_STARTED,
            trace_id="trace-2",
            process_run_id="run-2",
            source="runtime",
        )

        audit_log = bus.get_audit_log(trace_id="trace-1")

        assert len(audit_log) == 1
        assert audit_log[0]["trace_id"] == "trace-1"


class TestSecureBusConvenienceMethods:
    """Tests for convenience methods."""

    async def test_run_started(self):
        """Test run_started convenience method."""
        bus = SecureBus()
        event = await bus.run_started("trace-1", "run-1", {"workflow": "test"})

        assert event.event_type == BusEventType.RUN_STARTED
        assert event.data.get("workflow") == "test"

    async def test_run_completed(self):
        """Test run_completed convenience method."""
        bus = SecureBus()
        event = await bus.run_completed("trace-1", "run-1")

        assert event.event_type == BusEventType.RUN_COMPLETED

    async def test_run_failed(self):
        """Test run_failed convenience method."""
        bus = SecureBus()
        event = await bus.run_failed("trace-1", "run-1", "Connection timeout")

        assert event.event_type == BusEventType.RUN_FAILED
        assert event.data["error"] == "Connection timeout"

    async def test_step_started(self):
        """Test step_started convenience method."""
        bus = SecureBus()
        event = await bus.step_started("trace-1", "run-1", "step_1")

        assert event.event_type == BusEventType.STEP_STARTED
        assert event.source == "step_1"

    async def test_step_completed(self):
        """Test step_completed convenience method."""
        bus = SecureBus()
        event = await bus.step_completed("trace-1", "run-1", "step_1", {"result": "ok"})

        assert event.event_type == BusEventType.STEP_COMPLETED

    async def test_step_failed(self):
        """Test step_failed convenience method."""
        bus = SecureBus()
        event = await bus.step_failed("trace-1", "run-1", "step_1", "Validation error")

        assert event.event_type == BusEventType.STEP_FAILED
        assert event.data["error"] == "Validation error"
