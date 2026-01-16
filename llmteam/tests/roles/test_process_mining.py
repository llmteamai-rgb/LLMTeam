"""
Tests for process mining engine.

Tests cover:
- ProcessEvent
- ProcessMiningEngine
- Process model discovery
- Metrics calculation
- XES export
"""

import pytest
from datetime import datetime, timedelta

from llmteam.roles import (
    ProcessEvent,
    ProcessMetrics,
    ProcessModel,
    ProcessMiningEngine,
)


class TestProcessEvent:
    """Tests for ProcessEvent."""

    def test_create_event(self):
        """Test creating a process event."""
        event = ProcessEvent(
            event_id="evt_1",
            timestamp=datetime.now(),
            activity="test_activity",
            resource="agent_1",
            case_id="case_1",
            lifecycle="complete",
            duration_ms=100,
        )

        assert event.event_id == "evt_1"
        assert event.activity == "test_activity"
        assert event.resource == "agent_1"
        assert event.lifecycle == "complete"


class TestProcessMiningEngine:
    """Tests for ProcessMiningEngine."""

    def test_create_engine(self):
        """Test creating process mining engine."""
        engine = ProcessMiningEngine()

        assert engine.get_event_count() == 0
        assert engine.get_case_count() == 0

    def test_record_event(self):
        """Test recording an event."""
        engine = ProcessMiningEngine()

        event = ProcessEvent(
            event_id="evt_1",
            timestamp=datetime.now(),
            activity="activity_1",
            resource="agent_1",
            case_id="case_1",
        )

        engine.record_event(event)

        assert engine.get_event_count() == 1
        assert engine.get_case_count() == 1

    def test_record_multiple_events(self):
        """Test recording multiple events."""
        engine = ProcessMiningEngine()

        for i in range(3):
            event = ProcessEvent(
                event_id=f"evt_{i}",
                timestamp=datetime.now(),
                activity=f"activity_{i}",
                resource="agent_1",
                case_id="case_1",
            )
            engine.record_event(event)

        assert engine.get_event_count() == 3
        assert engine.get_case_count() == 1

    def test_record_multiple_cases(self):
        """Test recording events for multiple cases."""
        engine = ProcessMiningEngine()

        for case_id in ["case_1", "case_2"]:
            for i in range(2):
                event = ProcessEvent(
                    event_id=f"{case_id}_evt_{i}",
                    timestamp=datetime.now(),
                    activity=f"activity_{i}",
                    resource="agent_1",
                    case_id=case_id,
                )
                engine.record_event(event)

        assert engine.get_event_count() == 4
        assert engine.get_case_count() == 2

    def test_discover_model_simple(self):
        """Test discovering a simple process model."""
        engine = ProcessMiningEngine()

        # Create a simple sequential process: A -> B -> C
        base_time = datetime.now()
        events = [
            ProcessEvent("evt_1", base_time, "A", "agent_1", "case_1"),
            ProcessEvent("evt_2", base_time + timedelta(seconds=1), "B", "agent_1", "case_1"),
            ProcessEvent("evt_3", base_time + timedelta(seconds=2), "C", "agent_1", "case_1"),
        ]

        for event in events:
            engine.record_event(event)

        model = engine.discover_model()

        assert "A" in model.activities
        assert "B" in model.activities
        assert "C" in model.activities

        assert "B" in model.transitions["A"]
        assert "C" in model.transitions["B"]

        assert model.frequencies["A->B"] == 1
        assert model.frequencies["B->C"] == 1

    def test_discover_model_with_loops(self):
        """Test discovering model with loops."""
        engine = ProcessMiningEngine()

        base_time = datetime.now()
        # Process: A -> B -> A -> B -> C (with loop)
        events = [
            ProcessEvent("evt_1", base_time, "A", "agent_1", "case_1"),
            ProcessEvent("evt_2", base_time + timedelta(seconds=1), "B", "agent_1", "case_1"),
            ProcessEvent("evt_3", base_time + timedelta(seconds=2), "A", "agent_1", "case_1"),
            ProcessEvent("evt_4", base_time + timedelta(seconds=3), "B", "agent_1", "case_1"),
            ProcessEvent("evt_5", base_time + timedelta(seconds=4), "C", "agent_1", "case_1"),
        ]

        for event in events:
            engine.record_event(event)

        model = engine.discover_model()

        # A->B should appear twice
        assert model.frequencies["A->B"] == 2
        assert model.frequencies["B->A"] == 1
        assert model.frequencies["B->C"] == 1

    def test_calculate_metrics_simple(self):
        """Test calculating basic metrics."""
        engine = ProcessMiningEngine()

        base_time = datetime.now()
        events = [
            ProcessEvent("evt_1", base_time, "A", "agent_1", "case_1", lifecycle="start"),
            ProcessEvent("evt_2", base_time + timedelta(seconds=10), "A", "agent_1", "case_1", lifecycle="complete", duration_ms=10000),
        ]

        for event in events:
            engine.record_event(event)

        metrics = engine.calculate_metrics()

        assert metrics.completion_rate == 1.0
        assert metrics.error_rate == 0.0
        assert metrics.retry_rate == 0.0

    def test_calculate_metrics_with_errors(self):
        """Test calculating metrics with errors."""
        engine = ProcessMiningEngine()

        base_time = datetime.now()
        events = [
            ProcessEvent("evt_1", base_time, "activity_1", "agent_1", "case_1"),
            ProcessEvent("evt_2", base_time + timedelta(seconds=1), "error_handler", "agent_1", "case_1"),
            ProcessEvent("evt_3", base_time + timedelta(seconds=2), "retry_attempt", "agent_1", "case_1"),
        ]

        for event in events:
            engine.record_event(event)

        metrics = engine.calculate_metrics()

        assert metrics.error_rate > 0
        assert metrics.retry_rate > 0

    def test_find_bottlenecks(self):
        """Test finding bottleneck activities."""
        engine = ProcessMiningEngine()

        base_time = datetime.now()
        # Activity A takes 1000ms (fast)
        # Activity B takes 5000ms (slow - bottleneck)
        events = [
            ProcessEvent("evt_1", base_time, "activity_a", "agent_1", "case_1", duration_ms=1000),
            ProcessEvent("evt_2", base_time + timedelta(seconds=1), "activity_b", "agent_1", "case_1", duration_ms=5000),
        ]

        for event in events:
            engine.record_event(event)

        metrics = engine.calculate_metrics()

        # activity_b should be identified as bottleneck (longer duration)
        assert "activity_b" in metrics.bottleneck_activities

    def test_export_xes_empty(self):
        """Test exporting XES with no events."""
        engine = ProcessMiningEngine()

        xes = engine.export_xes()

        assert '<?xml version="1.0" encoding="UTF-8"?>' in xes
        assert '<log>' in xes
        assert '</log>' in xes

    def test_export_xes_with_events(self):
        """Test exporting XES with events."""
        engine = ProcessMiningEngine()

        base_time = datetime.now()
        event = ProcessEvent(
            event_id="evt_1",
            timestamp=base_time,
            activity="test_activity",
            resource="agent_1",
            case_id="case_1",
            lifecycle="complete",
        )

        engine.record_event(event)

        xes = engine.export_xes()

        assert '<?xml version="1.0" encoding="UTF-8"?>' in xes
        assert '<trace>' in xes
        assert '<string key="concept:name" value="case_1"/>' in xes
        assert '<string key="concept:name" value="test_activity"/>' in xes
        assert '<string key="org:resource" value="agent_1"/>' in xes
        assert '<string key="lifecycle:transition" value="complete"/>' in xes

    def test_export_xes_with_attributes(self):
        """Test exporting XES with custom attributes."""
        engine = ProcessMiningEngine()

        event = ProcessEvent(
            event_id="evt_1",
            timestamp=datetime.now(),
            activity="test_activity",
            resource="agent_1",
            case_id="case_1",
            attributes={
                "custom_field": "custom_value",
            },
        )

        engine.record_event(event)

        xes = engine.export_xes()

        assert '<string key="custom_field" value="custom_value"/>' in xes

    def test_clear(self):
        """Test clearing all events."""
        engine = ProcessMiningEngine()

        event = ProcessEvent(
            event_id="evt_1",
            timestamp=datetime.now(),
            activity="activity_1",
            resource="agent_1",
            case_id="case_1",
        )

        engine.record_event(event)
        assert engine.get_event_count() == 1

        engine.clear()

        assert engine.get_event_count() == 0
        assert engine.get_case_count() == 0
