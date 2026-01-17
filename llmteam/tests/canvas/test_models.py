"""
Tests for Canvas Models.
"""

import pytest
import json

from llmteam.canvas import (
    PortDefinition,
    StepPosition,
    StepUIMetadata,
    StepDefinition,
    EdgeDefinition,
    SegmentParams,
    SegmentDefinition,
)


# === Tests for StepPosition ===


class TestStepPosition:
    """Tests for StepPosition model."""

    def test_create(self) -> None:
        pos = StepPosition(x=100, y=200)

        assert pos.x == 100
        assert pos.y == 200

    def test_to_dict(self) -> None:
        pos = StepPosition(x=100.5, y=200.5)
        data = pos.to_dict()

        assert data == {"x": 100.5, "y": 200.5}

    def test_from_dict(self) -> None:
        data = {"x": 150, "y": 250}
        pos = StepPosition.from_dict(data)

        assert pos.x == 150
        assert pos.y == 250


# === Tests for StepUIMetadata ===


class TestStepUIMetadata:
    """Tests for StepUIMetadata model."""

    def test_create_defaults(self) -> None:
        ui = StepUIMetadata()

        assert ui.color is None
        assert ui.icon is None
        assert ui.collapsed is False

    def test_create_with_values(self) -> None:
        ui = StepUIMetadata(color="#FF0000", icon="star", collapsed=True)

        assert ui.color == "#FF0000"
        assert ui.icon == "star"
        assert ui.collapsed is True

    def test_to_dict(self) -> None:
        ui = StepUIMetadata(color="#00FF00", icon="check")
        data = ui.to_dict()

        assert data["color"] == "#00FF00"
        assert data["icon"] == "check"
        assert data["collapsed"] is False

    def test_from_dict(self) -> None:
        data = {"color": "#0000FF", "icon": "warning", "collapsed": True}
        ui = StepUIMetadata.from_dict(data)

        assert ui.color == "#0000FF"
        assert ui.icon == "warning"
        assert ui.collapsed is True


# === Tests for StepDefinition ===


class TestStepDefinition:
    """Tests for StepDefinition model."""

    def test_create_minimal(self) -> None:
        step = StepDefinition(step_id="step_1", type="llm_agent")

        assert step.step_id == "step_1"
        assert step.type == "llm_agent"
        assert step.name == ""
        assert step.config == {}
        assert step.input_ports == ["input"]
        assert step.output_ports == ["output"]

    def test_create_full(self) -> None:
        step = StepDefinition(
            step_id="validator",
            type="llm_agent",
            name="Input Validator",
            config={"llm_ref": "gpt4", "temperature": 0.1},
            input_ports=["data"],
            output_ports=["validated", "errors"],
            position=StepPosition(x=100, y=100),
            ui=StepUIMetadata(color="#4A90D9", icon="robot"),
        )

        assert step.step_id == "validator"
        assert step.name == "Input Validator"
        assert step.config["llm_ref"] == "gpt4"
        assert step.input_ports == ["data"]
        assert step.output_ports == ["validated", "errors"]
        assert step.position.x == 100
        assert step.ui.color == "#4A90D9"

    def test_to_dict(self) -> None:
        step = StepDefinition(
            step_id="step_1",
            type="llm_agent",
            name="Test Step",
            config={"key": "value"},
        )

        data = step.to_dict()

        assert data["step_id"] == "step_1"
        assert data["type"] == "llm_agent"
        assert data["name"] == "Test Step"
        assert data["config"] == {"key": "value"}
        assert data["ports"]["input"] == ["input"]
        assert data["ports"]["output"] == ["output"]

    def test_from_dict(self) -> None:
        data = {
            "step_id": "step_2",
            "type": "http_action",
            "name": "API Call",
            "config": {"path": "/api/v1"},
            "ports": {"input": ["body"], "output": ["response"]},
            "position": {"x": 200, "y": 300},
        }

        step = StepDefinition.from_dict(data)

        assert step.step_id == "step_2"
        assert step.type == "http_action"
        assert step.name == "API Call"
        assert step.input_ports == ["body"]
        assert step.output_ports == ["response"]
        assert step.position.x == 200

    def test_from_dict_minimal(self) -> None:
        data = {"step_id": "step_min", "type": "transform"}

        step = StepDefinition.from_dict(data)

        assert step.step_id == "step_min"
        assert step.type == "transform"
        assert step.input_ports == ["input"]
        assert step.output_ports == ["output"]


# === Tests for EdgeDefinition ===


class TestEdgeDefinition:
    """Tests for EdgeDefinition model."""

    def test_create_minimal(self) -> None:
        edge = EdgeDefinition(from_step="step_1", to_step="step_2")

        assert edge.from_step == "step_1"
        assert edge.to_step == "step_2"
        assert edge.from_port == "output"
        assert edge.to_port == "input"
        assert edge.condition is None

    def test_create_with_ports(self) -> None:
        edge = EdgeDefinition(
            from_step="validator",
            to_step="generator",
            from_port="validated",
            to_port="topic",
        )

        assert edge.from_port == "validated"
        assert edge.to_port == "topic"

    def test_create_with_condition(self) -> None:
        edge = EdgeDefinition(
            from_step="condition",
            to_step="handler",
            from_port="true",
            condition="result.status == 'success'",
        )

        assert edge.condition == "result.status == 'success'"

    def test_to_dict(self) -> None:
        edge = EdgeDefinition(from_step="a", to_step="b", from_port="out", to_port="in")

        data = edge.to_dict()

        assert data["from"] == "a"
        assert data["to"] == "b"
        assert data["from_port"] == "out"
        assert data["to_port"] == "in"

    def test_to_dict_with_condition(self) -> None:
        edge = EdgeDefinition(
            from_step="a", to_step="b", condition="x > 10"
        )

        data = edge.to_dict()

        assert data["condition"] == "x > 10"

    def test_from_dict(self) -> None:
        data = {
            "from": "step_a",
            "to": "step_b",
            "from_port": "success",
            "to_port": "data",
            "condition": "status == 'ok'",
        }

        edge = EdgeDefinition.from_dict(data)

        assert edge.from_step == "step_a"
        assert edge.to_step == "step_b"
        assert edge.from_port == "success"
        assert edge.to_port == "data"
        assert edge.condition == "status == 'ok'"


# === Tests for SegmentParams ===


class TestSegmentParams:
    """Tests for SegmentParams model."""

    def test_defaults(self) -> None:
        params = SegmentParams()

        assert params.max_retries == 3
        assert params.timeout_seconds == 300
        assert params.parallel_execution is False

    def test_custom(self) -> None:
        params = SegmentParams(
            max_retries=5, timeout_seconds=600, parallel_execution=True
        )

        assert params.max_retries == 5
        assert params.timeout_seconds == 600
        assert params.parallel_execution is True

    def test_to_dict(self) -> None:
        params = SegmentParams(max_retries=2, timeout_seconds=120)
        data = params.to_dict()

        assert data["max_retries"] == 2
        assert data["timeout_seconds"] == 120
        assert data["parallel_execution"] is False

    def test_from_dict(self) -> None:
        data = {"max_retries": 10, "timeout_seconds": 900, "parallel_execution": True}
        params = SegmentParams.from_dict(data)

        assert params.max_retries == 10
        assert params.timeout_seconds == 900
        assert params.parallel_execution is True


# === Tests for SegmentDefinition ===


class TestSegmentDefinition:
    """Tests for SegmentDefinition model."""

    def _create_simple_segment(self) -> SegmentDefinition:
        """Helper to create a simple valid segment."""
        return SegmentDefinition(
            segment_id="test_segment",
            name="Test Segment",
            entrypoint="step_1",
            steps=[
                StepDefinition(step_id="step_1", type="llm_agent"),
                StepDefinition(step_id="step_2", type="http_action"),
            ],
            edges=[
                EdgeDefinition(from_step="step_1", to_step="step_2"),
            ],
        )

    def test_create_minimal(self) -> None:
        segment = SegmentDefinition(
            segment_id="my_segment",
            name="My Segment",
            entrypoint="start",
            steps=[StepDefinition(step_id="start", type="llm_agent")],
        )

        assert segment.segment_id == "my_segment"
        assert segment.name == "My Segment"
        assert segment.entrypoint == "start"
        assert len(segment.steps) == 1
        assert segment.version == "1.0"

    def test_create_full(self) -> None:
        segment = SegmentDefinition(
            segment_id="content_pipeline",
            name="Content Pipeline",
            description="Generates content",
            entrypoint="validator",
            version="1.0",
            params=SegmentParams(max_retries=5, timeout_seconds=600),
            steps=[
                StepDefinition(step_id="validator", type="llm_agent"),
                StepDefinition(step_id="generator", type="llm_agent"),
            ],
            edges=[
                EdgeDefinition(from_step="validator", to_step="generator"),
            ],
            metadata={"author": "test@example.com"},
        )

        assert segment.description == "Generates content"
        assert segment.params.max_retries == 5
        assert segment.metadata["author"] == "test@example.com"

    def test_to_dict(self) -> None:
        segment = self._create_simple_segment()
        data = segment.to_dict()

        assert data["segment_id"] == "test_segment"
        assert data["name"] == "Test Segment"
        assert data["entrypoint"] == "step_1"
        assert len(data["steps"]) == 2
        assert len(data["edges"]) == 1
        assert data["version"] == "1.0"

    def test_to_json(self) -> None:
        segment = self._create_simple_segment()
        json_str = segment.to_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["segment_id"] == "test_segment"

    def test_from_dict(self) -> None:
        data = {
            "segment_id": "parsed_segment",
            "name": "Parsed Segment",
            "entrypoint": "start",
            "steps": [
                {"step_id": "start", "type": "llm_agent", "config": {"llm_ref": "gpt4"}},
                {"step_id": "end", "type": "http_action"},
            ],
            "edges": [{"from": "start", "to": "end"}],
            "params": {"max_retries": 5},
            "metadata": {"version": "test"},
        }

        segment = SegmentDefinition.from_dict(data)

        assert segment.segment_id == "parsed_segment"
        assert len(segment.steps) == 2
        assert segment.steps[0].config["llm_ref"] == "gpt4"
        assert len(segment.edges) == 1
        assert segment.params.max_retries == 5

    def test_from_json(self) -> None:
        json_str = """
        {
            "segment_id": "json_segment",
            "name": "JSON Segment",
            "entrypoint": "step_a",
            "steps": [
                {"step_id": "step_a", "type": "transform"}
            ]
        }
        """

        segment = SegmentDefinition.from_json(json_str)

        assert segment.segment_id == "json_segment"
        assert len(segment.steps) == 1

    def test_validate_valid_segment(self) -> None:
        segment = self._create_simple_segment()
        errors = segment.validate()

        assert errors == []

    def test_validate_invalid_entrypoint(self) -> None:
        segment = SegmentDefinition(
            segment_id="test",
            name="Test",
            entrypoint="nonexistent",
            steps=[StepDefinition(step_id="step_1", type="llm_agent")],
        )

        errors = segment.validate()

        assert len(errors) == 1
        assert "Entrypoint 'nonexistent' not found" in errors[0]

    def test_validate_invalid_edge_from(self) -> None:
        segment = SegmentDefinition(
            segment_id="test",
            name="Test",
            entrypoint="step_1",
            steps=[StepDefinition(step_id="step_1", type="llm_agent")],
            edges=[EdgeDefinition(from_step="unknown", to_step="step_1")],
        )

        errors = segment.validate()

        assert any("unknown" in e for e in errors)

    def test_validate_invalid_edge_to(self) -> None:
        segment = SegmentDefinition(
            segment_id="test",
            name="Test",
            entrypoint="step_1",
            steps=[StepDefinition(step_id="step_1", type="llm_agent")],
            edges=[EdgeDefinition(from_step="step_1", to_step="unknown")],
        )

        errors = segment.validate()

        assert any("unknown" in e for e in errors)

    def test_validate_invalid_step_id_format(self) -> None:
        segment = SegmentDefinition(
            segment_id="test",
            name="Test",
            entrypoint="Step1",  # Invalid: uppercase
            steps=[StepDefinition(step_id="Step1", type="llm_agent")],
        )

        errors = segment.validate()

        assert any("Invalid step_id format" in e for e in errors)

    def test_validate_invalid_segment_id_format(self) -> None:
        segment = SegmentDefinition(
            segment_id="123test",  # Invalid: starts with number
            name="Test",
            entrypoint="step_1",
            steps=[StepDefinition(step_id="step_1", type="llm_agent")],
        )

        errors = segment.validate()

        assert any("Invalid segment_id format" in e for e in errors)

    def test_get_step(self) -> None:
        segment = self._create_simple_segment()

        step = segment.get_step("step_1")
        assert step is not None
        assert step.type == "llm_agent"

        missing = segment.get_step("nonexistent")
        assert missing is None

    def test_get_outgoing_edges(self) -> None:
        segment = self._create_simple_segment()

        edges = segment.get_outgoing_edges("step_1")
        assert len(edges) == 1
        assert edges[0].to_step == "step_2"

    def test_get_incoming_edges(self) -> None:
        segment = self._create_simple_segment()

        edges = segment.get_incoming_edges("step_2")
        assert len(edges) == 1
        assert edges[0].from_step == "step_1"

    def test_get_next_steps(self) -> None:
        segment = self._create_simple_segment()

        next_steps = segment.get_next_steps("step_1")
        assert next_steps == ["step_2"]

    def test_get_previous_steps(self) -> None:
        segment = self._create_simple_segment()

        prev_steps = segment.get_previous_steps("step_2")
        assert prev_steps == ["step_1"]

    def test_roundtrip_json(self) -> None:
        """Test that to_json -> from_json preserves data."""
        original = SegmentDefinition(
            segment_id="roundtrip",
            name="Roundtrip Test",
            description="Testing JSON roundtrip",
            entrypoint="start",
            params=SegmentParams(max_retries=5, timeout_seconds=600),
            steps=[
                StepDefinition(
                    step_id="start",
                    type="llm_agent",
                    name="Start Step",
                    config={"llm_ref": "gpt4"},
                    input_ports=["data"],
                    output_ports=["result"],
                ),
            ],
            edges=[],
            metadata={"tag": "test"},
        )

        json_str = original.to_json()
        restored = SegmentDefinition.from_json(json_str)

        assert restored.segment_id == original.segment_id
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.params.max_retries == original.params.max_retries
        assert restored.steps[0].config == original.steps[0].config
        assert restored.metadata == original.metadata
