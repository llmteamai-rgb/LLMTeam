"""Tests for SubworkflowHandler."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from llmteam.canvas.handlers.subworkflow_handler import SubworkflowHandler
from llmteam.canvas import SegmentDefinition, StepDefinition, EdgeDefinition, SegmentStatus
from llmteam.runtime import StepContext


@pytest.fixture
def handler():
    return SubworkflowHandler()


@pytest.fixture
def mock_context():
    ctx = MagicMock(spec=StepContext)
    ctx.step_id = "parent_step"
    ctx.runtime = MagicMock()
    ctx.runtime.child_context = MagicMock(return_value=MagicMock())
    return ctx


@pytest.fixture
def child_segment():
    return SegmentDefinition(
        segment_id="child-workflow",
        name="Child Workflow",
        entrypoint="start",
        steps=[
            StepDefinition(
                step_id="start",
                type="transform",
                config={"expression": "input"},
            )
        ],
        edges=[],
    )


class TestSubworkflowHandlerAttributes:
    """Tests for handler class attributes."""

    def test_handler_attributes(self, handler):
        assert handler.STEP_TYPE == "subworkflow"
        assert handler.DISPLAY_NAME == "Subworkflow"
        assert handler.CATEGORY == "flow_control"
        assert "nested" in handler.DESCRIPTION.lower()


class TestSubworkflowRegistration:
    """Tests for segment registration."""

    def test_register_segment(self, handler, child_segment):
        handler.register_segment("my-segment", child_segment)

        assert "my-segment" in handler._registry
        assert handler._registry["my-segment"] == child_segment

    def test_init_with_registry(self, child_segment):
        registry = {"preset-segment": child_segment}
        handler = SubworkflowHandler(segment_registry=registry)

        assert "preset-segment" in handler._registry


class TestSubworkflowInputMapping:
    """Tests for input mapping functionality."""

    def test_map_input_no_mapping(self, handler):
        data = {"field1": "value1", "field2": "value2"}
        mapping = {}

        result = handler._map_input(data, mapping)

        assert result == data
        assert result is not data  # Should be a copy

    def test_map_input_simple_mapping(self, handler):
        data = {"source_field": "test_value", "other": "data"}
        mapping = {"target_field": "source_field"}

        result = handler._map_input(data, mapping)

        assert result == {"target_field": "test_value"}

    def test_map_input_multiple_fields(self, handler):
        data = {"a": 1, "b": 2, "c": 3}
        mapping = {"x": "a", "y": "b"}

        result = handler._map_input(data, mapping)

        assert result == {"x": 1, "y": 2}

    def test_map_input_missing_source_field(self, handler):
        data = {"existing": "value"}
        mapping = {"target": "nonexistent"}

        result = handler._map_input(data, mapping)

        assert result == {}


class TestSubworkflowOutputMapping:
    """Tests for output mapping functionality."""

    def test_map_output_no_mapping(self, handler):
        data = {"result": "success"}
        mapping = {}

        result = handler._map_output(data, mapping)

        assert result == data

    def test_map_output_simple_mapping(self, handler):
        data = {"child_result": "value", "extra": "data"}
        mapping = {"parent_result": "child_result"}

        result = handler._map_output(data, mapping)

        assert result == {"parent_result": "value"}

    def test_map_output_nested_output_field(self, handler):
        data = {"output": {"inner_field": "nested_value"}}
        mapping = {"result": "inner_field"}

        result = handler._map_output(data, mapping)

        assert result == {"result": "nested_value"}


class TestSubworkflowSegmentResolution:
    """Tests for segment resolution."""

    def test_resolve_from_internal_registry_by_id(self, handler, child_segment, mock_context):
        handler.register_segment("my-segment", child_segment)

        resolved = handler._resolve_segment("my-segment", None, mock_context)

        assert resolved == child_segment

    def test_resolve_from_internal_registry_by_ref(self, handler, child_segment, mock_context):
        handler.register_segment("my-ref", child_segment)

        resolved = handler._resolve_segment(None, "my-ref", mock_context)

        assert resolved == child_segment

    def test_resolve_not_found(self, handler, mock_context):
        # Ensure runtime doesn't have segments attribute
        del mock_context.runtime.segments

        resolved = handler._resolve_segment("nonexistent", None, mock_context)

        assert resolved is None

    def test_resolve_from_runtime_segments(self, handler, child_segment, mock_context):
        mock_context.runtime.segments = {"runtime-segment": child_segment}

        resolved = handler._resolve_segment("runtime-segment", None, mock_context)

        assert resolved == child_segment


class TestSubworkflowExecution:
    """Tests for subworkflow execution."""

    @pytest.mark.asyncio
    async def test_segment_not_found_raises(self, handler, mock_context):
        # Ensure runtime doesn't have segments attribute
        del mock_context.runtime.segments

        config = {
            "segment_id": "nonexistent-segment",
        }
        input_data = {}

        with pytest.raises(ValueError, match="Subworkflow segment not found"):
            await handler(mock_context, config, input_data)

    @pytest.mark.asyncio
    async def test_successful_execution(self, handler, mock_context, child_segment):
        handler.register_segment("child", child_segment)
        config = {
            "segment_id": "child",
        }
        input_data = {"query": "test"}

        # Mock the runner
        mock_result = MagicMock()
        mock_result.status = SegmentStatus.COMPLETED
        mock_result.output = {"response": "success"}

        with patch.object(handler._runner, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result

            result = await handler(mock_context, config, input_data)

            assert result == {"response": "success"}
            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_execution_with_input_mapping(self, handler, mock_context, child_segment):
        handler.register_segment("child", child_segment)
        config = {
            "segment_id": "child",
            "input_mapping": {"child_input": "parent_input"},
        }
        input_data = {"parent_input": "hello"}

        mock_result = MagicMock()
        mock_result.status = SegmentStatus.COMPLETED
        mock_result.output = {"result": "ok"}

        with patch.object(handler._runner, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result

            await handler(mock_context, config, input_data)

            # Verify input was mapped
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["input_data"] == {"child_input": "hello"}

    @pytest.mark.asyncio
    async def test_execution_with_output_mapping(self, handler, mock_context, child_segment):
        handler.register_segment("child", child_segment)
        config = {
            "segment_id": "child",
            "output_mapping": {"final_result": "child_output"},
        }
        input_data = {}

        mock_result = MagicMock()
        mock_result.status = SegmentStatus.COMPLETED
        mock_result.output = {"child_output": "mapped_value"}

        with patch.object(handler._runner, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result

            result = await handler(mock_context, config, input_data)

            assert result == {"final_result": "mapped_value"}

    @pytest.mark.asyncio
    async def test_failed_subworkflow_returns_error(self, handler, mock_context, child_segment):
        handler.register_segment("child", child_segment)
        config = {
            "segment_id": "child",
        }
        input_data = {}

        mock_result = MagicMock()
        mock_result.status = SegmentStatus.FAILED
        mock_result.error = "Something went wrong"

        with patch.object(handler._runner, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result

            result = await handler(mock_context, config, input_data)

            assert "error" in result
            assert result["error"]["type"] == "SubworkflowFailed"

    @pytest.mark.asyncio
    async def test_exception_during_execution(self, handler, mock_context, child_segment):
        handler.register_segment("child", child_segment)
        config = {
            "segment_id": "child",
        }
        input_data = {}

        with patch.object(handler._runner, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = RuntimeError("Execution failed")

            result = await handler(mock_context, config, input_data)

            assert "error" in result
            assert result["error"]["type"] == "RuntimeError"
            assert "Execution failed" in result["error"]["message"]


class TestSubworkflowIsolation:
    """Tests for isolated execution mode."""

    @pytest.mark.asyncio
    async def test_isolated_creates_different_context(self, handler, mock_context, child_segment):
        handler.register_segment("child", child_segment)
        config = {
            "segment_id": "child",
            "isolated": True,
        }
        input_data = {}

        mock_result = MagicMock()
        mock_result.status = SegmentStatus.COMPLETED
        mock_result.output = {}

        with patch.object(handler._runner, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result

            await handler(mock_context, config, input_data)

            # Should create child context with "sub_" prefix
            mock_context.runtime.child_context.assert_called_with("sub_child-workflow")

    @pytest.mark.asyncio
    async def test_non_isolated_uses_segment_id(self, handler, mock_context, child_segment):
        handler.register_segment("child", child_segment)
        config = {
            "segment_id": "child",
            "isolated": False,
        }
        input_data = {}

        mock_result = MagicMock()
        mock_result.status = SegmentStatus.COMPLETED
        mock_result.output = {}

        with patch.object(handler._runner, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result

            await handler(mock_context, config, input_data)

            # Should create child context with segment_id
            mock_context.runtime.child_context.assert_called_with("child-workflow")


class TestSubworkflowSegmentRef:
    """Tests for segment_ref resolution."""

    @pytest.mark.asyncio
    async def test_segment_ref_resolution(self, handler, mock_context, child_segment):
        handler.register_segment("my-alias", child_segment)
        config = {
            "segment_ref": "my-alias",
        }
        input_data = {}

        mock_result = MagicMock()
        mock_result.status = SegmentStatus.COMPLETED
        mock_result.output = {"done": True}

        with patch.object(handler._runner, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result

            result = await handler(mock_context, config, input_data)

            assert result == {"done": True}
