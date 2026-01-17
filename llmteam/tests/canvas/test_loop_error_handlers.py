"""Tests for LoopHandler and ErrorHandler."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from llmteam.canvas.handlers.loop_handler import LoopHandler
from llmteam.canvas.handlers.error_handler import ErrorHandler, TryCatchHandler


class TestLoopHandler:
    """Tests for LoopHandler."""

    def test_handler_metadata(self):
        """Handler has correct metadata."""
        handler = LoopHandler()

        assert handler.STEP_TYPE == "loop"
        assert handler.DISPLAY_NAME == "Loop"
        assert handler.CATEGORY == "flow_control"

    async def test_foreach_loop(self):
        """For-each loop iterates over collection."""
        handler = LoopHandler()
        ctx = MagicMock()
        ctx.step_id = "loop1"

        config = {
            "loop_type": "foreach",
            "collection_field": "items",
            "item_variable": "item",
        }
        input_data = {"items": [1, 2, 3]}

        result = await handler(ctx, config, input_data)

        assert result["iterations"] == 3
        assert result["completed"] is True
        assert len(result["results"]) == 3

    async def test_foreach_empty_collection(self):
        """For-each handles empty collection."""
        handler = LoopHandler()
        ctx = MagicMock()

        config = {"loop_type": "foreach", "collection_field": "items"}
        input_data = {"items": []}

        result = await handler(ctx, config, input_data)

        assert result["iterations"] == 0
        assert result["results"] == []

    async def test_foreach_max_iterations(self):
        """For-each respects max_iterations."""
        handler = LoopHandler()
        ctx = MagicMock()

        config = {
            "loop_type": "foreach",
            "collection_field": "items",
            "max_iterations": 2,
        }
        input_data = {"items": [1, 2, 3, 4, 5]}

        result = await handler(ctx, config, input_data)

        assert result["iterations"] == 2
        assert result["completed"] is False

    async def test_range_loop(self):
        """Range loop iterates over numeric range."""
        handler = LoopHandler()
        ctx = MagicMock()

        config = {
            "loop_type": "range",
            "range_start": 0,
            "range_end": 5,
            "range_step": 1,
        }
        input_data = {}

        result = await handler(ctx, config, input_data)

        assert result["iterations"] == 5

    async def test_range_loop_step(self):
        """Range loop respects step."""
        handler = LoopHandler()
        ctx = MagicMock()

        config = {
            "loop_type": "range",
            "range_start": 0,
            "range_end": 10,
            "range_step": 2,
        }
        input_data = {}

        result = await handler(ctx, config, input_data)

        assert result["iterations"] == 5

    async def test_while_loop_condition(self):
        """While loop checks condition."""
        handler = LoopHandler()
        ctx = MagicMock()

        config = {
            "loop_type": "while",
            "condition": "false",  # Immediately false
        }
        input_data = {}

        result = await handler(ctx, config, input_data)

        assert result["iterations"] == 0

    async def test_until_loop_terminates(self):
        """Until loop terminates when condition is true."""
        handler = LoopHandler()
        ctx = MagicMock()

        config = {
            "loop_type": "until",
            "condition": "true",  # Immediately true
            "max_iterations": 10,
        }
        input_data = {}

        result = await handler(ctx, config, input_data)

        assert result["iterations"] == 1

    def test_evaluate_condition_true_literal(self):
        """Evaluate condition handles true literal."""
        handler = LoopHandler()

        assert handler._evaluate_condition("true", {}) is True
        assert handler._evaluate_condition("True", {}) is True

    def test_evaluate_condition_false_literal(self):
        """Evaluate condition handles false literal."""
        handler = LoopHandler()

        assert handler._evaluate_condition("false", {}) is False
        assert handler._evaluate_condition("False", {}) is False

    def test_evaluate_condition_equality(self):
        """Evaluate condition handles equality."""
        handler = LoopHandler()

        assert handler._evaluate_condition("x == 5", {"x": 5}) is True
        assert handler._evaluate_condition("x == 5", {"x": 3}) is False

    def test_evaluate_condition_inequality(self):
        """Evaluate condition handles inequality."""
        handler = LoopHandler()

        assert handler._evaluate_condition("x != 5", {"x": 3}) is True
        assert handler._evaluate_condition("x != 5", {"x": 5}) is False

    def test_evaluate_condition_comparison(self):
        """Evaluate condition handles comparisons."""
        handler = LoopHandler()

        assert handler._evaluate_condition("x > 5", {"x": 10}) is True
        assert handler._evaluate_condition("x < 5", {"x": 3}) is True
        assert handler._evaluate_condition("x >= 5", {"x": 5}) is True
        assert handler._evaluate_condition("x <= 5", {"x": 5}) is True


class TestErrorHandler:
    """Tests for ErrorHandler."""

    def test_handler_metadata(self):
        """Handler has correct metadata."""
        handler = ErrorHandler()

        assert handler.STEP_TYPE == "error_handler"
        assert handler.DISPLAY_NAME == "Error Handler"
        assert handler.CATEGORY == "flow_control"

    async def test_catch_mode_success(self):
        """Catch mode passes through on success."""
        handler = ErrorHandler()
        ctx = MagicMock()
        ctx.step_id = "error1"
        ctx.metadata = {}

        config = {"mode": "catch"}
        input_data = {"body_result": {"data": "result"}}

        result = await handler(ctx, config, input_data)

        assert result["success"] is True
        assert result["result"] == {"data": "result"}
        assert result["error"] is None

    async def test_fallback_mode_success(self):
        """Fallback mode passes through on success."""
        handler = ErrorHandler()
        ctx = MagicMock()
        ctx.metadata = {}

        config = {"mode": "fallback", "fallback_value": {"default": True}}
        input_data = {"body_result": {"data": "result"}}

        result = await handler(ctx, config, input_data)

        assert result["success"] is True
        assert result["result"] == {"data": "result"}
        assert result["used_fallback"] is False

    async def test_retry_mode_success_first_try(self):
        """Retry mode succeeds on first try."""
        handler = ErrorHandler()
        ctx = MagicMock()
        ctx.metadata = {}

        config = {"mode": "retry", "max_retries": 3}
        input_data = {"body_result": {"data": "result"}}

        result = await handler(ctx, config, input_data)

        assert result["success"] is True
        assert result["retries"] == 0

    async def test_retry_mode_simulated_failure(self):
        """Retry mode handles simulated failure."""
        handler = ErrorHandler()
        ctx = MagicMock()
        ctx.metadata = {}

        config = {
            "mode": "retry",
            "max_retries": 3,
            "retry_delay_ms": 1,  # Fast for testing
        }
        input_data = {
            "_simulate_failure": True,
            "_fail_count": 2,
            "body_result": {"data": "result"},
        }

        result = await handler(ctx, config, input_data)

        assert result["success"] is True
        # Retries = attempts - 1, and it succeeds on attempt 3
        assert result["retries"] >= 2

    async def test_compensate_mode_success(self):
        """Compensate mode passes through on success."""
        handler = ErrorHandler()
        ctx = MagicMock()
        ctx.metadata = {}

        config = {"mode": "compensate"}
        input_data = {"body_result": {"data": "result"}}

        result = await handler(ctx, config, input_data)

        assert result["success"] is True
        assert result["compensated"] is False

    def test_create_error_info(self):
        """Create error info from exception."""
        handler = ErrorHandler()
        config = handler._parse_config({})
        error = ValueError("test error")

        info = handler._create_error_info(error, config)

        assert info["type"] == "ValueError"
        assert info["message"] == "test error"
        assert "timestamp" in info


class TestTryCatchHandler:
    """Tests for TryCatchHandler."""

    def test_handler_metadata(self):
        """Handler has correct metadata."""
        handler = TryCatchHandler()

        assert handler.STEP_TYPE == "try_catch"
        assert handler.DISPLAY_NAME == "Try-Catch"
        assert handler.CATEGORY == "flow_control"

    async def test_try_success(self):
        """Try block succeeds."""
        handler = TryCatchHandler()
        ctx = MagicMock()

        config = {}
        input_data = {"try_input": {"data": "result"}}

        result = await handler(ctx, config, input_data)

        assert result["success"] is True
        assert result["result"] == {"data": "result"}
        assert result["error"] is None
        assert result["caught"] is False

    async def test_finally_executes(self):
        """Finally block executes."""
        handler = TryCatchHandler()
        ctx = MagicMock()

        config = {"finally_step": "cleanup"}
        input_data = {"try_input": {"data": "result"}}

        result = await handler(ctx, config, input_data)

        assert result["finally_result"] == {"executed": True}
