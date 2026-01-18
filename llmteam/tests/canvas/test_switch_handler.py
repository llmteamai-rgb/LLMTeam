"""Tests for SwitchHandler."""

import pytest
from unittest.mock import MagicMock

from llmteam.canvas.handlers.switch_handler import SwitchHandler
from llmteam.runtime import StepContext


@pytest.fixture
def handler():
    return SwitchHandler()


@pytest.fixture
def mock_context():
    ctx = MagicMock(spec=StepContext)
    ctx.step_id = "test_step"
    return ctx


class TestSwitchHandlerAttributes:
    """Tests for handler class attributes."""

    def test_handler_attributes(self, handler):
        assert handler.STEP_TYPE == "switch"
        assert handler.DISPLAY_NAME == "Switch"
        assert handler.CATEGORY == "flow_control"


class TestSwitchValueMatching:
    """Tests for value matching in switch handler."""

    @pytest.mark.asyncio
    async def test_simple_string_match(self, handler, mock_context):
        config = {
            "target": "update",
            "cases": [
                {"value": "create", "port": "create_handler"},
                {"value": "update", "port": "update_handler"},
                {"value": "delete", "port": "delete_handler"},
            ],
        }
        input_data = {"action": "update", "data": {"id": 1}}

        result = await handler(mock_context, config, input_data)

        assert "update_handler" in result
        assert result["update_handler"] == input_data

    @pytest.mark.asyncio
    async def test_integer_match(self, handler, mock_context):
        config = {
            "target": "404",
            "cases": [
                {"value": "200", "port": "success_handler"},
                {"value": "404", "port": "not_found_handler"},
                {"value": "500", "port": "error_handler"},
            ],
        }
        input_data = {"status_code": 404}

        result = await handler(mock_context, config, input_data)

        assert "not_found_handler" in result

    @pytest.mark.asyncio
    async def test_boolean_match(self, handler, mock_context):
        config = {
            "target": "True",
            "cases": [
                {"value": "True", "port": "valid_handler"},
                {"value": "False", "port": "invalid_handler"},
            ],
        }
        input_data = {"is_valid": True}

        result = await handler(mock_context, config, input_data)

        assert "valid_handler" in result


class TestSwitchDefaultBehavior:
    """Tests for default port behavior."""

    @pytest.mark.asyncio
    async def test_default_when_no_match(self, handler, mock_context):
        config = {
            "target": "delete",
            "cases": [
                {"value": "create", "port": "create_handler"},
                {"value": "update", "port": "update_handler"},
            ],
            "default_port": "unknown_handler",
        }
        input_data = {"action": "delete"}

        result = await handler(mock_context, config, input_data)

        assert "unknown_handler" in result
        assert result["unknown_handler"] == input_data

    @pytest.mark.asyncio
    async def test_no_match_uses_default_port(self, handler, mock_context):
        config = {
            "target": "unknown_action",
            "cases": [
                {"value": "create", "port": "create_handler"},
            ],
        }
        input_data = {"action": "delete"}

        result = await handler(mock_context, config, input_data)

        # Default port is "default" when not specified
        assert "default" in result


class TestSwitchExpressionMatching:
    """Tests for expression-based matching."""

    @pytest.mark.asyncio
    async def test_expression_match(self, handler, mock_context):
        config = {
            "target": 15,
            "cases": [
                {"expression": "value > 10", "port": "high_handler"},
                {"expression": "value <= 10", "port": "low_handler"},
            ],
        }
        input_data = {"count": 15}

        result = await handler(mock_context, config, input_data)

        assert "high_handler" in result

    @pytest.mark.asyncio
    async def test_expression_with_input_access(self, handler, mock_context):
        config = {
            "target": "test",
            "cases": [
                {"expression": "'urgent' in input.get('tags', [])", "port": "urgent_handler"},
                {"value": "test", "port": "normal_handler"},
            ],
        }
        input_data = {"tags": ["urgent", "important"]}

        result = await handler(mock_context, config, input_data)

        assert "urgent_handler" in result

    @pytest.mark.asyncio
    async def test_expression_error_skips_case(self, handler, mock_context):
        config = {
            "target": "test",
            "cases": [
                {"expression": "undefined_var > 10", "port": "error_handler"},
                {"value": "test", "port": "fallback_handler"},
            ],
        }
        input_data = {}

        result = await handler(mock_context, config, input_data)

        # Expression error should skip to next case
        assert "fallback_handler" in result


class TestSwitchFirstMatchWins:
    """Tests for first-match-wins behavior."""

    @pytest.mark.asyncio
    async def test_first_match_selected(self, handler, mock_context):
        config = {
            "target": "test",
            "cases": [
                {"value": "test", "port": "first_handler"},
                {"value": "test", "port": "second_handler"},
            ],
        }
        input_data = {}

        result = await handler(mock_context, config, input_data)

        # First matching case should win
        assert len(result) == 1
        assert "first_handler" in result

    @pytest.mark.asyncio
    async def test_stops_at_first_match(self, handler, mock_context):
        config = {
            "target": "a",
            "cases": [
                {"value": "a", "port": "port_a"},
                {"value": "b", "port": "port_b"},
                {"expression": "True", "port": "always_match"},
            ],
        }
        input_data = {"type": "a"}

        result = await handler(mock_context, config, input_data)

        assert len(result) == 1
        assert "port_a" in result


class TestSwitchMatchesMethod:
    """Tests for the _matches method."""

    def test_matches_exact_value(self, handler):
        case = {"value": "test", "port": "test_port"}
        assert handler._matches(case, "test", {}) is True
        assert handler._matches(case, "other", {}) is False

    def test_matches_numeric_as_string(self, handler):
        case = {"value": "42", "port": "numeric_port"}
        assert handler._matches(case, 42, {}) is True
        assert handler._matches(case, "42", {}) is True

    def test_matches_expression(self, handler):
        case = {"expression": "value == 'hello'", "port": "expr_port"}
        assert handler._matches(case, "hello", {}) is True
        assert handler._matches(case, "world", {}) is False

    def test_matches_empty_case(self, handler):
        case = {"port": "empty_port"}
        assert handler._matches(case, "anything", {}) is False


class TestSwitchEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_cases_list(self, handler, mock_context):
        config = {
            "target": "test",
            "cases": [],
            "default_port": "fallback",
        }
        input_data = {}

        result = await handler(mock_context, config, input_data)

        assert "fallback" in result

    @pytest.mark.asyncio
    async def test_none_target_value(self, handler, mock_context):
        config = {
            "target": None,
            "cases": [
                {"value": "None", "port": "none_handler"},
            ],
            "default_port": "default",
        }
        input_data = {}

        result = await handler(mock_context, config, input_data)

        # None converted to string "None" should match
        assert "none_handler" in result

    @pytest.mark.asyncio
    async def test_missing_port_in_case(self, handler, mock_context):
        config = {
            "target": "test",
            "cases": [
                {"value": "test"},  # Missing port
            ],
            "default_port": "default",
        }
        input_data = {}

        result = await handler(mock_context, config, input_data)

        # Should return None as port key
        assert None in result
