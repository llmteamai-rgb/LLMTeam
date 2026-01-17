"""Tests for ConditionHandler."""

import pytest
from unittest.mock import MagicMock

from llmteam.canvas.handlers import ConditionHandler
from llmteam.runtime import StepContext


@pytest.fixture
def handler():
    """Create condition handler."""
    return ConditionHandler()


@pytest.fixture
def mock_ctx():
    """Create mock step context."""
    ctx = MagicMock(spec=StepContext)
    ctx.step_id = "condition_step"
    ctx.run_id = "run_123"
    return ctx


class TestConditionHandler:
    """Tests for ConditionHandler."""

    async def test_true_literal(self, handler, mock_ctx):
        """'true' literal returns true output."""
        input_data = {"key": "value"}

        result = await handler(mock_ctx, {"expression": "true"}, input_data)

        assert "true" in result
        assert result["true"] == input_data

    async def test_false_literal(self, handler, mock_ctx):
        """'false' literal returns false output."""
        input_data = {"key": "value"}

        result = await handler(mock_ctx, {"expression": "false"}, input_data)

        assert "false" in result
        assert result["false"] == input_data

    async def test_equality_comparison_true(self, handler, mock_ctx):
        """Equality comparison returns true when equal."""
        input_data = {"status": "active"}

        result = await handler(mock_ctx, {"expression": "status == 'active'"}, input_data)

        assert "true" in result

    async def test_equality_comparison_false(self, handler, mock_ctx):
        """Equality comparison returns false when not equal."""
        input_data = {"status": "inactive"}

        result = await handler(mock_ctx, {"expression": "status == 'active'"}, input_data)

        assert "false" in result

    async def test_inequality_comparison(self, handler, mock_ctx):
        """Inequality comparison works."""
        input_data = {"count": 5}

        result = await handler(mock_ctx, {"expression": "count != 0"}, input_data)

        assert "true" in result

    async def test_greater_than_true(self, handler, mock_ctx):
        """Greater than comparison returns true when greater."""
        input_data = {"score": 85}

        result = await handler(mock_ctx, {"expression": "score > 80"}, input_data)

        assert "true" in result

    async def test_greater_than_false(self, handler, mock_ctx):
        """Greater than comparison returns false when not greater."""
        input_data = {"score": 75}

        result = await handler(mock_ctx, {"expression": "score > 80"}, input_data)

        assert "false" in result

    async def test_less_than_or_equal(self, handler, mock_ctx):
        """Less than or equal comparison works."""
        input_data = {"age": 18}

        result = await handler(mock_ctx, {"expression": "age <= 18"}, input_data)

        assert "true" in result

    async def test_membership_in(self, handler, mock_ctx):
        """'in' operator checks membership."""
        input_data = {"role": "admin"}

        result = await handler(
            mock_ctx,
            {"expression": "role in ['admin', 'superuser']"},
            input_data
        )

        assert "true" in result

    async def test_membership_not_in(self, handler, mock_ctx):
        """'not in' operator checks non-membership."""
        input_data = {"role": "guest"}

        result = await handler(
            mock_ctx,
            {"expression": "role not in ['admin', 'superuser']"},
            input_data
        )

        assert "true" in result

    async def test_truthy_field_check_true(self, handler, mock_ctx):
        """Field name returns true if field is truthy."""
        input_data = {"enabled": True, "count": 5}

        result = await handler(mock_ctx, {"expression": "enabled"}, input_data)

        assert "true" in result

    async def test_truthy_field_check_false(self, handler, mock_ctx):
        """Field name returns false if field is falsy."""
        input_data = {"enabled": False}

        result = await handler(mock_ctx, {"expression": "enabled"}, input_data)

        assert "false" in result

    async def test_nested_field_comparison(self, handler, mock_ctx):
        """Compare nested field value."""
        input_data = {"user": {"role": "admin"}}

        result = await handler(
            mock_ctx,
            {"expression": "user.role == 'admin'"},
            input_data
        )

        assert "true" in result

    async def test_logical_and(self, handler, mock_ctx):
        """Logical AND combines conditions."""
        input_data = {"age": 25, "active": True}

        result = await handler(
            mock_ctx,
            {"expression": "age > 18 and active"},
            input_data
        )

        assert "true" in result

    async def test_logical_or(self, handler, mock_ctx):
        """Logical OR combines conditions."""
        input_data = {"role": "guest", "premium": True}

        result = await handler(
            mock_ctx,
            {"expression": "role == 'admin' or premium"},
            input_data
        )

        assert "true" in result

    async def test_logical_not(self, handler, mock_ctx):
        """Logical NOT negates condition."""
        input_data = {"disabled": False}

        result = await handler(mock_ctx, {"expression": "not disabled"}, input_data)

        assert "true" in result

    async def test_error_returns_false(self, handler, mock_ctx):
        """Error during evaluation returns false."""
        input_data = {"key": "value"}

        # Invalid expression that causes error
        result = await handler(mock_ctx, {"expression": "invalid[syntax"}, input_data)

        assert "false" in result

    async def test_contains_operator(self, handler, mock_ctx):
        """'contains' operator checks if value contains substring."""
        input_data = {"message": "Hello, World!"}

        result = await handler(
            mock_ctx,
            {"expression": "message contains 'World'"},
            input_data
        )

        assert "true" in result

    async def test_startswith_operator(self, handler, mock_ctx):
        """'startswith' operator checks string prefix."""
        input_data = {"url": "https://example.com"}

        result = await handler(
            mock_ctx,
            {"expression": "url startswith 'https'"},
            input_data
        )

        assert "true" in result

    async def test_endswith_operator(self, handler, mock_ctx):
        """'endswith' operator checks string suffix."""
        input_data = {"filename": "document.pdf"}

        result = await handler(
            mock_ctx,
            {"expression": "filename endswith '.pdf'"},
            input_data
        )

        assert "true" in result
