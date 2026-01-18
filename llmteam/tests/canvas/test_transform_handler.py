"""Tests for TransformHandler."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from llmteam.canvas.handlers import TransformHandler
from llmteam.runtime import StepContext


@pytest.fixture
def handler():
    """Create transform handler."""
    return TransformHandler()


@pytest.fixture
def mock_ctx():
    """Create mock step context."""
    ctx = MagicMock(spec=StepContext)
    ctx.step_id = "transform_step"
    ctx.run_id = "run_123"
    return ctx


class TestTransformHandler:
    """Tests for TransformHandler."""

    async def test_passthrough_no_config(self, handler, mock_ctx):
        """No expression or mapping passes data through."""
        input_data = {"key": "value", "number": 42}

        result = await handler(mock_ctx, {}, input_data)

        assert result["output"] == input_data

    async def test_simple_field_expression(self, handler, mock_ctx):
        """Access a simple field by name."""
        input_data = {"name": "Alice", "age": 30}

        result = await handler(mock_ctx, {"expression": "name"}, input_data)

        assert result["output"] == "Alice"

    async def test_nested_field_expression(self, handler, mock_ctx):
        """Access nested field with dot notation."""
        input_data = {"user": {"name": "Bob", "email": "bob@example.com"}}

        result = await handler(mock_ctx, {"expression": "user.name"}, input_data)

        assert result["output"] == "Bob"

    async def test_array_indexing(self, handler, mock_ctx):
        """Access array element by index."""
        input_data = {"items": ["apple", "banana", "cherry"]}

        result = await handler(mock_ctx, {"expression": "items[1]"}, input_data)

        assert result["output"] == "banana"

    async def test_keys_function(self, handler, mock_ctx):
        """Get dictionary keys."""
        input_data = {"a": 1, "b": 2, "c": 3}

        result = await handler(mock_ctx, {"expression": "keys()"}, input_data)

        assert set(result["output"]) == {"a", "b", "c"}

    async def test_values_function(self, handler, mock_ctx):
        """Get dictionary values."""
        input_data = {"a": 1, "b": 2}

        result = await handler(mock_ctx, {"expression": "values()"}, input_data)

        assert set(result["output"]) == {1, 2}

    async def test_len_function(self, handler, mock_ctx):
        """Get length of data."""
        input_data = {"items": [1, 2, 3, 4, 5]}

        result = await handler(mock_ctx, {"expression": "len()"}, {"items": [1, 2, 3, 4, 5]})

        # len() returns length of root dict
        assert result["output"] == 1

    async def test_json_function(self, handler, mock_ctx):
        """Serialize to JSON string."""
        input_data = {"key": "value"}

        result = await handler(mock_ctx, {"expression": "json()"}, input_data)

        assert '"key"' in result["output"]
        assert '"value"' in result["output"]

    async def test_field_mapping(self, handler, mock_ctx):
        """Apply field mapping."""
        input_data = {"firstName": "John", "lastName": "Doe"}
        mapping = {
            "full_name": "firstName",
            "family_name": "lastName",
        }

        result = await handler(mock_ctx, {"mapping": mapping}, input_data)

        assert result["output"]["full_name"] == "John"
        assert result["output"]["family_name"] == "Doe"

    async def test_default_value_on_error(self, handler, mock_ctx):
        """Return default value when expression fails."""
        input_data = {"key": "value"}

        result = await handler(
            mock_ctx,
            {"expression": "nonexistent.field", "default": "fallback"},
            input_data
        )

        assert result["output"] == "fallback"

    async def test_input_passthrough_expression(self, handler, mock_ctx):
        """'input' expression returns data as-is."""
        input_data = {"nested": {"data": [1, 2, 3]}}

        result = await handler(mock_ctx, {"expression": "input"}, input_data)

        assert result["output"] == input_data

    # === JSONPath Tests ===

    async def test_jsonpath_explicit_config(self, handler, mock_ctx):
        """Test explicit 'jsonpath' config key."""
        input_data = {
            "store": {
                "book": [
                    {"category": "fiction", "title": "Book 1"},
                    {"category": "reference", "title": "Book 2"}
                ]
            }
        }
        
        result = await handler(
            mock_ctx,
            {"jsonpath": "$.store.book[0].title"},
            input_data
        )
        
        assert result["output"] == "Book 1"

    async def test_jsonpath_implicit_expression(self, handler, mock_ctx):
        """Test JSONPath via 'expression' key (auto-detect)."""
        input_data = {
            "store": {
                "book": [
                    {"category": "fiction", "title": "Book 1"},
                    {"category": "reference", "title": "Book 2"}
                ]
            }
        }
        
        result = await handler(
            mock_ctx,
            {"expression": "$.store.book[1].title"},
            input_data
        )
        
        assert result["output"] == "Book 2"

    async def test_jsonpath_wildcard(self, handler, mock_ctx):
        """Test JSONPath wildcard returning list."""
        input_data = {
            "store": {
                "book": [
                    {"category": "fiction", "title": "Book 1"},
                    {"category": "reference", "title": "Book 2"}
                ]
            }
        }
        
        result = await handler(
            mock_ctx,
            {"jsonpath": "$.store.book[*].category"},
            input_data
        )
        
        # Order is usually preserved
        assert result["output"] == ["fiction", "reference"]

    async def test_jsonpath_default_value(self, handler, mock_ctx):
        """Test default value with JSONPath."""
        input_data = {"foo": "bar"}
        
        result = await handler(
            mock_ctx,
            {"jsonpath": "$.missing.key", "default": "not_found"},
            input_data
        )
        
        assert result["output"] == "not_found"
