"""Tests for LLMAgentHandler."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from llmteam.engine.handlers import LLMAgentHandler
from llmteam.runtime import StepContext


@pytest.fixture
def handler():
    """Create LLM agent handler."""
    return LLMAgentHandler()


@pytest.fixture
def mock_ctx():
    """Create mock step context with LLM provider."""
    ctx = MagicMock(spec=StepContext)
    ctx.step_id = "llm_step"
    ctx.run_id = "run_123"

    # Mock LLM provider
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value="LLM response text")
    ctx.get_llm = MagicMock(return_value=mock_llm)

    return ctx


class TestLLMAgentHandler:
    """Tests for LLMAgentHandler."""

    async def test_basic_execution(self, handler, mock_ctx):
        """Basic LLM execution returns response."""
        config = {"llm_ref": "default"}
        input_data = {"prompt": "Hello, world!"}

        result = await handler(mock_ctx, config, input_data)

        assert "output" in result
        assert result["output"] == "LLM response text"
        mock_ctx.get_llm.assert_called_once_with("default")

    async def test_with_temperature(self, handler, mock_ctx):
        """Temperature is passed to LLM."""
        config = {"llm_ref": "gpt4", "temperature": 0.5}
        input_data = {"prompt": "Test"}

        await handler(mock_ctx, config, input_data)

        mock_llm = mock_ctx.get_llm.return_value
        mock_llm.complete.assert_called_once()
        call_kwargs = mock_llm.complete.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5

    async def test_with_max_tokens(self, handler, mock_ctx):
        """Max tokens is passed to LLM."""
        config = {"llm_ref": "gpt4", "max_tokens": 500}
        input_data = {"prompt": "Test"}

        await handler(mock_ctx, config, input_data)

        mock_llm = mock_ctx.get_llm.return_value
        call_kwargs = mock_llm.complete.call_args.kwargs
        assert call_kwargs["max_tokens"] == 500

    async def test_with_system_prompt(self, handler, mock_ctx):
        """System prompt is passed to LLM."""
        config = {
            "llm_ref": "default",
            "system_prompt": "You are a helpful assistant.",
        }
        input_data = {"prompt": "Hello"}

        await handler(mock_ctx, config, input_data)

        mock_llm = mock_ctx.get_llm.return_value
        call_kwargs = mock_llm.complete.call_args.kwargs
        assert call_kwargs["system_prompt"] == "You are a helpful assistant."

    async def test_direct_prompt_substitution(self, handler, mock_ctx):
        """Direct prompt with variable substitution."""
        config = {
            "llm_ref": "default",
            "prompt": "Hello, {name}!",
        }
        input_data = {"name": "Alice"}

        await handler(mock_ctx, config, input_data)

        mock_llm = mock_ctx.get_llm.return_value
        call_kwargs = mock_llm.complete.call_args.kwargs
        assert call_kwargs["prompt"] == "Hello, Alice!"

    async def test_input_with_query_key(self, handler, mock_ctx):
        """Input with 'query' key is used as prompt."""
        config = {"llm_ref": "default"}
        input_data = {"query": "What is the meaning of life?"}

        await handler(mock_ctx, config, input_data)

        mock_llm = mock_ctx.get_llm.return_value
        call_kwargs = mock_llm.complete.call_args.kwargs
        assert "What is the meaning of life?" in call_kwargs["prompt"]

    async def test_input_with_text_key(self, handler, mock_ctx):
        """Input with 'text' key is used as prompt."""
        config = {"llm_ref": "default"}
        input_data = {"text": "Analyze this text."}

        await handler(mock_ctx, config, input_data)

        mock_llm = mock_ctx.get_llm.return_value
        call_kwargs = mock_llm.complete.call_args.kwargs
        assert "Analyze this text." in call_kwargs["prompt"]

    async def test_llm_error_returns_error_info(self, handler, mock_ctx):
        """LLM error returns error info in output."""
        mock_llm = mock_ctx.get_llm.return_value
        mock_llm.complete = AsyncMock(side_effect=Exception("API Error"))

        config = {"llm_ref": "default"}
        input_data = {"prompt": "Test"}

        result = await handler(mock_ctx, config, input_data)

        assert "error" in result
        assert result["error"]["type"] == "Exception"
        assert "API Error" in result["error"]["message"]
        assert result["output"] == ""

    async def test_default_temperature(self, handler, mock_ctx):
        """Default temperature is used when not specified."""
        config = {"llm_ref": "default"}
        input_data = {"prompt": "Test"}

        await handler(mock_ctx, config, input_data)

        mock_llm = mock_ctx.get_llm.return_value
        call_kwargs = mock_llm.complete.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7  # default

    async def test_default_max_tokens(self, handler, mock_ctx):
        """Default max_tokens is used when not specified."""
        config = {"llm_ref": "default"}
        input_data = {"prompt": "Test"}

        await handler(mock_ctx, config, input_data)

        mock_llm = mock_ctx.get_llm.return_value
        call_kwargs = mock_llm.complete.call_args.kwargs
        assert call_kwargs["max_tokens"] == 1000  # default

    async def test_custom_handler_defaults(self, mock_ctx):
        """Handler with custom defaults uses those values."""
        handler = LLMAgentHandler(
            default_temperature=0.3,
            default_max_tokens=2000,
        )
        config = {"llm_ref": "default"}
        input_data = {"prompt": "Test"}

        await handler(mock_ctx, config, input_data)

        mock_llm = mock_ctx.get_llm.return_value
        call_kwargs = mock_llm.complete.call_args.kwargs
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_tokens"] == 2000

    async def test_json_input_serialization(self, handler, mock_ctx):
        """Complex input is serialized to JSON."""
        config = {"llm_ref": "default"}
        input_data = {"data": {"nested": [1, 2, 3]}}

        await handler(mock_ctx, config, input_data)

        mock_llm = mock_ctx.get_llm.return_value
        call_kwargs = mock_llm.complete.call_args.kwargs
        # Should contain JSON representation
        assert "nested" in call_kwargs["prompt"]
