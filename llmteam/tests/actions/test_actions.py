"""
Tests for External Actions module (v1.9.0).
"""

import pytest

from llmteam.actions import (
    ActionType,
    ActionStatus,
    ActionConfig,
    ActionContext,
    ActionResult,
    ActionRegistry,
    ActionExecutor,
)


class TestActionModels:
    """Tests for action data models."""

    def test_action_config(self):
        """Test ActionConfig creation."""
        config = ActionConfig(
            name="test_action",
            action_type=ActionType.FUNCTION,
            timeout_seconds=10.0,
        )

        assert config.name == "test_action"
        assert config.action_type == ActionType.FUNCTION
        assert config.timeout_seconds == 10.0

    def test_action_context(self):
        """Test ActionContext creation."""
        context = ActionContext(
            action_name="test",
            run_id="run_123",
            agent_name="agent_1",
            tenant_id="tenant_1",
            input_data={"key": "value"},
        )

        assert context.action_name == "test"
        assert context.input_data["key"] == "value"


class TestActionRegistry:
    """Tests for ActionRegistry."""

    def test_register_function(self):
        """Test registering a function action."""
        registry = ActionRegistry()

        def my_function(input_data, pipeline_state):
            return {"result": "success"}

        registry.register_function("my_func", my_function)

        assert registry.has_action("my_func")
        assert "my_func" in registry.list_actions()

    def test_unregister(self):
        """Test unregistering an action."""
        registry = ActionRegistry()

        def my_function(input_data, pipeline_state):
            return {}

        registry.register_function("my_func", my_function)
        registry.unregister("my_func")

        assert not registry.has_action("my_func")


class TestActionExecutor:
    """Tests for ActionExecutor."""

    @pytest.mark.asyncio
    async def test_execute_function(self):
        """Test executing a function action."""
        registry = ActionRegistry()

        def my_function(input_data, pipeline_state):
            return {"result": input_data.get("value", 0) * 2}

        registry.register_function("double", my_function)

        executor = ActionExecutor(registry)
        context = ActionContext(
            action_name="double",
            run_id="run_123",
            agent_name="agent_1",
            tenant_id="tenant_1",
            input_data={"value": 5},
        )

        result = await executor.execute("double", context)

        assert result.status == ActionStatus.COMPLETED
        assert result.response_data["result"] == 10

    @pytest.mark.asyncio
    async def test_execute_not_found(self):
        """Test executing non-existent action."""
        registry = ActionRegistry()
        executor = ActionExecutor(registry)

        context = ActionContext(
            action_name="missing",
            run_id="run_123",
            agent_name="agent_1",
            tenant_id="tenant_1",
        )

        result = await executor.execute("missing", context)

        assert result.status == ActionStatus.FAILED
        assert "not found" in result.error_message.lower()
