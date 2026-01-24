"""
Tests for RFC-013: Tool/Function Calling.
"""

import pytest
from typing import List, Optional, Dict

from llmteam import (
    LLMTeam,
    ParamType,
    ToolParameter,
    ToolDefinition,
    ToolResult,
    tool,
    ToolExecutor,
)


class TestParamType:
    """Tests for ParamType enum."""

    def test_basic_types(self):
        """ParamType should have basic types."""
        assert ParamType.STRING == "string"
        assert ParamType.INTEGER == "integer"
        assert ParamType.FLOAT == "float"
        assert ParamType.BOOLEAN == "boolean"
        assert ParamType.LIST == "list"
        assert ParamType.DICT == "dict"

    def test_from_python_type_str(self):
        """from_python_type should convert str."""
        assert ParamType.from_python_type(str) == ParamType.STRING

    def test_from_python_type_int(self):
        """from_python_type should convert int."""
        assert ParamType.from_python_type(int) == ParamType.INTEGER

    def test_from_python_type_float(self):
        """from_python_type should convert float."""
        assert ParamType.from_python_type(float) == ParamType.FLOAT

    def test_from_python_type_bool(self):
        """from_python_type should convert bool."""
        assert ParamType.from_python_type(bool) == ParamType.BOOLEAN

    def test_from_python_type_list(self):
        """from_python_type should convert List."""
        assert ParamType.from_python_type(List[str]) == ParamType.LIST

    def test_from_python_type_dict(self):
        """from_python_type should convert Dict."""
        assert ParamType.from_python_type(Dict[str, int]) == ParamType.DICT

    def test_from_python_type_optional(self):
        """from_python_type should unwrap Optional."""
        assert ParamType.from_python_type(Optional[int]) == ParamType.INTEGER

    def test_from_python_type_unknown(self):
        """from_python_type should default to STRING for unknown types."""
        class Custom:
            pass
        assert ParamType.from_python_type(Custom) == ParamType.STRING


class TestToolParameter:
    """Tests for ToolParameter."""

    def test_creation(self):
        """ToolParameter should be created with defaults."""
        param = ToolParameter(name="city")

        assert param.name == "city"
        assert param.type == ParamType.STRING
        assert param.required is True
        assert param.default is None

    def test_validate_string(self):
        """validate should coerce to string."""
        param = ToolParameter(name="x", type=ParamType.STRING)
        assert param.validate(123) == "123"

    def test_validate_integer(self):
        """validate should coerce to int."""
        param = ToolParameter(name="x", type=ParamType.INTEGER)
        assert param.validate("42") == 42

    def test_validate_float(self):
        """validate should coerce to float."""
        param = ToolParameter(name="x", type=ParamType.FLOAT)
        assert param.validate("3.14") == 3.14

    def test_validate_boolean_true(self):
        """validate should coerce to bool (true)."""
        param = ToolParameter(name="x", type=ParamType.BOOLEAN)
        assert param.validate("true") is True
        assert param.validate("yes") is True
        assert param.validate(1) is True

    def test_validate_boolean_false(self):
        """validate should coerce to bool (false)."""
        param = ToolParameter(name="x", type=ParamType.BOOLEAN)
        assert param.validate("false") is False
        assert param.validate("no") is False

    def test_validate_list(self):
        """validate should accept list."""
        param = ToolParameter(name="x", type=ParamType.LIST)
        assert param.validate([1, 2, 3]) == [1, 2, 3]

    def test_validate_list_rejects_non_list(self):
        """validate should reject non-list for LIST type."""
        param = ToolParameter(name="x", type=ParamType.LIST)
        with pytest.raises(TypeError):
            param.validate("not a list")

    def test_validate_dict(self):
        """validate should accept dict."""
        param = ToolParameter(name="x", type=ParamType.DICT)
        assert param.validate({"a": 1}) == {"a": 1}

    def test_validate_dict_rejects_non_dict(self):
        """validate should reject non-dict for DICT type."""
        param = ToolParameter(name="x", type=ParamType.DICT)
        with pytest.raises(TypeError):
            param.validate("not a dict")

    def test_validate_required_none(self):
        """validate should raise for required param with None."""
        param = ToolParameter(name="x", required=True)
        with pytest.raises(TypeError, match="required"):
            param.validate(None)

    def test_validate_optional_none(self):
        """validate should return default for optional param with None."""
        param = ToolParameter(name="x", required=False, default="default_val")
        assert param.validate(None) == "default_val"

    def test_to_dict(self):
        """to_dict should serialize parameter."""
        param = ToolParameter(
            name="city",
            type=ParamType.STRING,
            description="City name",
            required=True,
        )

        data = param.to_dict()

        assert data["name"] == "city"
        assert data["type"] == "string"
        assert data["description"] == "City name"
        assert data["required"] is True


class TestToolDefinition:
    """Tests for ToolDefinition."""

    def test_creation(self):
        """ToolDefinition should be created with name."""
        tool_def = ToolDefinition(name="get_weather")

        assert tool_def.name == "get_weather"
        assert tool_def.parameters == []
        assert tool_def.handler is None

    def test_to_schema(self):
        """to_schema should produce OpenAI-compatible schema."""
        tool_def = ToolDefinition(
            name="get_weather",
            description="Get current weather",
            parameters=[
                ToolParameter(name="city", type=ParamType.STRING, description="City name"),
                ToolParameter(name="units", type=ParamType.STRING, required=False, default="celsius"),
            ],
        )

        schema = tool_def.to_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"
        assert schema["function"]["description"] == "Get current weather"
        assert "city" in schema["function"]["parameters"]["properties"]
        assert "city" in schema["function"]["parameters"]["required"]
        assert "units" not in schema["function"]["parameters"]["required"]

    def test_validate_args(self):
        """validate_args should validate and coerce arguments."""
        tool_def = ToolDefinition(
            name="add",
            parameters=[
                ToolParameter(name="a", type=ParamType.INTEGER),
                ToolParameter(name="b", type=ParamType.INTEGER),
            ],
        )

        result = tool_def.validate_args({"a": "5", "b": "3"})

        assert result == {"a": 5, "b": 3}

    def test_validate_args_missing_required(self):
        """validate_args should raise on missing required param."""
        tool_def = ToolDefinition(
            name="test",
            parameters=[
                ToolParameter(name="x", required=True),
            ],
        )

        with pytest.raises(TypeError, match="required"):
            tool_def.validate_args({})

    def test_to_dict(self):
        """to_dict should serialize tool definition."""
        tool_def = ToolDefinition(
            name="search",
            description="Search for items",
            parameters=[
                ToolParameter(name="query", type=ParamType.STRING),
            ],
        )

        data = tool_def.to_dict()

        assert data["name"] == "search"
        assert data["description"] == "Search for items"
        assert len(data["parameters"]) == 1


class TestToolResult:
    """Tests for ToolResult."""

    def test_success(self):
        """ToolResult should indicate success."""
        result = ToolResult(tool_name="get_weather", output="Sunny, 22°C")

        assert result.success is True
        assert result.output == "Sunny, 22°C"
        assert result.error is None

    def test_failure(self):
        """ToolResult should indicate failure."""
        result = ToolResult(tool_name="get_weather", success=False, error="City not found")

        assert result.success is False
        assert result.error == "City not found"

    def test_to_dict(self):
        """ToolResult.to_dict should serialize."""
        result = ToolResult(tool_name="test", output=42)

        data = result.to_dict()

        assert data["tool_name"] == "test"
        assert data["output"] == 42
        assert data["success"] is True


class TestToolDecorator:
    """Tests for @tool decorator."""

    def test_basic_decorator(self):
        """@tool should create ToolDefinition from function."""
        @tool()
        def greet(name: str) -> str:
            """Say hello."""
            return f"Hello, {name}!"

        assert hasattr(greet, "tool_definition")
        assert greet.tool_definition.name == "greet"
        assert greet.tool_definition.description == "Say hello."
        assert len(greet.tool_definition.parameters) == 1
        assert greet.tool_definition.parameters[0].name == "name"
        assert greet.tool_definition.parameters[0].type == ParamType.STRING

    def test_custom_name(self):
        """@tool should accept custom name."""
        @tool(name="my_tool")
        def something():
            pass

        assert something.tool_definition.name == "my_tool"

    def test_custom_description(self):
        """@tool should accept custom description."""
        @tool(description="Custom desc")
        def something():
            """This is ignored."""
            pass

        assert something.tool_definition.description == "Custom desc"

    def test_multiple_params(self):
        """@tool should handle multiple parameters."""
        @tool()
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        params = add.tool_definition.parameters
        assert len(params) == 2
        assert params[0].name == "a"
        assert params[0].type == ParamType.INTEGER
        assert params[1].name == "b"
        assert params[1].type == ParamType.INTEGER

    def test_optional_params(self):
        """@tool should detect optional parameters."""
        @tool()
        def search(query: str, limit: int = 10):
            """Search items."""
            pass

        params = search.tool_definition.parameters
        assert params[0].required is True
        assert params[1].required is False
        assert params[1].default == 10

    def test_handler_is_set(self):
        """@tool should set handler to the function."""
        @tool()
        def my_func():
            return 42

        assert my_func.tool_definition.handler is my_func

    def test_function_still_callable(self):
        """Decorated function should still be callable."""
        @tool()
        def add(a: int, b: int) -> int:
            return a + b

        assert add(3, 4) == 7

    def test_list_param_type(self):
        """@tool should handle List parameter type."""
        @tool()
        def process(items: List[str]):
            pass

        assert process.tool_definition.parameters[0].type == ParamType.LIST

    def test_dict_param_type(self):
        """@tool should handle Dict parameter type."""
        @tool()
        def process(data: Dict[str, int]):
            pass

        assert process.tool_definition.parameters[0].type == ParamType.DICT


class TestToolExecutor:
    """Tests for ToolExecutor."""

    def _make_tool(self, name="test_tool", handler=None):
        """Helper to create a tool."""
        if handler is None:
            def handler(x: str) -> str:
                return f"result: {x}"

        return ToolDefinition(
            name=name,
            description="Test tool",
            parameters=[
                ToolParameter(name="x", type=ParamType.STRING),
            ],
            handler=handler,
        )

    def test_register(self):
        """register should add tool."""
        executor = ToolExecutor()
        tool_def = self._make_tool()
        executor.register(tool_def)

        assert executor.get_tool("test_tool") is tool_def

    def test_register_duplicate_raises(self):
        """register should raise on duplicate name."""
        executor = ToolExecutor()
        executor.register(self._make_tool())

        with pytest.raises(ValueError, match="already registered"):
            executor.register(self._make_tool())

    def test_unregister(self):
        """unregister should remove tool."""
        executor = ToolExecutor()
        executor.register(self._make_tool())

        assert executor.unregister("test_tool") is True
        assert executor.get_tool("test_tool") is None

    def test_unregister_not_found(self):
        """unregister should return False if not found."""
        executor = ToolExecutor()
        assert executor.unregister("nonexistent") is False

    def test_list_tools(self):
        """list_tools should return registered names."""
        executor = ToolExecutor()
        executor.register(self._make_tool("tool1"))
        executor.register(self._make_tool("tool2"))

        assert set(executor.list_tools()) == {"tool1", "tool2"}

    def test_get_schemas(self):
        """get_schemas should return all tool schemas."""
        executor = ToolExecutor()
        executor.register(self._make_tool("tool1"))
        executor.register(self._make_tool("tool2"))

        schemas = executor.get_schemas()
        assert len(schemas) == 2
        assert schemas[0]["type"] == "function"

    async def test_execute_sync_function(self):
        """execute should handle sync functions."""
        def add(x: str) -> str:
            return f"got: {x}"

        tool_def = ToolDefinition(
            name="add",
            parameters=[ToolParameter(name="x", type=ParamType.STRING)],
            handler=add,
        )
        executor = ToolExecutor(tools=[tool_def])

        result = await executor.execute("add", {"x": "hello"})

        assert result.success is True
        assert result.output == "got: hello"

    async def test_execute_async_function(self):
        """execute should handle async functions."""
        async def fetch(x: str) -> str:
            return f"fetched: {x}"

        tool_def = ToolDefinition(
            name="fetch",
            parameters=[ToolParameter(name="x", type=ParamType.STRING)],
            handler=fetch,
        )
        executor = ToolExecutor(tools=[tool_def])

        result = await executor.execute("fetch", {"x": "data"})

        assert result.success is True
        assert result.output == "fetched: data"

    async def test_execute_not_found(self):
        """execute should return error for unknown tool."""
        executor = ToolExecutor()

        result = await executor.execute("unknown", {})

        assert result.success is False
        assert "not found" in result.error

    async def test_execute_no_handler(self):
        """execute should return error for tool without handler."""
        tool_def = ToolDefinition(name="empty")
        executor = ToolExecutor(tools=[tool_def])

        result = await executor.execute("empty", {})

        assert result.success is False
        assert "no handler" in result.error

    async def test_execute_validation_error(self):
        """execute should return error on invalid args."""
        tool_def = ToolDefinition(
            name="test",
            parameters=[ToolParameter(name="x", type=ParamType.INTEGER, required=True)],
            handler=lambda x: x,
        )
        executor = ToolExecutor(tools=[tool_def])

        result = await executor.execute("test", {})  # Missing required 'x'

        assert result.success is False
        assert "required" in result.error

    async def test_execute_handler_exception(self):
        """execute should catch handler exceptions."""
        def bad_func(x: str):
            raise ValueError("something wrong")

        tool_def = ToolDefinition(
            name="bad",
            parameters=[ToolParameter(name="x", type=ParamType.STRING)],
            handler=bad_func,
        )
        executor = ToolExecutor(tools=[tool_def])

        result = await executor.execute("bad", {"x": "test"})

        assert result.success is False
        assert "ValueError" in result.error
        assert "something wrong" in result.error

    async def test_execute_records_history(self):
        """execute should record call history."""
        def echo(x: str) -> str:
            return x

        tool_def = ToolDefinition(
            name="echo",
            parameters=[ToolParameter(name="x", type=ParamType.STRING)],
            handler=echo,
        )
        executor = ToolExecutor(tools=[tool_def])

        await executor.execute("echo", {"x": "hello"})
        await executor.execute("echo", {"x": "world"})

        assert len(executor.call_history) == 2
        assert executor.call_history[0]["tool_name"] == "echo"

    async def test_clear_history(self):
        """clear_history should reset call history."""
        def echo(x: str) -> str:
            return x

        tool_def = ToolDefinition(
            name="echo",
            parameters=[ToolParameter(name="x", type=ParamType.STRING)],
            handler=echo,
        )
        executor = ToolExecutor(tools=[tool_def])

        await executor.execute("echo", {"x": "test"})
        executor.clear_history()

        assert len(executor.call_history) == 0

    async def test_on_call_callback(self):
        """on_call callback should fire on each execution."""
        calls = []

        def echo(x: str) -> str:
            return x

        tool_def = ToolDefinition(
            name="echo",
            parameters=[ToolParameter(name="x", type=ParamType.STRING)],
            handler=echo,
        )
        executor = ToolExecutor(
            tools=[tool_def],
            on_call=lambda name, args: calls.append((name, args)),
        )

        await executor.execute("echo", {"x": "hi"})

        assert len(calls) == 1
        assert calls[0][0] == "echo"

    async def test_init_with_tools(self):
        """ToolExecutor should accept initial tools list."""
        tool1 = ToolDefinition(name="t1", handler=lambda: 1)
        tool2 = ToolDefinition(name="t2", handler=lambda: 2)

        executor = ToolExecutor(tools=[tool1, tool2])

        assert len(executor.list_tools()) == 2


class TestAgentToolIntegration:
    """Tests for per-agent tool integration."""

    def test_agent_with_tools_from_definition(self):
        """Agent should accept tools as ToolDefinition list."""
        @tool()
        def search(query: str) -> str:
            return f"results for {query}"

        team = LLMTeam(team_id="test")
        agent = team.add_agent({
            "type": "llm",
            "role": "searcher",
            "prompt": "test",
            "tools": [search.tool_definition],
        })

        assert agent.tool_executor is not None
        assert "search" in agent.tool_executor.list_tools()

    def test_agent_with_tools_from_dict(self):
        """Agent should accept tools as dict list."""
        team = LLMTeam(team_id="test")
        agent = team.add_agent({
            "type": "llm",
            "role": "searcher",
            "prompt": "test",
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": [
                        {"name": "city", "type": "string", "description": "City name"},
                    ],
                },
            ],
        })

        assert agent.tool_executor is not None
        assert "get_weather" in agent.tool_executor.list_tools()

    def test_agent_without_tools(self):
        """Agent without tools should have None tool_executor."""
        team = LLMTeam(team_id="test")
        agent = team.add_agent({
            "type": "llm",
            "role": "basic",
            "prompt": "test",
        })

        assert agent.tool_executor is None

    async def test_agent_tool_execution(self):
        """Agent's tool executor should be callable."""
        @tool()
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        team = LLMTeam(team_id="test")
        agent = team.add_agent({
            "type": "llm",
            "role": "calc",
            "prompt": "test",
            "tools": [multiply.tool_definition],
        })

        result = await agent.tool_executor.execute("multiply", {"a": 3, "b": 4})

        assert result.success is True
        assert result.output == 12

    def test_agent_tool_schemas(self):
        """Agent's tool executor should provide schemas."""
        @tool(description="Get weather for city")
        def get_weather(city: str, units: str = "celsius") -> str:
            return "sunny"

        team = LLMTeam(team_id="test")
        agent = team.add_agent({
            "type": "llm",
            "role": "weather",
            "prompt": "test",
            "tools": [get_weather.tool_definition],
        })

        schemas = agent.tool_executor.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "get_weather"
        assert "city" in schemas[0]["function"]["parameters"]["properties"]
