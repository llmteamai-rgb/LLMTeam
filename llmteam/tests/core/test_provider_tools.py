"""
Tests for RFC-015: Provider Function Calling.
Tests for RFC-016: Agent Tool Execution Loop.
Tests for RFC-017: Agent-level Token Streaming.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llmteam.providers.base import (
    BaseLLMProvider,
    ToolCall,
    LLMResponse,
    ToolMessage,
    CompletionConfig,
)
from llmteam.events.streaming import StreamEventType, StreamEvent


# ===== RFC-015: Provider Function Calling =====


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_creation(self):
        tc = ToolCall(id="call-1", name="get_weather", arguments={"city": "London"})
        assert tc.id == "call-1"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "London"}

    def test_default_arguments(self):
        tc = ToolCall(id="call-1", name="test")
        assert tc.arguments == {}


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_text_response(self):
        resp = LLMResponse(content="Hello world", finish_reason="stop")
        assert resp.content == "Hello world"
        assert resp.has_tool_calls is False
        assert resp.finish_reason == "stop"

    def test_tool_call_response(self):
        resp = LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(id="c1", name="search", arguments={"q": "test"}),
                ToolCall(id="c2", name="fetch", arguments={"url": "http://x.com"}),
            ],
            finish_reason="tool_calls",
        )
        assert resp.content is None
        assert resp.has_tool_calls is True
        assert len(resp.tool_calls) == 2
        assert resp.tool_calls[0].name == "search"

    def test_tokens(self):
        resp = LLMResponse(
            content="hi",
            tokens_used=150,
            input_tokens=100,
            output_tokens=50,
            model="gpt-4o",
        )
        assert resp.tokens_used == 150
        assert resp.input_tokens == 100
        assert resp.output_tokens == 50
        assert resp.model == "gpt-4o"

    def test_defaults(self):
        resp = LLMResponse()
        assert resp.content is None
        assert resp.tool_calls == []
        assert resp.tokens_used == 0
        assert resp.finish_reason == "stop"
        assert resp.has_tool_calls is False


class TestToolMessage:
    """Tests for ToolMessage dataclass."""

    def test_creation(self):
        msg = ToolMessage(tool_call_id="c1", name="search", content="result data")
        assert msg.tool_call_id == "c1"
        assert msg.name == "search"
        assert msg.content == "result data"


class TestBaseLLMProviderCompleteWithTools:
    """Tests for BaseLLMProvider.complete_with_tools() default implementation."""

    async def test_default_falls_back_to_complete_with_messages(self):
        """Default complete_with_tools should call complete_with_messages."""

        class MockProvider(BaseLLMProvider):
            async def complete(self, prompt, **kwargs):
                return "text response"

        provider = MockProvider(model="test")
        messages = [{"role": "user", "content": "Hello"}]

        resp = await provider.complete_with_tools(messages=messages)

        assert isinstance(resp, LLMResponse)
        assert resp.content == "text response"
        assert resp.has_tool_calls is False
        assert resp.finish_reason == "stop"

    async def test_with_tools_param(self):
        """Tools param should be accepted (even if ignored in default impl)."""

        class MockProvider(BaseLLMProvider):
            async def complete(self, prompt, **kwargs):
                return "ok"

        provider = MockProvider(model="test")
        tools = [{"type": "function", "function": {"name": "test"}}]

        resp = await provider.complete_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
        )
        assert resp.content == "ok"


# ===== RFC-016: Agent Tool Execution Loop =====


class TestLLMAgentToolLoop:
    """Tests for LLMAgent tool execution loop."""

    def _make_team_and_agent(self, tools=None, max_tool_rounds=5):
        """Helper to create a team with an LLM agent."""
        from llmteam import LLMTeam
        from llmteam.tools import ToolDefinition, ToolParameter, ParamType

        team = LLMTeam(team_id="test", orchestration=True)

        agent_config = {
            "type": "llm",
            "role": "assistant",
            "prompt": "Help: {query}",
            "max_tool_rounds": max_tool_rounds,
        }
        if tools:
            agent_config["tools"] = tools

        team.add_agent(agent_config)
        return team, team.get_agent("assistant")

    def test_max_tool_rounds_config(self):
        """max_tool_rounds should be configurable."""
        from llmteam import LLMTeam

        team = LLMTeam(team_id="test", orchestration=True)
        team.add_agent({
            "type": "llm",
            "role": "agent1",
            "prompt": "test",
            "max_tool_rounds": 10,
        })

        agent = team.get_agent("agent1")
        assert agent.max_tool_rounds == 10

    def test_default_max_tool_rounds(self):
        """Default max_tool_rounds should be 5."""
        from llmteam import LLMTeam

        team = LLMTeam(team_id="test", orchestration=True)
        team.add_agent({"type": "llm", "role": "agent1", "prompt": "test"})

        agent = team.get_agent("agent1")
        assert agent.max_tool_rounds == 5

    async def test_execute_without_tools_no_provider(self):
        """Without provider, should return fallback."""
        team, agent = self._make_team_and_agent()

        # Call internal _execute directly for testing
        result = await agent._execute({"query": "hello"}, {})

        assert result.success is True
        assert "LLM would generate response" in result.output

    async def test_execute_with_tools_no_provider(self):
        """With tools but no provider, should return fallback."""
        from llmteam.tools import ToolDefinition, ToolParameter, ParamType

        tool_def = ToolDefinition(
            name="greet", description="Greet",
            parameters=[ToolParameter(name="name", type=ParamType.STRING, required=True)],
            handler=lambda name: f"Hello {name}",
        )
        team, agent = self._make_team_and_agent(tools=[tool_def])

        result = await agent._execute({"query": "hello"}, {})

        assert result.success is True
        assert "LLM would generate response" in result.output

    async def test_tool_loop_single_round(self):
        """Tool loop: LLM calls tool once, then returns text."""
        from llmteam.tools import ToolDefinition, ToolParameter, ParamType

        tool_def = ToolDefinition(
            name="get_weather", description="Get weather",
            parameters=[ToolParameter(name="city", type=ParamType.STRING, required=True)],
            handler=lambda city: f"Sunny in {city}",
        )
        team, agent = self._make_team_and_agent(tools=[tool_def])

        # Mock provider
        mock_provider = AsyncMock()
        # First call: tool_call, second call: text response
        mock_provider.complete_with_tools = AsyncMock(side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="c1", name="get_weather", arguments={"city": "London"})],
                tokens_used=50,
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content="The weather in London is sunny.",
                tool_calls=[],
                tokens_used=30,
                finish_reason="stop",
            ),
        ])

        agent._get_provider = lambda model=None: mock_provider

        result = await agent._execute({"query": "weather in London"}, {})

        assert result.success is True
        assert "sunny" in result.output.lower()
        assert result.tokens_used == 80
        assert result.context_payload is not None
        assert len(result.context_payload["tool_calls_made"]) == 1
        assert result.context_payload["tool_calls_made"][0]["name"] == "get_weather"

    async def test_tool_loop_no_tools_called(self):
        """If LLM responds with text immediately, no loop."""
        from llmteam.tools import ToolDefinition, ToolParameter, ParamType

        tool_def = ToolDefinition(
            name="calc", description="Calculate",
            parameters=[ToolParameter(name="expr", type=ParamType.STRING, required=True)],
            handler=lambda expr: str(eval(expr)),
        )
        team, agent = self._make_team_and_agent(tools=[tool_def])

        mock_provider = AsyncMock()
        mock_provider.complete_with_tools = AsyncMock(return_value=LLMResponse(
            content="I don't need tools for this.",
            tokens_used=20,
            finish_reason="stop",
        ))
        agent._get_provider = lambda model=None: mock_provider

        result = await agent._execute({"query": "hi"}, {})

        assert result.success is True
        assert result.output == "I don't need tools for this."
        assert result.context_payload is None  # No tool calls

    async def test_tool_loop_max_rounds(self):
        """Loop should stop after max_tool_rounds."""
        from llmteam.tools import ToolDefinition, ToolParameter, ParamType

        tool_def = ToolDefinition(
            name="loop_tool", description="Always loops",
            parameters=[],
            handler=lambda: "looping",
        )
        team, agent = self._make_team_and_agent(tools=[tool_def], max_tool_rounds=3)

        mock_provider = AsyncMock()
        # Always returns tool calls (infinite loop scenario)
        mock_provider.complete_with_tools = AsyncMock(return_value=LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="c1", name="loop_tool", arguments={})],
            tokens_used=10,
            finish_reason="tool_calls",
        ))
        agent._get_provider = lambda model=None: mock_provider

        result = await agent._execute({"query": "loop"}, {})

        assert result.success is True
        assert result.context_payload["max_rounds_reached"] is True
        assert result.context_payload["tool_rounds"] == 3
        assert mock_provider.complete_with_tools.call_count == 3

    async def test_tool_loop_emits_events(self):
        """Tool loop should emit TOOL_CALL and TOOL_RESULT events."""
        from llmteam.tools import ToolDefinition, ToolParameter, ParamType

        tool_def = ToolDefinition(
            name="ping", description="Ping",
            parameters=[],
            handler=lambda: "pong",
        )
        team, agent = self._make_team_and_agent(tools=[tool_def])

        mock_provider = AsyncMock()
        mock_provider.complete_with_tools = AsyncMock(side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="c1", name="ping", arguments={})],
                tokens_used=10,
                finish_reason="tool_calls",
            ),
            LLMResponse(content="pong!", tokens_used=5, finish_reason="stop"),
        ])
        agent._get_provider = lambda model=None: mock_provider

        events = []

        async def capture_event(event_type, data, agent_id):
            events.append((event_type, data, agent_id))

        agent._event_callback = capture_event

        await agent._execute({"query": "ping"}, {})

        assert len(events) == 2
        assert events[0][0] == "tool_call"
        assert events[0][1]["tool_name"] == "ping"
        assert events[1][0] == "tool_result"
        assert events[1][1]["output"] == "pong"
        assert events[1][1]["success"] is True

    async def test_tool_execution_failure(self):
        """Tool execution failure should be reported in output."""
        from llmteam.tools import ToolDefinition, ToolParameter, ParamType

        def failing_handler(x: str) -> str:
            raise ValueError("Tool failed")

        tool_def = ToolDefinition(
            name="fail_tool", description="Fails",
            parameters=[ToolParameter(name="x", type=ParamType.STRING, required=True)],
            handler=failing_handler,
        )
        team, agent = self._make_team_and_agent(tools=[tool_def])

        mock_provider = AsyncMock()
        mock_provider.complete_with_tools = AsyncMock(side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="c1", name="fail_tool", arguments={"x": "test"})],
                tokens_used=10,
                finish_reason="tool_calls",
            ),
            LLMResponse(content="Tool failed, sorry.", tokens_used=10, finish_reason="stop"),
        ])
        agent._get_provider = lambda model=None: mock_provider

        result = await agent._execute({"query": "fail"}, {})

        assert result.success is True
        assert result.context_payload["tool_calls_made"][0]["success"] is False


# ===== RFC-017: Agent-level Token Streaming =====


class TestStreamEventTypeNew:
    """Tests for new StreamEventType values."""

    def test_tool_call_event(self):
        assert StreamEventType.TOOL_CALL == "tool_call"

    def test_tool_result_event(self):
        assert StreamEventType.TOOL_RESULT == "tool_result"

    def test_agent_thinking_event(self):
        assert StreamEventType.AGENT_THINKING == "agent_thinking"


class TestStreamToolEvents:
    """Tests for tool events in team.stream()."""

    async def test_stream_emits_tool_events(self):
        """team.stream() should emit TOOL_CALL/TOOL_RESULT from agent."""
        from llmteam import LLMTeam
        from llmteam.tools import ToolDefinition, ToolParameter, ParamType
        from llmteam.agents.orchestrator import RoutingDecision

        team = LLMTeam(team_id="test", orchestration=True)

        tool_def = ToolDefinition(
            name="calc", description="Calculate",
            parameters=[ToolParameter(name="expr", type=ParamType.STRING, required=True)],
            handler=lambda expr: "42",
        )
        team.add_agent({
            "type": "llm",
            "role": "agent1",
            "prompt": "Help: {query}",
            "tools": [tool_def],
        })

        # Mock orchestrator
        async def mock_decide(current_state, available_agents):
            return RoutingDecision(next_agent="agent1", reason="test")

        team._orchestrator.decide_next_agent = mock_decide

        # Mock provider on agent
        agent = team.get_agent("agent1")
        mock_provider = AsyncMock()
        mock_provider.complete_with_tools = AsyncMock(side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="c1", name="calc", arguments={"expr": "1+1"})],
                tokens_used=10,
                finish_reason="tool_calls",
            ),
            LLMResponse(content="The answer is 42.", tokens_used=10, finish_reason="stop"),
        ])
        agent._get_provider = lambda model=None: mock_provider

        events = []
        async for event in team.stream({"query": "calculate"}):
            events.append(event)

        event_types = [e.type for e in events]
        assert StreamEventType.TOOL_CALL in event_types
        assert StreamEventType.TOOL_RESULT in event_types

        # Check tool_call event data
        tool_call_event = next(e for e in events if e.type == StreamEventType.TOOL_CALL)
        assert tool_call_event.data["tool_name"] == "calc"
        assert tool_call_event.agent_id == "agent1"

        # Check tool_result event data
        tool_result_event = next(e for e in events if e.type == StreamEventType.TOOL_RESULT)
        assert tool_result_event.data["output"] == "42"
        assert tool_result_event.data["success"] is True

    async def test_stream_without_tools_no_tool_events(self):
        """stream() without tools should not emit tool events."""
        from llmteam import LLMTeam
        from llmteam.agents.orchestrator import RoutingDecision
        from llmteam.agents.result import AgentResult

        team = LLMTeam(team_id="test", orchestration=True)
        team.add_agent({"type": "llm", "role": "agent1", "prompt": "test"})

        async def mock_decide(current_state, available_agents):
            return RoutingDecision(next_agent="agent1", reason="test")

        team._orchestrator.decide_next_agent = mock_decide

        agent = team.get_agent("agent1")

        async def mock_execute(input_data, context, run_id=None):
            return AgentResult(output="hello", success=True)

        agent.execute = mock_execute

        events = []
        async for event in team.stream({"query": "hi"}):
            events.append(event)

        event_types = [e.type for e in events]
        assert StreamEventType.TOOL_CALL not in event_types
        assert StreamEventType.TOOL_RESULT not in event_types
