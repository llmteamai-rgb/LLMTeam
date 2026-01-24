"""
Tests for RFC-011: Streaming Output (EventEmitter + SecureBus integration).
"""

import pytest
from datetime import datetime

from llmteam import (
    LLMTeam,
    StreamEventType,
    StreamEvent,
)
from llmteam.agents.orchestrator import OrchestratorConfig, OrchestratorMode, RoutingDecision


class TestStreamEventType:
    """Tests for StreamEventType enum."""

    def test_run_lifecycle_events(self):
        """StreamEventType should have run lifecycle events."""
        assert StreamEventType.RUN_STARTED == "run_started"
        assert StreamEventType.RUN_COMPLETED == "run_completed"
        assert StreamEventType.RUN_FAILED == "run_failed"

    def test_agent_lifecycle_events(self):
        """StreamEventType should have agent lifecycle events."""
        assert StreamEventType.AGENT_STARTED == "agent_started"
        assert StreamEventType.AGENT_COMPLETED == "agent_completed"
        assert StreamEventType.AGENT_FAILED == "agent_failed"

    def test_token_streaming_events(self):
        """StreamEventType should have token streaming events."""
        assert StreamEventType.TOKEN == "token"
        assert StreamEventType.CHUNK == "chunk"

    def test_progress_event(self):
        """StreamEventType should have progress event."""
        assert StreamEventType.PROGRESS == "progress"

    def test_cost_event(self):
        """StreamEventType should have cost update event."""
        assert StreamEventType.COST_UPDATE == "cost_update"


class TestStreamEvent:
    """Tests for StreamEvent dataclass."""

    def test_creation(self):
        """StreamEvent should be created with required fields."""
        event = StreamEvent(type=StreamEventType.RUN_STARTED)

        assert event.type == StreamEventType.RUN_STARTED
        assert event.data == {}
        assert event.run_id is None
        assert event.agent_id is None
        assert isinstance(event.timestamp, datetime)

    def test_creation_with_data(self):
        """StreamEvent should accept data and ids."""
        event = StreamEvent(
            type=StreamEventType.AGENT_COMPLETED,
            data={"output": "hello"},
            run_id="run-1",
            agent_id="agent1",
        )

        assert event.data == {"output": "hello"}
        assert event.run_id == "run-1"
        assert event.agent_id == "agent1"

    def test_to_dict(self):
        """StreamEvent.to_dict should serialize all fields."""
        event = StreamEvent(
            type=StreamEventType.TOKEN,
            data={"token": "Hello"},
            run_id="run-1",
            agent_id="agent1",
        )

        data = event.to_dict()

        assert data["type"] == "token"
        assert data["data"] == {"token": "Hello"}
        assert data["run_id"] == "run-1"
        assert data["agent_id"] == "agent1"
        assert "timestamp" in data

    def test_to_sse(self):
        """StreamEvent.to_sse should format as SSE string."""
        event = StreamEvent(
            type=StreamEventType.RUN_STARTED,
            data={"team_id": "test"},
            run_id="run-1",
        )

        sse = event.to_sse()

        assert sse.startswith("event: run_started\n")
        assert "data: " in sse
        assert sse.endswith("\n\n")

    def test_to_dict_with_string_type(self):
        """to_dict should handle string type gracefully."""
        event = StreamEvent(
            type=StreamEventType.PROGRESS,
            data={"step": 1},
        )

        data = event.to_dict()
        assert data["type"] == "progress"


class TestLLMTeamStream:
    """Tests for LLMTeam.stream() method."""

    async def test_stream_no_agents(self):
        """stream() with no agents should yield RUN_FAILED."""
        team = LLMTeam(team_id="test", orchestration=True)

        events = []
        async for event in team.stream({"query": "test"}):
            events.append(event)

        assert len(events) == 1
        assert events[0].type == StreamEventType.RUN_FAILED
        assert "No agents" in events[0].data["error"]

    async def test_stream_not_router_mode(self):
        """stream() without ROUTER mode should yield RUN_FAILED."""
        team = LLMTeam(team_id="test")
        team.add_agent({"type": "llm", "role": "agent1", "prompt": "test"})

        events = []
        async for event in team.stream({"query": "test"}):
            events.append(event)

        assert len(events) == 1
        assert events[0].type == StreamEventType.RUN_FAILED
        assert "ROUTER mode" in events[0].data["error"]

    async def test_stream_successful_run(self):
        """stream() should emit lifecycle events for successful run."""
        team = LLMTeam(team_id="test", orchestration=True)
        team.add_agent({"type": "llm", "role": "agent1", "prompt": "test"})

        # Mock orchestrator
        async def mock_decide(current_state, available_agents):
            return RoutingDecision(
                next_agent=available_agents[0] if available_agents else "",
                reason="mock routing",
            )

        team._orchestrator.decide_next_agent = mock_decide

        # Mock agent execution
        agent = team.get_agent("agent1")

        async def mock_execute(input_data, context, run_id=None):
            from llmteam import AgentResult
            return AgentResult(output="result", tokens_used=100, model="gpt-4o-mini")

        agent.execute = mock_execute

        events = []
        async for event in team.stream({"query": "test"}):
            events.append(event)

        # Should have: RUN_STARTED, AGENT_STARTED, COST_UPDATE, AGENT_COMPLETED, RUN_COMPLETED
        event_types = [e.type for e in events]
        assert StreamEventType.RUN_STARTED in event_types
        assert StreamEventType.AGENT_STARTED in event_types
        assert StreamEventType.AGENT_COMPLETED in event_types
        assert StreamEventType.RUN_COMPLETED in event_types

    async def test_stream_run_id_propagated(self):
        """stream() should propagate run_id to all events."""
        team = LLMTeam(team_id="test", orchestration=True)
        team.add_agent({"type": "llm", "role": "agent1", "prompt": "test"})

        async def mock_decide(current_state, available_agents):
            return RoutingDecision(
                next_agent=available_agents[0] if available_agents else "",
                reason="test",
            )

        team._orchestrator.decide_next_agent = mock_decide

        agent = team.get_agent("agent1")

        async def mock_execute(input_data, context, run_id=None):
            from llmteam import AgentResult
            return AgentResult(output="ok")

        agent.execute = mock_execute

        events = []
        async for event in team.stream({"query": "test"}, run_id="my-run"):
            events.append(event)

        for event in events:
            assert event.run_id == "my-run"

    async def test_stream_agent_id_on_agent_events(self):
        """Agent events should have agent_id set."""
        team = LLMTeam(team_id="test", orchestration=True)
        team.add_agent({"type": "llm", "role": "agent1", "prompt": "test"})

        async def mock_decide(current_state, available_agents):
            return RoutingDecision(
                next_agent=available_agents[0] if available_agents else "",
                reason="test",
            )

        team._orchestrator.decide_next_agent = mock_decide

        agent = team.get_agent("agent1")

        async def mock_execute(input_data, context, run_id=None):
            from llmteam import AgentResult
            return AgentResult(output="ok")

        agent.execute = mock_execute

        events = []
        async for event in team.stream({"query": "test"}):
            events.append(event)

        agent_events = [
            e for e in events
            if e.type in (StreamEventType.AGENT_STARTED, StreamEventType.AGENT_COMPLETED)
        ]
        for event in agent_events:
            assert event.agent_id == "agent1"

    async def test_stream_cost_update(self):
        """stream() should emit COST_UPDATE with token usage."""
        team = LLMTeam(team_id="test", orchestration=True)
        team.add_agent({"type": "llm", "role": "agent1", "prompt": "test"})

        async def mock_decide(current_state, available_agents):
            return RoutingDecision(
                next_agent=available_agents[0] if available_agents else "",
                reason="test",
            )

        team._orchestrator.decide_next_agent = mock_decide

        agent = team.get_agent("agent1")

        async def mock_execute(input_data, context, run_id=None):
            from llmteam import AgentResult
            return AgentResult(output="ok", tokens_used=500, model="gpt-4o-mini")

        agent.execute = mock_execute

        events = []
        async for event in team.stream({"query": "test"}):
            events.append(event)

        cost_events = [e for e in events if e.type == StreamEventType.COST_UPDATE]
        assert len(cost_events) == 1
        assert cost_events[0].data["tokens"] == 500
        assert cost_events[0].data["current_cost"] > 0

    async def test_stream_agent_failed(self):
        """stream() should emit AGENT_FAILED for unsuccessful agent."""
        team = LLMTeam(team_id="test", orchestration=True)
        team.add_agent({"type": "llm", "role": "agent1", "prompt": "test"})

        call_count = [0]

        async def mock_decide(current_state, available_agents):
            call_count[0] += 1
            if call_count[0] > 1:
                return RoutingDecision(next_agent="", reason="done")
            return RoutingDecision(
                next_agent=available_agents[0] if available_agents else "",
                reason="test",
            )

        team._orchestrator.decide_next_agent = mock_decide

        agent = team.get_agent("agent1")

        async def mock_execute(input_data, context, run_id=None):
            from llmteam import AgentResult
            return AgentResult(output=None, success=False, error="something failed")

        agent.execute = mock_execute

        events = []
        async for event in team.stream({"query": "test"}):
            events.append(event)

        event_types = [e.type for e in events]
        assert StreamEventType.AGENT_FAILED in event_types
        failed_event = next(e for e in events if e.type == StreamEventType.AGENT_FAILED)
        assert "something failed" in failed_event.data["error"]

    async def test_stream_run_completed_data(self):
        """RUN_COMPLETED event should include output and agents_called."""
        team = LLMTeam(team_id="test", orchestration=True)
        team.add_agent({"type": "llm", "role": "agent1", "prompt": "test"})

        async def mock_decide(current_state, available_agents):
            return RoutingDecision(
                next_agent=available_agents[0] if available_agents else "",
                reason="test",
            )

        team._orchestrator.decide_next_agent = mock_decide

        agent = team.get_agent("agent1")

        async def mock_execute(input_data, context, run_id=None):
            from llmteam import AgentResult
            return AgentResult(output="final result")

        agent.execute = mock_execute

        events = []
        async for event in team.stream({"query": "test"}):
            events.append(event)

        completed = next(e for e in events if e.type == StreamEventType.RUN_COMPLETED)
        assert completed.data["success"] is True
        assert completed.data["output"] == "final result"
        assert "agent1" in completed.data["agents_called"]

    async def test_stream_exception_handling(self):
        """stream() should emit RUN_FAILED on exception."""
        team = LLMTeam(team_id="test", orchestration=True)
        team.add_agent({"type": "llm", "role": "agent1", "prompt": "test"})

        async def mock_decide(current_state, available_agents):
            raise RuntimeError("orchestrator crash")

        team._orchestrator.decide_next_agent = mock_decide

        events = []
        async for event in team.stream({"query": "test"}):
            events.append(event)

        event_types = [e.type for e in events]
        assert StreamEventType.RUN_STARTED in event_types
        assert StreamEventType.RUN_FAILED in event_types
        failed = next(e for e in events if e.type == StreamEventType.RUN_FAILED)
        assert "orchestrator crash" in failed.data["error"]

    async def test_stream_auto_generates_run_id(self):
        """stream() should auto-generate run_id if not provided."""
        team = LLMTeam(team_id="test", orchestration=True)
        team.add_agent({"type": "llm", "role": "agent1", "prompt": "test"})

        async def mock_decide(current_state, available_agents):
            return RoutingDecision(
                next_agent=available_agents[0] if available_agents else "",
                reason="test",
            )

        team._orchestrator.decide_next_agent = mock_decide

        agent = team.get_agent("agent1")

        async def mock_execute(input_data, context, run_id=None):
            from llmteam import AgentResult
            return AgentResult(output="ok")

        agent.execute = mock_execute

        events = []
        async for event in team.stream({"query": "test"}):
            events.append(event)

        # All events should have the same non-None run_id
        run_ids = set(e.run_id for e in events)
        assert len(run_ids) == 1
        assert None not in run_ids

    async def test_stream_run_started_has_agents_list(self):
        """RUN_STARTED event should list available agents."""
        team = LLMTeam(team_id="test", orchestration=True)
        team.add_agent({"type": "llm", "role": "agent1", "prompt": "test"})
        team.add_agent({"type": "llm", "role": "agent2", "prompt": "test"})

        async def mock_decide(current_state, available_agents):
            return RoutingDecision(
                next_agent=available_agents[0] if available_agents else "",
                reason="test",
            )

        team._orchestrator.decide_next_agent = mock_decide

        agent1 = team.get_agent("agent1")

        async def mock_execute(input_data, context, run_id=None):
            from llmteam import AgentResult
            return AgentResult(output="ok")

        agent1.execute = mock_execute

        events = []
        async for event in team.stream({"query": "test"}):
            events.append(event)

        started = next(e for e in events if e.type == StreamEventType.RUN_STARTED)
        assert "agent1" in started.data["agents"]
        assert "agent2" in started.data["agents"]
        assert started.data["team_id"] == "test"
