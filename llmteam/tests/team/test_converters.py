"""Tests for llmteam.team.converters module."""

import pytest
from unittest.mock import MagicMock

from llmteam.team.converters import (
    parse_flow_string,
    parse_flow_dict,
    build_segment,
    result_from_segment_result,
    create_sequential_edges,
    agents_to_steps,
)
from llmteam.agents.types import AgentType
from llmteam.canvas.runner import SegmentResult, SegmentStatus


class TestParseFlowString:
    """Tests for parse_flow_string function."""

    def test_simple_sequential(self):
        """Test parsing simple sequential flow."""
        edges = parse_flow_string("a -> b -> c", agent_ids=["a", "b", "c"])

        assert len(edges) == 2
        assert edges[0].from_step == "a"
        assert edges[0].to_step == "b"
        assert edges[1].from_step == "b"
        assert edges[1].to_step == "c"

    def test_single_edge(self):
        """Test parsing single edge."""
        edges = parse_flow_string("input -> output", agent_ids=["input", "output"])

        assert len(edges) == 1
        assert edges[0].from_step == "input"
        assert edges[0].to_step == "output"

    def test_whitespace_handling(self):
        """Test whitespace is handled correctly."""
        edges = parse_flow_string("  a  ->  b  ->  c  ", agent_ids=["a", "b", "c"])

        assert len(edges) == 2
        assert edges[0].from_step == "a"
        assert edges[0].to_step == "b"
        assert edges[1].from_step == "b"
        assert edges[1].to_step == "c"

    def test_parallel_notation(self):
        """Test parsing parallel notation with comma."""
        edges = parse_flow_string("a, b -> c", agent_ids=["a", "b", "c"])

        # a and b both connect to c
        assert len(edges) == 2
        from_steps = [e.from_step for e in edges]
        to_steps = [e.to_step for e in edges]
        assert "a" in from_steps
        assert "b" in from_steps
        assert all(t == "c" for t in to_steps)

    def test_sequential_keyword(self):
        """Test 'sequential' creates edges through all agents."""
        edges = parse_flow_string("sequential", agent_ids=["a", "b", "c"])
        assert len(edges) == 2

    def test_empty_string(self):
        """Test empty string creates sequential edges."""
        edges = parse_flow_string("", agent_ids=["a", "b"])
        assert len(edges) == 1
        assert edges[0].from_step == "a"
        assert edges[0].to_step == "b"


class TestParseFlowDict:
    """Tests for parse_flow_dict function."""

    def test_edges_list(self):
        """Test parsing edges from dict."""
        flow = {
            "edges": [
                {"from_step": "a", "to_step": "b"},
                {"from_step": "b", "to_step": "c"},
            ]
        }
        edges = parse_flow_dict(flow)

        assert len(edges) == 2
        assert edges[0].from_step == "a"
        assert edges[0].to_step == "b"
        assert edges[1].from_step == "b"
        assert edges[1].to_step == "c"

    def test_edges_with_conditions(self):
        """Test parsing edges with conditions."""
        flow = {
            "edges": [
                {"from_step": "check", "to_step": "yes", "condition": "result.approved"},
                {"from_step": "check", "to_step": "no", "condition": "not result.approved"},
            ]
        }
        edges = parse_flow_dict(flow)

        assert len(edges) == 2
        assert edges[0].condition == "result.approved"
        assert edges[1].condition == "not result.approved"

    def test_empty_edges(self):
        """Test empty edges list."""
        edges = parse_flow_dict({"edges": []})
        assert edges == []

    def test_no_edges_key(self):
        """Test dict without edges key."""
        edges = parse_flow_dict({})
        assert edges == []

    def test_alternate_key_names(self):
        """Test 'from' and 'to' key names."""
        flow = {
            "edges": [
                {"from": "a", "to": "b"},
            ]
        }
        edges = parse_flow_dict(flow)

        assert len(edges) == 1
        assert edges[0].from_step == "a"
        assert edges[0].to_step == "b"


class TestCreateSequentialEdges:
    """Tests for create_sequential_edges function."""

    def test_create_edges(self):
        """Test creating sequential edges."""
        edges = create_sequential_edges(["a", "b", "c"])

        assert len(edges) == 2
        assert edges[0].from_step == "a"
        assert edges[0].to_step == "b"
        assert edges[1].from_step == "b"
        assert edges[1].to_step == "c"

    def test_single_agent(self):
        """Test single agent has no edges."""
        edges = create_sequential_edges(["a"])
        assert edges == []

    def test_empty_list(self):
        """Test empty list has no edges."""
        edges = create_sequential_edges([])
        assert edges == []


class TestAgentsToSteps:
    """Tests for agents_to_steps function."""

    def test_convert_llm_agent(self):
        """Test converting LLM agent to step."""
        agent = MagicMock()
        agent.agent_type = AgentType.LLM
        agent.role = "writer"
        agent.name = "Writer Agent"
        agent.description = "Writes content"
        agent.prompt = "Write: {input}"
        agent.model = "gpt-4o"
        agent.temperature = 0.7

        steps = agents_to_steps({"writer": agent})

        assert len(steps) == 1
        assert steps[0].step_id == "writer"
        assert steps[0].type == "llm_agent"  # LLM agents map to llm_agent handler
        assert steps[0].config["role"] == "writer"
        assert steps[0].config["prompt"] == "Write: {input}"

    def test_convert_rag_agent(self):
        """Test converting RAG agent to step."""
        agent = MagicMock()
        agent.agent_type = AgentType.RAG
        agent.role = "retriever"
        agent.name = "Retriever"
        agent.description = "Retrieves documents"
        agent.collection = "docs"
        agent.top_k = 5

        steps = agents_to_steps({"retriever": agent})

        assert len(steps) == 1
        assert steps[0].type == "rag"
        assert steps[0].config["collection"] == "docs"
        assert steps[0].config["top_k"] == 5


class TestBuildSegment:
    """Tests for build_segment function."""

    def test_build_simple_segment(self):
        """Test building segment from agents."""
        # Create mock agents
        agent1 = MagicMock()
        agent1.agent_id = "writer"
        agent1.role = "writer"
        agent1.name = "Writer"
        agent1.description = ""
        agent1.agent_type = AgentType.LLM
        agent1.prompt = "Write"
        agent1.model = "gpt-4o"
        agent1.temperature = 0.7

        agent2 = MagicMock()
        agent2.agent_id = "reviewer"
        agent2.role = "reviewer"
        agent2.name = "Reviewer"
        agent2.description = ""
        agent2.agent_type = AgentType.LLM
        agent2.prompt = "Review"
        agent2.model = "gpt-4o"
        agent2.temperature = 0.7

        agents = {"writer": agent1, "reviewer": agent2}

        segment = build_segment(
            team_id="test_team",
            agents=agents,
            flow="writer -> reviewer",
        )

        assert segment.segment_id == "test_team"
        assert len(segment.steps) == 2
        assert len(segment.edges) == 1

    def test_build_sequential_segment(self):
        """Test building sequential segment."""
        agent1 = MagicMock()
        agent1.agent_id = "a"
        agent1.role = "a"
        agent1.name = "A"
        agent1.description = ""
        agent1.agent_type = AgentType.LLM
        agent1.prompt = "A"
        agent1.model = "gpt-4o"
        agent1.temperature = 0.7

        agent2 = MagicMock()
        agent2.agent_id = "b"
        agent2.role = "b"
        agent2.name = "B"
        agent2.description = ""
        agent2.agent_type = AgentType.LLM
        agent2.prompt = "B"
        agent2.model = "gpt-4o"
        agent2.temperature = 0.7

        agents = {"a": agent1, "b": agent2}

        segment = build_segment(
            team_id="test",
            agents=agents,
            flow="sequential",
        )

        # Sequential should auto-chain agents
        assert segment.segment_id == "test"
        assert len(segment.steps) == 2
        assert segment.entrypoint == "a"

    def test_build_with_dict_flow(self):
        """Test building segment with dict flow."""
        agent1 = MagicMock()
        agent1.agent_id = "start"
        agent1.role = "start"
        agent1.name = "Start"
        agent1.description = ""
        agent1.agent_type = AgentType.LLM
        agent1.prompt = "Start"
        agent1.model = "gpt-4o"
        agent1.temperature = 0.7

        agent2 = MagicMock()
        agent2.agent_id = "end"
        agent2.role = "end"
        agent2.name = "End"
        agent2.description = ""
        agent2.agent_type = AgentType.LLM
        agent2.prompt = "End"
        agent2.model = "gpt-4o"
        agent2.temperature = 0.7

        agents = {"start": agent1, "end": agent2}

        flow = {
            "edges": [
                {"from_step": "start", "to_step": "end"},
            ]
        }

        segment = build_segment(
            team_id="test",
            agents=agents,
            flow=flow,
        )

        assert len(segment.edges) == 1

    def test_build_requires_agents(self):
        """Test build_segment raises error with no agents."""
        with pytest.raises(ValueError, match="At least one agent"):
            build_segment(team_id="test", agents={}, flow="sequential")


class TestResultFromSegmentResult:
    """Tests for result_from_segment_result function."""

    def test_successful_result(self):
        """Test converting successful SegmentResult."""
        segment_result = MagicMock()
        segment_result.status = SegmentStatus.COMPLETED
        segment_result.output = {"output": "Hello"}
        segment_result.step_outputs = {"agent1": {"output": "Hello"}}
        segment_result.completed_steps = ["agent1"]
        segment_result.steps_completed = 1
        segment_result.duration_ms = 1500
        segment_result.error = None
        segment_result.started_at = None
        segment_result.completed_at = None

        result = result_from_segment_result(segment_result, {})

        assert result.success is True
        assert result.output == {"agent1": {"output": "Hello"}}
        assert result.final_output == "Hello"
        assert result.duration_ms == 1500

    def test_failed_result(self):
        """Test converting failed SegmentResult."""
        segment_result = MagicMock()
        segment_result.status = SegmentStatus.FAILED
        segment_result.output = {}
        segment_result.step_outputs = {}
        segment_result.completed_steps = []
        segment_result.steps_completed = 0
        segment_result.duration_ms = 500
        segment_result.error = Exception("Agent failed")
        segment_result.started_at = None
        segment_result.completed_at = None

        result = result_from_segment_result(segment_result, {})

        assert result.success is False
        assert result.error == "Agent failed"

    def test_agents_called_from_completed_steps(self):
        """Test agents_called extracted from completed_steps."""
        segment_result = MagicMock()
        segment_result.status = SegmentStatus.COMPLETED
        segment_result.output = {"output": "Final"}
        segment_result.step_outputs = {
            "writer": {"output": "result1"},
            "reviewer": {"output": "result2"},
            "publisher": {"output": "result3"},
        }
        segment_result.completed_steps = ["writer", "reviewer", "publisher"]
        segment_result.steps_completed = 3
        segment_result.duration_ms = 2000
        segment_result.error = None
        segment_result.started_at = None
        segment_result.completed_at = None

        result = result_from_segment_result(segment_result, {})

        assert "writer" in result.agents_called
        assert "reviewer" in result.agents_called
        assert "publisher" in result.agents_called
        assert result.final_output == "result3"
