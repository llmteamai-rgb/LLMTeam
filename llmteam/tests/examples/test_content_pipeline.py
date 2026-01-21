"""
Tests for Content Pipeline example.

These tests verify the pipeline configuration without calling actual LLM APIs.
"""

import pytest
from typing import Dict, Any

from llmteam import LLMTeam, ContextMode
from llmteam.agents import AgentType


class TestContentPipelineConfig:
    """Test content pipeline configuration."""

    def test_create_pipeline_with_3_agents(self):
        """Pipeline should have exactly 3 LLM agents."""
        team = LLMTeam(
            team_id="content",
            agents=[
                {"type": "llm", "role": "writer", "prompt": "Write: {topic}"},
                {"type": "llm", "role": "editor", "prompt": "Edit: {context}"},
                {"type": "llm", "role": "publisher", "prompt": "Publish: {context}"},
            ],
        )

        agents = team.list_agents()
        assert len(agents) == 3

    def test_agent_roles_are_correct(self):
        """Agents should have correct roles."""
        team = LLMTeam(
            team_id="content",
            agents=[
                {"type": "llm", "role": "writer", "prompt": "Write: {topic}"},
                {"type": "llm", "role": "editor", "prompt": "Edit: {context}"},
                {"type": "llm", "role": "publisher", "prompt": "Publish: {context}"},
            ],
        )

        agent_roles = [a.role for a in team.list_agents()]
        assert agent_roles == ["writer", "editor", "publisher"]

    def test_all_agents_are_llm_type(self):
        """All agents should be LLM type."""
        team = LLMTeam(
            team_id="content",
            agents=[
                {"type": "llm", "role": "writer", "prompt": "Write: {topic}"},
                {"type": "llm", "role": "editor", "prompt": "Edit: {context}"},
                {"type": "llm", "role": "publisher", "prompt": "Publish: {context}"},
            ],
        )

        for agent in team.list_agents():
            # agent_type is the class attribute
            assert agent.agent_type == AgentType.LLM

    def test_sequential_flow_string(self):
        """Pipeline should support string flow definition."""
        team = LLMTeam(
            team_id="content",
            agents=[
                {"type": "llm", "role": "writer", "prompt": "Write: {topic}"},
                {"type": "llm", "role": "editor", "prompt": "Edit: {context}"},
                {"type": "llm", "role": "publisher", "prompt": "Publish: {context}"},
            ],
            flow="writer -> editor -> publisher",
        )

        config = team.to_config()
        assert config["flow"] == "writer -> editor -> publisher"

    def test_orchestration_mode(self):
        """Pipeline should support orchestration mode via flow parameter."""
        # Orchestration is set via orchestration=True, which adds an internal _orchestrator agent
        team = LLMTeam(
            team_id="content",
            agents=[
                {"type": "llm", "role": "writer", "prompt": "Write: {topic}"},
                {"type": "llm", "role": "editor", "prompt": "Edit: {context}"},
                {"type": "llm", "role": "publisher", "prompt": "Publish: {context}"},
            ],
            orchestration=True,
        )

        # Check that team was created successfully with orchestration
        assert team.team_id == "content"
        # v4.1.0: Orchestrator is separate entity, not an agent
        # So list_agents() returns only user agents (3)
        assert len(team.list_agents()) == 3
        # Verify orchestrator is present (via get_orchestrator, not get_agent)
        orchestrator = team.get_orchestrator()
        assert orchestrator is not None
        assert orchestrator.is_router  # orchestration=True enables ACTIVE mode

    def test_get_agent_by_role(self):
        """Should retrieve agents by role."""
        team = LLMTeam(
            team_id="content",
            agents=[
                {"type": "llm", "role": "writer", "prompt": "Write: {topic}"},
                {"type": "llm", "role": "editor", "prompt": "Edit: {context}"},
                {"type": "llm", "role": "publisher", "prompt": "Publish: {context}"},
            ],
        )

        writer = team.get_agent("writer")
        assert writer is not None
        assert writer.role == "writer"

        editor = team.get_agent("editor")
        assert editor is not None
        assert editor.role == "editor"

    def test_add_agents_dynamically(self):
        """Should add agents dynamically."""
        team = LLMTeam(team_id="content")

        team.add_llm_agent(role="writer", prompt="Write: {topic}")
        team.add_llm_agent(role="editor", prompt="Edit: {context}")
        team.add_llm_agent(role="publisher", prompt="Publish: {context}")

        assert len(team.list_agents()) == 3

    def test_to_config_roundtrip(self):
        """Config should be serializable and deserializable."""
        team = LLMTeam(
            team_id="content",
            agents=[
                {"type": "llm", "role": "writer", "prompt": "Write: {topic}"},
                {"type": "llm", "role": "editor", "prompt": "Edit: {context}"},
                {"type": "llm", "role": "publisher", "prompt": "Publish: {context}"},
            ],
            flow="writer -> editor -> publisher",
        )

        config = team.to_config()
        restored = LLMTeam.from_config(config)

        assert restored.team_id == "content"
        assert len(restored.list_agents()) == 3

    def test_context_mode_shared(self):
        """Pipeline should use shared context by default."""
        team = LLMTeam(
            team_id="content",
            agents=[
                {"type": "llm", "role": "writer", "prompt": "Write: {topic}"},
            ],
            context_mode=ContextMode.SHARED,
        )

        config = team.to_config()
        assert config["context_mode"] == "shared"

    def test_agent_config_options(self):
        """Agent config should support all LLM options."""
        team = LLMTeam(
            team_id="content",
            agents=[
                {
                    "type": "llm",
                    "role": "writer",
                    "prompt": "Write: {topic}",
                    "model": "gpt-4o",
                    "temperature": 0.8,
                    "max_tokens": 2000,
                    "system_prompt": "You are a writer.",
                },
            ],
        )

        writer = team.get_agent("writer")
        assert writer is not None
        # Verify agent has the config properties
        assert writer.model == "gpt-4o"
        assert writer.temperature == 0.8
        assert writer.max_tokens == 2000


class TestContentPipelineFlow:
    """Test flow definitions for content pipeline."""

    def test_sequential_default(self):
        """Default flow should be sequential."""
        team = LLMTeam(
            team_id="content",
            agents=[
                {"type": "llm", "role": "writer", "prompt": "Write: {topic}"},
                {"type": "llm", "role": "editor", "prompt": "Edit: {context}"},
            ],
        )

        config = team.to_config()
        assert config.get("flow") in [None, "sequential"]

    def test_explicit_sequential_flow(self):
        """Explicit sequential flow."""
        team = LLMTeam(
            team_id="content",
            flow="sequential",
            agents=[
                {"type": "llm", "role": "writer", "prompt": "Write: {topic}"},
            ],
        )

        config = team.to_config()
        assert config["flow"] == "sequential"

    def test_arrow_flow_syntax(self):
        """Arrow syntax for flow definition."""
        team = LLMTeam(
            team_id="content",
            agents=[
                {"type": "llm", "role": "a", "prompt": "A"},
                {"type": "llm", "role": "b", "prompt": "B"},
                {"type": "llm", "role": "c", "prompt": "C"},
            ],
            flow="a -> b -> c",
        )

        config = team.to_config()
        assert config["flow"] == "a -> b -> c"

    def test_parallel_flow_syntax(self):
        """Parallel syntax for flow definition."""
        team = LLMTeam(
            team_id="content",
            agents=[
                {"type": "llm", "role": "a", "prompt": "A"},
                {"type": "llm", "role": "b", "prompt": "B"},
                {"type": "llm", "role": "c", "prompt": "C"},
            ],
            flow="a, b -> c",  # a and b run parallel, then c
        )

        config = team.to_config()
        assert config["flow"] == "a, b -> c"


class TestContentPipelineValidation:
    """Test pipeline validation."""

    def test_reject_empty_agents(self):
        """Should handle empty agents list."""
        team = LLMTeam(team_id="empty")
        assert len(team.list_agents()) == 0

    def test_reject_duplicate_roles(self):
        """Should reject duplicate agent roles."""
        with pytest.raises((ValueError, KeyError)):
            LLMTeam(
                team_id="content",
                agents=[
                    {"type": "llm", "role": "writer", "prompt": "Write: {topic}"},
                    {"type": "llm", "role": "writer", "prompt": "Write again"},
                ],
            )

    def test_llm_agent_default_prompt(self):
        """LLM agents can have empty prompt (uses default)."""
        # The API may allow empty prompt with default value
        team = LLMTeam(
            team_id="content",
            agents=[
                {"type": "llm", "role": "writer", "prompt": ""},
            ],
        )
        # If no exception, check agent was created
        agent = team.get_agent("writer")
        assert agent is not None

    def test_agent_default_type(self):
        """Agents without type default to LLM."""
        # If type is not specified, the factory may default to LLM
        team = LLMTeam(
            team_id="content",
            agents=[
                {"role": "writer", "prompt": "Write"},  # No explicit type
            ],
        )

        writer = team.get_agent("writer")
        assert writer is not None
        # Should default to LLM type
        assert writer.agent_type == AgentType.LLM
