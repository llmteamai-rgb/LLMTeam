"""Tests for llmteam.team.team module (LLMTeam)."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from llmteam.team.team import LLMTeam
from llmteam.team.result import RunStatus, ContextMode
from llmteam.agents.types import AgentType


class TestLLMTeamCreation:
    """Tests for LLMTeam creation."""

    def test_create_empty_team(self):
        """Test creating empty team."""
        team = LLMTeam(team_id="test_team")

        assert team.team_id == "test_team"
        assert len(team) == 0

    def test_create_team_with_agents(self):
        """Test creating team with agents list."""
        team = LLMTeam(
            team_id="test_team",
            agents=[
                {"type": "llm", "role": "writer", "prompt": "Write: {input}"},
                {"type": "rag", "role": "retriever", "collection": "docs"},
            ],
        )

        assert len(team) == 2
        assert team.get_agent("writer") is not None
        assert team.get_agent("retriever") is not None

    def test_create_team_with_flow(self):
        """Test creating team with flow."""
        team = LLMTeam(
            team_id="test_team",
            agents=[
                {"type": "llm", "role": "a", "prompt": "A"},
                {"type": "llm", "role": "b", "prompt": "B"},
            ],
            flow="a -> b",
        )

        assert team._flow == "a -> b"

    def test_create_team_with_context_mode(self):
        """Test creating team with context mode."""
        team = LLMTeam(
            team_id="test_team",
            context_mode=ContextMode.NOT_SHARED,
        )

        assert team._context_mode == ContextMode.NOT_SHARED

    def test_default_model(self):
        """Test default model setting."""
        team = LLMTeam(team_id="test_team")
        assert team._model == "gpt-4o-mini"

    def test_custom_model(self):
        """Test custom model setting."""
        team = LLMTeam(team_id="test_team", model="gpt-4o")
        assert team._model == "gpt-4o"


class TestLLMTeamAgentManagement:
    """Tests for agent management."""

    def test_add_agent_dict(self):
        """Test adding agent from dict."""
        team = LLMTeam(team_id="test")
        agent = team.add_agent({
            "type": "llm",
            "role": "writer",
            "prompt": "Write: {input}",
        })

        assert agent.role == "writer"
        assert "writer" in team

    def test_add_llm_agent(self):
        """Test add_llm_agent shortcut."""
        team = LLMTeam(team_id="test")
        agent = team.add_llm_agent(
            role="analyzer",
            prompt="Analyze: {text}",
            model="gpt-4o",
        )

        assert agent.role == "analyzer"
        assert agent.agent_type == AgentType.LLM

    def test_add_rag_agent(self):
        """Test add_rag_agent shortcut."""
        team = LLMTeam(team_id="test")
        agent = team.add_rag_agent(
            role="searcher",
            collection="documents",
        )

        assert agent.role == "searcher"
        assert agent.agent_type == AgentType.RAG

    def test_add_kag_agent(self):
        """Test add_kag_agent shortcut."""
        team = LLMTeam(team_id="test")
        agent = team.add_kag_agent(
            role="knowledge",
            max_hops=3,
        )

        assert agent.role == "knowledge"
        assert agent.agent_type == AgentType.KAG

    def test_add_duplicate_raises_error(self):
        """Test adding duplicate agent raises error."""
        team = LLMTeam(team_id="test")
        team.add_agent({"type": "llm", "role": "writer", "prompt": "test"})

        with pytest.raises(ValueError, match="already exists"):
            team.add_agent({"type": "llm", "role": "writer", "prompt": "test"})

    def test_get_agent(self):
        """Test getting agent by id."""
        team = LLMTeam(team_id="test")
        team.add_agent({"type": "llm", "role": "writer", "prompt": "test"})

        agent = team.get_agent("writer")
        assert agent is not None
        assert agent.role == "writer"

    def test_get_nonexistent_agent(self):
        """Test getting nonexistent agent returns None."""
        team = LLMTeam(team_id="test")
        agent = team.get_agent("nonexistent")
        assert agent is None

    def test_list_agents(self):
        """Test listing all agents."""
        team = LLMTeam(team_id="test")
        team.add_agent({"type": "llm", "role": "a", "prompt": "A"})
        team.add_agent({"type": "llm", "role": "b", "prompt": "B"})

        agents = team.list_agents()
        assert len(agents) == 2

    def test_contains(self):
        """Test __contains__ operator."""
        team = LLMTeam(team_id="test")
        team.add_agent({"type": "llm", "role": "writer", "prompt": "test"})

        assert "writer" in team
        assert "reader" not in team


class TestLLMTeamExecution:
    """Tests for team execution."""

    @pytest.mark.asyncio
    async def test_run_empty_team_fails(self):
        """Test running empty team returns error."""
        team = LLMTeam(team_id="test")
        result = await team.run({"input": "test"})

        assert result.success is False
        assert "No agents" in result.error

    @pytest.mark.asyncio
    async def test_run_returns_result(self):
        """Test run returns RunResult."""
        team = LLMTeam(
            team_id="test",
            agents=[{"type": "llm", "role": "writer", "prompt": "test"}],
        )

        # Mock the runner
        with patch.object(team, "_get_runner") as mock_get_runner:
            mock_runner = MagicMock()
            mock_runner.run = AsyncMock(return_value=MagicMock(
                status=MagicMock(value="completed"),
                output={"response": "Hello"},
                steps_executed=["writer"],
                duration_ms=100,
                error=None,
            ))
            mock_get_runner.return_value = mock_runner

            result = await team.run({"input": "test"})

            assert result is not None
            # Result type should be RunResult
            assert hasattr(result, "success")

    @pytest.mark.asyncio
    async def test_run_with_run_id(self):
        """Test run with custom run_id."""
        team = LLMTeam(
            team_id="test",
            agents=[{"type": "llm", "role": "writer", "prompt": "test"}],
        )

        with patch.object(team, "_get_runner") as mock_get_runner:
            mock_runner = MagicMock()
            mock_runner.run = AsyncMock(return_value=MagicMock(
                status=MagicMock(value="completed"),
                output={},
                steps_executed=[],
                duration_ms=0,
                error=None,
            ))
            mock_get_runner.return_value = mock_runner

            await team.run({"input": "test"}, run_id="custom-run-123")

            # Verify run was called
            mock_runner.run.assert_called_once()


class TestLLMTeamOrchestration:
    """Tests for orchestration features."""

    def test_add_orchestrator(self):
        """Test orchestration=True enables ROUTER mode."""
        team = LLMTeam(
            team_id="test",
            agents=[
                {"type": "llm", "role": "writer", "prompt": "Write"},
            ],
            orchestration=True,
        )

        # v4.1: orchestrator is now separate, check is_router_mode
        assert team.get_orchestrator() is not None
        assert team.is_router_mode is True
        # Should have only 1 working agent (orchestrator is separate)
        assert len(team) == 1

    def test_adaptive_flow_adds_orchestrator(self):
        """Test flow='adaptive' enables ROUTER mode."""
        team = LLMTeam(
            team_id="test",
            agents=[
                {"type": "llm", "role": "writer", "prompt": "Write"},
            ],
            flow="adaptive",
        )

        # v4.1: adaptive flow enables ROUTER mode
        assert team.get_orchestrator() is not None
        assert team.is_router_mode is True

    def test_default_passive_mode(self):
        """Test default mode is PASSIVE (no routing)."""
        team = LLMTeam(
            team_id="test",
            agents=[
                {"type": "llm", "role": "writer", "prompt": "Write"},
            ],
        )

        # Default: PASSIVE mode (SUPERVISOR + REPORTER, no ROUTER)
        assert team.get_orchestrator() is not None
        assert team.is_router_mode is False


class TestLLMTeamEvents:
    """Tests for event handling."""

    def test_register_event_callback(self):
        """Test registering event callback."""
        team = LLMTeam(team_id="test")
        callback = MagicMock()

        team.on("agent_complete", callback)

        assert "agent_complete" in team._event_callbacks
        assert callback in team._event_callbacks["agent_complete"]

    def test_unregister_event_callback(self):
        """Test unregistering event callback."""
        team = LLMTeam(team_id="test")
        callback = MagicMock()

        team.on("agent_complete", callback)
        team.off("agent_complete", callback)

        assert callback not in team._event_callbacks.get("agent_complete", [])


class TestLLMTeamSerialization:
    """Tests for serialization."""

    def test_to_config(self):
        """Test export to config dict."""
        team = LLMTeam(
            team_id="test",
            agents=[
                {"type": "llm", "role": "writer", "prompt": "Write"},
            ],
            flow="sequential",
            model="gpt-4o",
        )

        config = team.to_config()

        assert config["team_id"] == "test"
        assert config["flow"] == "sequential"
        assert config["model"] == "gpt-4o"
        assert len(config["agents"]) == 1

    def test_from_config(self):
        """Test create from config dict."""
        config = {
            "team_id": "restored",
            "agents": [
                {"type": "llm", "role": "writer", "prompt": "Write"},
            ],
            "flow": "sequential",
            "model": "gpt-4o",
        }

        team = LLMTeam.from_config(config)

        assert team.team_id == "restored"
        assert team._model == "gpt-4o"
        assert len(team) == 1


class TestLLMTeamGroup:
    """Tests for group creation."""

    def test_create_group(self):
        """Test creating group with this team as leader."""
        team1 = LLMTeam(team_id="team1")
        team2 = LLMTeam(team_id="team2")

        group = team1.create_group(
            group_id="test_group",
            teams=[team2],
        )

        assert group.group_id == "test_group"
        assert group.leader == team1
        assert len(group) == 2  # team1 + team2


class TestLLMTeamMagicMethods:
    """Tests for magic methods."""

    def test_repr(self):
        """Test __repr__."""
        team = LLMTeam(team_id="test")
        team.add_agent({"type": "llm", "role": "writer", "prompt": "test"})

        repr_str = repr(team)
        assert "test" in repr_str
        assert "1" in repr_str  # agents count

    def test_len(self):
        """Test __len__."""
        team = LLMTeam(team_id="test")
        assert len(team) == 0

        team.add_agent({"type": "llm", "role": "a", "prompt": "A"})
        assert len(team) == 1

        team.add_agent({"type": "llm", "role": "b", "prompt": "B"})
        assert len(team) == 2
