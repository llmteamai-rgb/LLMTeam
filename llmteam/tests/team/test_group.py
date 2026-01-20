"""Tests for llmteam.team.group module (LLMGroup)."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from llmteam.team.group import LLMGroup
from llmteam.team.team import LLMTeam
from llmteam.team.result import RunResult, RunStatus


class TestLLMGroupCreation:
    """Tests for LLMGroup creation."""

    def test_create_group(self):
        """Test creating LLMGroup."""
        leader = LLMTeam(team_id="leader")
        team1 = LLMTeam(team_id="team1")
        team2 = LLMTeam(team_id="team2")

        group = LLMGroup(
            group_id="test_group",
            leader=leader,
            teams=[team1, team2],
        )

        assert group.group_id == "test_group"
        assert group.leader == leader
        assert len(group) == 3  # leader + team1 + team2

    def test_group_includes_leader(self):
        """Test group includes leader in teams."""
        leader = LLMTeam(team_id="leader")
        team1 = LLMTeam(team_id="team1")

        group = LLMGroup(
            group_id="test",
            leader=leader,
            teams=[team1],
        )

        teams = group.teams
        team_ids = [t.team_id for t in teams]
        assert "leader" in team_ids
        assert "team1" in team_ids

    def test_default_model(self):
        """Test default model for group orchestrator."""
        leader = LLMTeam(team_id="leader")
        group = LLMGroup(
            group_id="test",
            leader=leader,
            teams=[],
        )

        assert group._model == "gpt-4o-mini"

    def test_custom_model(self):
        """Test custom model for group orchestrator."""
        leader = LLMTeam(team_id="leader")
        group = LLMGroup(
            group_id="test",
            leader=leader,
            teams=[],
            model="gpt-4o",
        )

        assert group._model == "gpt-4o"

    def test_max_iterations(self):
        """Test max iterations setting."""
        leader = LLMTeam(team_id="leader")
        group = LLMGroup(
            group_id="test",
            leader=leader,
            teams=[],
            max_iterations=5,
        )

        assert group._max_iterations == 5


class TestLLMGroupTeamAccess:
    """Tests for team access methods."""

    def test_get_team(self):
        """Test getting team by id."""
        leader = LLMTeam(team_id="leader")
        team1 = LLMTeam(team_id="team1")

        group = LLMGroup(
            group_id="test",
            leader=leader,
            teams=[team1],
        )

        assert group.get_team("leader") == leader
        assert group.get_team("team1") == team1

    def test_get_nonexistent_team(self):
        """Test getting nonexistent team returns None."""
        leader = LLMTeam(team_id="leader")
        group = LLMGroup(
            group_id="test",
            leader=leader,
            teams=[],
        )

        assert group.get_team("nonexistent") is None

    def test_leader_property(self):
        """Test leader property."""
        leader = LLMTeam(team_id="leader")
        group = LLMGroup(
            group_id="test",
            leader=leader,
            teams=[],
        )

        assert group.leader == leader

    def test_teams_property(self):
        """Test teams property returns all teams."""
        leader = LLMTeam(team_id="leader")
        team1 = LLMTeam(team_id="team1")

        group = LLMGroup(
            group_id="test",
            leader=leader,
            teams=[team1],
        )

        teams = group.teams
        assert len(teams) == 2


class TestLLMGroupExecution:
    """Tests for group execution."""

    @pytest.fixture
    def mock_teams(self):
        """Create mock teams for testing."""
        leader = MagicMock(spec=LLMTeam)
        leader.team_id = "leader"
        leader.list_agents.return_value = []
        leader.run = AsyncMock(return_value=RunResult(
            success=True,
            output={"response": "Leader result"},
            agents_called=["leader_agent"],
            iterations=1,
        ))

        team1 = MagicMock(spec=LLMTeam)
        team1.team_id = "team1"
        team1.list_agents.return_value = []
        team1.run = AsyncMock(return_value=RunResult(
            success=True,
            output={"response": "Team1 result"},
            agents_called=["team1_agent"],
            iterations=1,
        ))

        return leader, team1

    @pytest.mark.asyncio
    async def test_run_returns_result(self, mock_teams):
        """Test run returns RunResult."""
        leader, team1 = mock_teams

        group = LLMGroup(
            group_id="test",
            leader=leader,
            teams=[team1],
        )

        # Mock the orchestrator team
        with patch.object(group, "_orchestrator_team") as mock_orch:
            mock_orch.run = AsyncMock(return_value=RunResult(
                success=True,
                output={"next_team": "leader", "should_continue": False},
            ))

            result = await group.run({"query": "test"})

            assert result is not None
            assert hasattr(result, "success")

    @pytest.mark.asyncio
    async def test_run_with_run_id(self, mock_teams):
        """Test run with custom run_id."""
        leader, team1 = mock_teams

        group = LLMGroup(
            group_id="test",
            leader=leader,
            teams=[team1],
        )

        with patch.object(group, "_orchestrator_team") as mock_orch:
            mock_orch.run = AsyncMock(return_value=RunResult(
                success=True,
                output={"next_team": "leader", "should_continue": False},
            ))

            await group.run({"query": "test"}, run_id="custom-run-id")

            # Orchestrator should be called with run_id containing custom prefix
            call_args = mock_orch.run.call_args
            assert "custom-run-id" in call_args.kwargs.get("run_id", "")


class TestLLMGroupDecisionParsing:
    """Tests for decision parsing."""

    def test_parse_dict_decision(self):
        """Test parsing dict decision."""
        leader = LLMTeam(team_id="leader")
        group = LLMGroup(
            group_id="test",
            leader=leader,
            teams=[],
        )

        decision = group._parse_decision({
            "next_team": "leader",
            "should_continue": True,
        })

        assert decision["next_team"] == "leader"
        assert decision["should_continue"] is True

    def test_parse_string_decision(self):
        """Test parsing JSON string decision."""
        leader = LLMTeam(team_id="leader")
        group = LLMGroup(
            group_id="test",
            leader=leader,
            teams=[],
        )

        decision = group._parse_decision(
            'Some text {"next_team": "team1", "should_continue": true} more text'
        )

        assert decision["next_team"] == "team1"
        assert decision["should_continue"] is True

    def test_parse_invalid_returns_default(self):
        """Test invalid decision returns default to leader."""
        leader = LLMTeam(team_id="leader")
        group = LLMGroup(
            group_id="test",
            leader=leader,
            teams=[],
        )

        decision = group._parse_decision("invalid output")

        assert decision["next_team"] == "leader"
        assert decision["should_continue"] is True


class TestLLMGroupSerialization:
    """Tests for serialization."""

    def test_to_config(self):
        """Test export to config dict."""
        leader = LLMTeam(team_id="leader")
        team1 = LLMTeam(team_id="team1")

        group = LLMGroup(
            group_id="test_group",
            leader=leader,
            teams=[team1],
            model="gpt-4o",
            max_iterations=5,
        )

        config = group.to_config()

        assert config["group_id"] == "test_group"
        assert config["leader"] == "leader"
        assert config["model"] == "gpt-4o"
        assert config["max_iterations"] == 5
        assert len(config["teams"]) == 2


class TestLLMGroupMagicMethods:
    """Tests for magic methods."""

    def test_repr(self):
        """Test __repr__."""
        leader = LLMTeam(team_id="leader")
        team1 = LLMTeam(team_id="team1")

        group = LLMGroup(
            group_id="test_group",
            leader=leader,
            teams=[team1],
        )

        repr_str = repr(group)
        assert "test_group" in repr_str
        assert "2" in repr_str  # teams count

    def test_len(self):
        """Test __len__."""
        leader = LLMTeam(team_id="leader")
        team1 = LLMTeam(team_id="team1")
        team2 = LLMTeam(team_id="team2")

        group = LLMGroup(
            group_id="test",
            leader=leader,
            teams=[team1, team2],
        )

        assert len(group) == 3  # leader + 2 teams
