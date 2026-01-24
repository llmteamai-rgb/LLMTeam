"""
Tests for RFC-009: Group Architecture Unification.
"""

import pytest
from llmteam import LLMTeam
from llmteam.orchestration import (
    GroupOrchestrator,
    GroupRole,
    TeamRole,
    GroupContext,
    EscalationRequest,
    EscalationResponse,
    GroupEscalationAction,
    EscalationAction,  # Alias for GroupEscalationAction
    TeamReport,
    GroupReport,
    GroupResult,
)


class TestTeamRole:
    """Tests for TeamRole enum."""

    def test_team_roles(self):
        """TeamRole should have all expected values."""
        assert TeamRole.LEADER.value == "leader"
        assert TeamRole.MEMBER.value == "member"
        assert TeamRole.SPECIALIST.value == "specialist"
        assert TeamRole.FALLBACK.value == "fallback"


class TestGroupEscalationAction:
    """Tests for GroupEscalationAction enum."""

    def test_group_escalation_actions(self):
        """GroupEscalationAction should have all expected values."""
        assert GroupEscalationAction.RETRY.value == "retry"
        assert GroupEscalationAction.SKIP.value == "skip"
        assert GroupEscalationAction.REROUTE.value == "reroute"
        assert GroupEscalationAction.ABORT.value == "abort"
        assert GroupEscalationAction.CONTINUE.value == "continue"
        assert GroupEscalationAction.HUMAN.value == "human"

    def test_escalation_action_alias(self):
        """EscalationAction should be alias for GroupEscalationAction."""
        assert EscalationAction is GroupEscalationAction


class TestGroupRole:
    """Tests for GroupRole enum (RFC-009 extensions)."""

    def test_passive_roles(self):
        """GroupRole should have passive roles."""
        assert GroupRole.REPORT_COLLECTOR.value == "report_collector"
        assert GroupRole.COORDINATOR.value == "coordinator"

    def test_active_roles(self):
        """GroupRole should have active roles."""
        assert GroupRole.ROUTER.value == "router"
        assert GroupRole.AGGREGATOR.value == "aggregator"
        assert GroupRole.ARBITER.value == "arbiter"


class TestGroupContext:
    """Tests for GroupContext dataclass."""

    def test_create_context(self):
        """GroupContext should be created with required fields."""
        orch = GroupOrchestrator(group_id="test_group")

        context = GroupContext(
            group_id="test_group",
            group_orchestrator=orch,
            team_role=TeamRole.MEMBER,
        )

        assert context.group_id == "test_group"
        assert context.team_role == TeamRole.MEMBER
        assert context.can_escalate is True
        assert context.can_request_team is False

    def test_context_defaults(self):
        """GroupContext should have sensible defaults."""
        orch = GroupOrchestrator(group_id="test")

        context = GroupContext(
            group_id="test",
            group_orchestrator=orch,
            team_role=TeamRole.LEADER,
        )

        assert context.other_teams == []
        assert context.leader_team is None
        assert context.visible_teams == set()
        assert context.shared_context == {}


class TestEscalationModels:
    """Tests for escalation request/response models."""

    def test_escalation_request(self):
        """EscalationRequest should capture escalation info."""
        request = EscalationRequest(
            source_team_id="team1",
            reason="Test escalation",
            source_agent_id="agent1",
        )

        assert request.source_team_id == "team1"
        assert request.reason == "Test escalation"
        assert request.source_agent_id == "agent1"
        assert request.error is None
        assert request.context == {}

    def test_escalation_response(self):
        """EscalationResponse should provide action."""
        response = EscalationResponse(
            action=EscalationAction.RETRY,
            reason="Auto-retry on error",
        )

        assert response.action == EscalationAction.RETRY
        assert response.reason == "Auto-retry on error"
        assert response.route_to_team is None


class TestGroupOrchestratorInit:
    """Tests for GroupOrchestrator initialization."""

    def test_default_init(self):
        """GroupOrchestrator should initialize with defaults."""
        orch = GroupOrchestrator()

        assert orch.group_id.startswith("group_")
        assert orch.role == GroupRole.REPORT_COLLECTOR
        assert orch.teams_count == 0

    def test_custom_role(self):
        """GroupOrchestrator should accept custom role."""
        orch = GroupOrchestrator(
            group_id="my_group",
            role=GroupRole.COORDINATOR,
        )

        assert orch.group_id == "my_group"
        assert orch.role == GroupRole.COORDINATOR


class TestGroupOrchestratorTeamManagement:
    """Tests for team management with roles."""

    def test_add_team_with_role(self):
        """Add team with specific role."""
        orch = GroupOrchestrator(group_id="test")
        team = LLMTeam(team_id="team1")

        orch.add_team(team, role=TeamRole.LEADER)

        assert orch.teams_count == 1
        assert orch.get_team_role("team1") == TeamRole.LEADER
        assert orch.leader is team

    def test_add_team_default_member(self):
        """Add team defaults to MEMBER role."""
        orch = GroupOrchestrator(group_id="test")
        team = LLMTeam(team_id="team1")

        orch.add_team(team)

        assert orch.get_team_role("team1") == TeamRole.MEMBER

    def test_only_one_leader(self):
        """Only one LEADER allowed per group."""
        orch = GroupOrchestrator(group_id="test")
        team1 = LLMTeam(team_id="team1")
        team2 = LLMTeam(team_id="team2")

        orch.add_team(team1, role=TeamRole.LEADER)

        with pytest.raises(ValueError, match="already has leader"):
            orch.add_team(team2, role=TeamRole.LEADER)

    def test_remove_team(self):
        """Remove team from group."""
        orch = GroupOrchestrator(group_id="test")
        team = LLMTeam(team_id="team1")

        orch.add_team(team, role=TeamRole.LEADER)
        assert orch.teams_count == 1

        result = orch.remove_team("team1")

        assert result is True
        assert orch.teams_count == 0
        assert orch.leader is None


class TestLLMTeamGroupIntegration:
    """Tests for LLMTeam group integration (RFC-009)."""

    def test_team_not_in_group(self):
        """Team not in group by default."""
        team = LLMTeam(team_id="test")

        assert team.is_in_group is False
        assert team.group_id is None
        assert team.group_role is None

    def test_team_joins_group(self):
        """Team receives context when added to group."""
        orch = GroupOrchestrator(group_id="my_group")
        team = LLMTeam(team_id="team1")

        orch.add_team(team, role=TeamRole.LEADER)

        assert team.is_in_group is True
        assert team.group_id == "my_group"
        assert team.group_role == "leader"

    def test_team_leaves_group(self):
        """Team context cleared when removed from group."""
        orch = GroupOrchestrator(group_id="my_group")
        team = LLMTeam(team_id="team1")

        orch.add_team(team, role=TeamRole.MEMBER)
        assert team.is_in_group is True

        orch.remove_team("team1")

        assert team.is_in_group is False
        assert team.group_id is None

    async def test_escalate_without_group(self):
        """Escalation fails if team not in group."""
        team = LLMTeam(team_id="test")

        with pytest.raises(RuntimeError, match="not in a group"):
            await team.escalate_to_group(reason="Test")

    async def test_request_team_without_group(self):
        """Request team fails if not in group."""
        team = LLMTeam(team_id="test")

        with pytest.raises(RuntimeError, match="not in a group"):
            await team.request_team("other", {"task": "test"})


class TestGroupOrchestratorExecution:
    """Tests for GroupOrchestrator execution with different roles."""

    def test_execution_order_leader_first(self):
        """Coordinator executes leader first."""
        orch = GroupOrchestrator(
            group_id="test",
            role=GroupRole.COORDINATOR,
        )

        team1 = LLMTeam(team_id="team1")
        team2 = LLMTeam(team_id="team2")

        orch.add_team(team1, role=TeamRole.LEADER)
        orch.add_team(team2, role=TeamRole.MEMBER)

        order = orch._get_execution_order()
        assert order[0] == "team1"
        assert "team2" in order


class TestReportModels:
    """Tests for RFC-009 report model extensions."""

    def test_team_report_with_role(self):
        """TeamReport should include team_role field."""
        report = TeamReport(
            team_id="team1",
            run_id="run1",
            success=True,
            duration_ms=100,
            agents_executed=["agent1"],
            team_role="leader",
            escalations_sent=1,
        )

        assert report.team_role == "leader"
        assert report.escalations_sent == 1

        data = report.to_dict()
        assert data["team_role"] == "leader"
        assert data["escalations_sent"] == 1

    def test_group_report_with_run_id(self):
        """GroupReport should include run_id and escalations."""
        report = GroupReport(
            group_id="group1",
            role="coordinator",
            teams_count=2,
            teams_succeeded=2,
            teams_failed=0,
            total_duration_ms=200,
            run_id="run1",
            escalations_handled=3,
        )

        assert report.run_id == "run1"
        assert report.escalations_handled == 3

        data = report.to_dict()
        assert data["run_id"] == "run1"
        assert data["escalations_handled"] == 3

    def test_group_result_with_duration(self):
        """GroupResult should include run_id and duration."""
        result = GroupResult(
            success=True,
            output={"result": "ok"},
            run_id="run1",
            duration_ms=150,
        )

        assert result.run_id == "run1"
        assert result.duration_ms == 150

        data = result.to_dict()
        assert data["run_id"] == "run1"
        assert data["duration_ms"] == 150
