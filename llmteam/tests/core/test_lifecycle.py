"""
Tests for RFC-014: Enhanced Configurator Mode (opt-in lifecycle).
"""

import pytest

from llmteam import (
    LLMTeam,
    TeamState,
    ProposalStatus,
    ConfigurationProposal,
    TeamLifecycle,
    LifecycleError,
)
from llmteam.agents.orchestrator import OrchestratorConfig, OrchestratorMode, RoutingDecision


class TestTeamState:
    """Tests for TeamState enum."""

    def test_states(self):
        """TeamState should have all lifecycle states."""
        assert TeamState.UNCONFIGURED == "unconfigured"
        assert TeamState.CONFIGURING == "configuring"
        assert TeamState.READY == "ready"
        assert TeamState.RUNNING == "running"
        assert TeamState.PAUSED == "paused"
        assert TeamState.COMPLETED == "completed"
        assert TeamState.FAILED == "failed"


class TestProposalStatus:
    """Tests for ProposalStatus enum."""

    def test_statuses(self):
        """ProposalStatus should have all statuses."""
        assert ProposalStatus.PENDING == "pending"
        assert ProposalStatus.APPROVED == "approved"
        assert ProposalStatus.REJECTED == "rejected"
        assert ProposalStatus.APPLIED == "applied"


class TestConfigurationProposal:
    """Tests for ConfigurationProposal."""

    def test_creation(self):
        """ConfigurationProposal should be created with defaults."""
        proposal = ConfigurationProposal(
            proposal_id="p-1",
            changes={"model": "gpt-4o"},
            reason="Better accuracy",
        )

        assert proposal.proposal_id == "p-1"
        assert proposal.changes == {"model": "gpt-4o"}
        assert proposal.reason == "Better accuracy"
        assert proposal.status == ProposalStatus.PENDING

    def test_approve(self):
        """approve should change status."""
        proposal = ConfigurationProposal(proposal_id="p-1")
        proposal.approve()

        assert proposal.status == ProposalStatus.APPROVED
        assert proposal.reviewed_at is not None

    def test_reject(self):
        """reject should change status and append reason."""
        proposal = ConfigurationProposal(proposal_id="p-1", reason="Original")
        proposal.reject("Too expensive")

        assert proposal.status == ProposalStatus.REJECTED
        assert "Too expensive" in proposal.reason

    def test_mark_applied(self):
        """mark_applied should change status."""
        proposal = ConfigurationProposal(proposal_id="p-1")
        proposal.approve()
        proposal.mark_applied()

        assert proposal.status == ProposalStatus.APPLIED

    def test_to_dict(self):
        """to_dict should serialize proposal."""
        proposal = ConfigurationProposal(
            proposal_id="p-1",
            changes={"key": "value"},
        )

        data = proposal.to_dict()

        assert data["proposal_id"] == "p-1"
        assert data["status"] == "pending"
        assert data["changes"] == {"key": "value"}


class TestTeamLifecycle:
    """Tests for TeamLifecycle."""

    def test_initial_state(self):
        """TeamLifecycle should start as UNCONFIGURED."""
        lc = TeamLifecycle()
        assert lc.state == TeamState.UNCONFIGURED

    def test_transition_unconfigured_to_configuring(self):
        """Should allow UNCONFIGURED → CONFIGURING."""
        lc = TeamLifecycle()
        lc.transition_to(TeamState.CONFIGURING)
        assert lc.state == TeamState.CONFIGURING

    def test_transition_configuring_to_ready(self):
        """Should allow CONFIGURING → READY."""
        lc = TeamLifecycle()
        lc.transition_to(TeamState.CONFIGURING)
        lc.transition_to(TeamState.READY)
        assert lc.state == TeamState.READY

    def test_transition_ready_to_running(self):
        """Should allow READY → RUNNING."""
        lc = TeamLifecycle()
        lc.transition_to(TeamState.CONFIGURING)
        lc.transition_to(TeamState.READY)
        lc.transition_to(TeamState.RUNNING)
        assert lc.state == TeamState.RUNNING

    def test_transition_running_to_completed(self):
        """Should allow RUNNING → COMPLETED."""
        lc = TeamLifecycle()
        lc.transition_to(TeamState.CONFIGURING)
        lc.transition_to(TeamState.READY)
        lc.transition_to(TeamState.RUNNING)
        lc.transition_to(TeamState.COMPLETED)
        assert lc.state == TeamState.COMPLETED

    def test_transition_running_to_failed(self):
        """Should allow RUNNING → FAILED."""
        lc = TeamLifecycle()
        lc.transition_to(TeamState.CONFIGURING)
        lc.transition_to(TeamState.READY)
        lc.transition_to(TeamState.RUNNING)
        lc.transition_to(TeamState.FAILED)
        assert lc.state == TeamState.FAILED

    def test_transition_completed_to_ready(self):
        """Should allow COMPLETED → READY (re-run)."""
        lc = TeamLifecycle()
        lc.transition_to(TeamState.CONFIGURING)
        lc.transition_to(TeamState.READY)
        lc.transition_to(TeamState.RUNNING)
        lc.transition_to(TeamState.COMPLETED)
        lc.transition_to(TeamState.READY)
        assert lc.state == TeamState.READY

    def test_invalid_transition_raises(self):
        """Invalid transition should raise LifecycleError."""
        lc = TeamLifecycle()

        with pytest.raises(LifecycleError):
            lc.transition_to(TeamState.RUNNING)  # Can't skip CONFIGURING → READY

    def test_can_transition_to(self):
        """can_transition_to should check without performing."""
        lc = TeamLifecycle()

        assert lc.can_transition_to(TeamState.CONFIGURING) is True
        assert lc.can_transition_to(TeamState.RUNNING) is False

    def test_ensure_ready_when_ready(self):
        """ensure_ready should pass when in READY state."""
        lc = TeamLifecycle()
        lc.transition_to(TeamState.CONFIGURING)
        lc.transition_to(TeamState.READY)

        lc.ensure_ready()  # Should not raise

    def test_ensure_ready_when_not_ready(self):
        """ensure_ready should raise when not READY."""
        lc = TeamLifecycle()

        with pytest.raises(LifecycleError, match="must be in READY"):
            lc.ensure_ready()

    def test_add_proposal(self):
        """add_proposal should add to proposals list."""
        lc = TeamLifecycle()
        proposal = ConfigurationProposal(proposal_id="p-1", changes={"x": 1})
        lc.add_proposal(proposal)

        assert len(lc.proposals) == 1
        assert lc.proposals[0].proposal_id == "p-1"

    def test_pending_proposals(self):
        """pending_proposals should filter by status."""
        lc = TeamLifecycle()
        p1 = ConfigurationProposal(proposal_id="p-1")
        p2 = ConfigurationProposal(proposal_id="p-2")
        p2.approve()
        lc.add_proposal(p1)
        lc.add_proposal(p2)

        assert len(lc.pending_proposals) == 1
        assert lc.pending_proposals[0].proposal_id == "p-1"

    def test_approve_all(self):
        """approve_all should approve all pending proposals."""
        lc = TeamLifecycle()
        lc.add_proposal(ConfigurationProposal(proposal_id="p-1"))
        lc.add_proposal(ConfigurationProposal(proposal_id="p-2"))

        count = lc.approve_all()

        assert count == 2
        assert len(lc.pending_proposals) == 0

    def test_history(self):
        """history should record transitions."""
        lc = TeamLifecycle()
        lc.transition_to(TeamState.CONFIGURING)
        lc.transition_to(TeamState.READY)

        assert len(lc.history) == 2
        assert lc.history[0]["from"] == "unconfigured"
        assert lc.history[0]["to"] == "configuring"

    def test_to_dict(self):
        """to_dict should serialize state."""
        lc = TeamLifecycle()
        lc.add_proposal(ConfigurationProposal(proposal_id="p-1"))

        data = lc.to_dict()

        assert data["state"] == "unconfigured"
        assert data["proposals_count"] == 1

    def test_paused_to_running(self):
        """Should allow PAUSED → RUNNING (resume)."""
        lc = TeamLifecycle()
        lc.transition_to(TeamState.CONFIGURING)
        lc.transition_to(TeamState.READY)
        lc.transition_to(TeamState.RUNNING)
        lc.transition_to(TeamState.PAUSED)
        lc.transition_to(TeamState.RUNNING)
        assert lc.state == TeamState.RUNNING


class TestLLMTeamLifecycleIntegration:
    """Tests for LLMTeam lifecycle enforcement."""

    def test_lifecycle_disabled_by_default(self):
        """LLMTeam should not enforce lifecycle by default."""
        team = LLMTeam(team_id="test")

        assert team.lifecycle is None
        assert team.state is None

    def test_lifecycle_enabled(self):
        """LLMTeam with enforce_lifecycle=True should start UNCONFIGURED."""
        team = LLMTeam(team_id="test", enforce_lifecycle=True)

        assert team.lifecycle is not None
        assert team.state == "unconfigured"

    def test_mark_configuring(self):
        """mark_configuring should transition to CONFIGURING."""
        team = LLMTeam(team_id="test", enforce_lifecycle=True)
        team.mark_configuring()

        assert team.state == "configuring"

    def test_mark_ready(self):
        """mark_ready should transition to READY."""
        team = LLMTeam(team_id="test", enforce_lifecycle=True)
        team.mark_configuring()
        team.mark_ready()

        assert team.state == "ready"

    def test_mark_configuring_without_lifecycle_raises(self):
        """mark_configuring without lifecycle should raise."""
        team = LLMTeam(team_id="test")

        with pytest.raises(RuntimeError, match="not enforced"):
            team.mark_configuring()

    def test_mark_ready_without_lifecycle_raises(self):
        """mark_ready without lifecycle should raise."""
        team = LLMTeam(team_id="test")

        with pytest.raises(RuntimeError, match="not enforced"):
            team.mark_ready()

    async def test_run_without_ready_fails(self):
        """run() should fail when lifecycle enforced but not READY."""
        team = LLMTeam(team_id="test", orchestration=True, enforce_lifecycle=True)
        team.add_agent({"type": "llm", "role": "agent1", "prompt": "test"})

        result = await team.run({"query": "test"})

        assert result.success is False
        assert "READY" in result.error

    async def test_run_when_ready_succeeds(self):
        """run() should succeed when lifecycle is READY."""
        team = LLMTeam(team_id="test", orchestration=True, enforce_lifecycle=True)
        team.add_agent({"type": "llm", "role": "agent1", "prompt": "test"})

        # Mock orchestrator
        async def mock_decide(current_state, available_agents):
            return RoutingDecision(
                next_agent=available_agents[0] if available_agents else "",
                reason="test",
            )

        team._orchestrator.decide_next_agent = mock_decide

        # Mock agent
        agent = team.get_agent("agent1")

        async def mock_execute(input_data, context, run_id=None):
            from llmteam import AgentResult
            return AgentResult(output="result")

        agent.execute = mock_execute

        # Transition to READY
        team.mark_configuring()
        team.mark_ready()

        result = await team.run({"query": "test"})

        assert result.success is True
        assert team.state == "completed"

    async def test_run_failure_transitions_to_failed(self):
        """run() failure should transition to FAILED state."""
        team = LLMTeam(team_id="test", orchestration=True, enforce_lifecycle=True)
        team.add_agent({"type": "llm", "role": "agent1", "prompt": "test"})

        # Mock orchestrator to raise
        async def mock_decide(current_state, available_agents):
            raise RuntimeError("crash")

        team._orchestrator.decide_next_agent = mock_decide

        # Transition to READY
        team.mark_configuring()
        team.mark_ready()

        result = await team.run({"query": "test"})

        assert result.success is False
        assert team.state == "failed"

    async def test_run_without_lifecycle_works_normally(self):
        """run() without lifecycle should work as before."""
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

        result = await team.run({"query": "test"})

        assert result.success is True
        assert team.state is None  # No lifecycle

    async def test_re_run_after_completed(self):
        """Team should be re-runnable after COMPLETED → READY."""
        team = LLMTeam(team_id="test", orchestration=True, enforce_lifecycle=True)
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

        # First run
        team.mark_configuring()
        team.mark_ready()
        result1 = await team.run({"query": "test"})
        assert result1.success is True
        assert team.state == "completed"

        # Re-configure and re-run
        team.mark_ready()  # COMPLETED → READY is valid
        result2 = await team.run({"query": "test2"})
        assert result2.success is True
