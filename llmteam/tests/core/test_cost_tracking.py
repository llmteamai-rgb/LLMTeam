"""
Tests for RFC-010: Cost Tracking & Budget Management.
"""

import pytest
from datetime import datetime

from llmteam import (
    LLMTeam,
    TokenUsage,
    RunCost,
    CostTracker,
    ModelPricing,
    PricingRegistry,
    Budget,
    BudgetPeriod,
    BudgetStatus,
    BudgetManager,
    BudgetExceededError,
)


class TestModelPricing:
    """Tests for ModelPricing."""

    def test_calculate_cost(self):
        """ModelPricing should calculate cost from tokens."""
        pricing = ModelPricing(input_per_1k=0.01, output_per_1k=0.03)

        cost = pricing.calculate(input_tokens=1000, output_tokens=500)

        # (1000/1000)*0.01 + (500/1000)*0.03 = 0.01 + 0.015 = 0.025
        assert abs(cost - 0.025) < 1e-6

    def test_zero_tokens(self):
        """Zero tokens should produce zero cost."""
        pricing = ModelPricing(input_per_1k=0.01, output_per_1k=0.03)

        cost = pricing.calculate(input_tokens=0, output_tokens=0)

        assert cost == 0.0


class TestPricingRegistry:
    """Tests for PricingRegistry."""

    def test_default_models(self):
        """Registry should have default model pricing."""
        registry = PricingRegistry()

        assert registry.get_pricing("gpt-4o") is not None
        assert registry.get_pricing("gpt-4o-mini") is not None
        assert registry.get_pricing("claude-3-5-sonnet-20241022") is not None

    def test_calculate_cost(self):
        """calculate_cost should return USD amount."""
        registry = PricingRegistry()

        cost = registry.calculate_cost("gpt-4o-mini", 1000, 500)

        # (1000/1000)*0.00015 + (500/1000)*0.0006 = 0.00015 + 0.0003 = 0.00045
        assert abs(cost - 0.00045) < 1e-6

    def test_unknown_model_returns_zero(self):
        """Unknown model should return 0 cost."""
        registry = PricingRegistry()

        cost = registry.calculate_cost("unknown-model", 1000, 500)

        assert cost == 0.0

    def test_prefix_match(self):
        """Registry should match model prefixes."""
        registry = PricingRegistry()

        # "gpt-4o-2024-08-06" should match "gpt-4o"
        pricing = registry.get_pricing("gpt-4o-2024-08-06")

        assert pricing is not None
        assert pricing.input_per_1k == 0.0025

    def test_register_custom_model(self):
        """Users should be able to register custom pricing."""
        registry = PricingRegistry()
        registry.register("my-model", ModelPricing(input_per_1k=0.005, output_per_1k=0.01))

        cost = registry.calculate_cost("my-model", 1000, 1000)

        assert abs(cost - 0.015) < 1e-6

    def test_list_models(self):
        """list_models should return all registered models."""
        registry = PricingRegistry()

        models = registry.list_models()

        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models
        assert len(models) >= 10


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_creation(self):
        """TokenUsage should be created with required fields."""
        usage = TokenUsage(
            input_tokens=500,
            output_tokens=200,
            model="gpt-4o",
            cost_usd=0.0045,
        )

        assert usage.input_tokens == 500
        assert usage.output_tokens == 200
        assert usage.total_tokens == 700
        assert usage.model == "gpt-4o"
        assert usage.cost_usd == 0.0045
        assert usage.agent_id is None

    def test_to_dict(self):
        """TokenUsage.to_dict should serialize all fields."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            model="gpt-4o-mini",
            cost_usd=0.0001,
            agent_id="agent1",
        )

        data = usage.to_dict()

        assert data["input_tokens"] == 100
        assert data["output_tokens"] == 50
        assert data["total_tokens"] == 150
        assert data["model"] == "gpt-4o-mini"
        assert data["agent_id"] == "agent1"


class TestRunCost:
    """Tests for RunCost dataclass."""

    def test_creation(self):
        """RunCost should be created with required fields."""
        run_cost = RunCost(run_id="run-1", team_id="team-1")

        assert run_cost.run_id == "run-1"
        assert run_cost.team_id == "team-1"
        assert run_cost.total_cost == 0.0
        assert run_cost.total_tokens == 0
        assert run_cost.calls_count == 0

    def test_aggregation(self):
        """RunCost should aggregate token usage."""
        run_cost = RunCost(run_id="run-1", team_id="team-1")
        run_cost.token_usage.append(
            TokenUsage(input_tokens=500, output_tokens=200, model="gpt-4o", cost_usd=0.01)
        )
        run_cost.token_usage.append(
            TokenUsage(input_tokens=300, output_tokens=100, model="gpt-4o", cost_usd=0.005)
        )
        run_cost.total_cost = 0.015

        assert run_cost.total_tokens == 1100
        assert run_cost.total_input_tokens == 800
        assert run_cost.total_output_tokens == 300
        assert run_cost.calls_count == 2

    def test_to_dict(self):
        """RunCost.to_dict should serialize aggregated data."""
        run_cost = RunCost(run_id="run-1", team_id="team-1", total_cost=0.05)

        data = run_cost.to_dict()

        assert data["run_id"] == "run-1"
        assert data["team_id"] == "team-1"
        assert data["total_cost"] == 0.05


class TestCostTracker:
    """Tests for CostTracker."""

    def test_start_end_run(self):
        """CostTracker should track run lifecycle."""
        tracker = CostTracker()

        tracker.start_run("run-1", "team-1")
        assert tracker.current_run is not None
        assert tracker.current_cost == 0.0

        run_cost = tracker.end_run()

        assert run_cost.run_id == "run-1"
        assert run_cost.team_id == "team-1"
        assert run_cost.completed_at is not None
        assert tracker.current_run is None

    def test_record_usage(self):
        """CostTracker should record token usage."""
        tracker = CostTracker()
        tracker.start_run("run-1", "team-1")

        usage = tracker.record_usage("gpt-4o-mini", input_tokens=500, output_tokens=200)

        assert usage.input_tokens == 500
        assert usage.output_tokens == 200
        assert usage.cost_usd > 0
        assert tracker.current_cost > 0

    def test_multiple_records(self):
        """CostTracker should accumulate costs."""
        tracker = CostTracker()
        tracker.start_run("run-1", "team-1")

        tracker.record_usage("gpt-4o-mini", 500, 200, agent_id="agent1")
        tracker.record_usage("gpt-4o", 300, 100, agent_id="agent2")

        assert tracker.current_cost > 0
        run_cost = tracker.end_run()
        assert run_cost.calls_count == 2

    def test_history(self):
        """CostTracker should maintain run history."""
        tracker = CostTracker()

        tracker.start_run("run-1", "team-1")
        tracker.record_usage("gpt-4o-mini", 500, 200)
        tracker.end_run()

        tracker.start_run("run-2", "team-1")
        tracker.record_usage("gpt-4o", 1000, 500)
        tracker.end_run()

        assert len(tracker.history) == 2
        assert tracker.total_spent > 0

    def test_end_run_without_start_raises(self):
        """end_run without start should raise."""
        tracker = CostTracker()

        with pytest.raises(RuntimeError, match="No active run"):
            tracker.end_run()

    def test_on_usage_callback(self):
        """on_usage callback should fire on each record."""
        usage_log = []
        tracker = CostTracker(on_usage=lambda u: usage_log.append(u))
        tracker.start_run("run-1", "team-1")

        tracker.record_usage("gpt-4o-mini", 500, 200)
        tracker.record_usage("gpt-4o", 300, 100)

        assert len(usage_log) == 2
        assert usage_log[0].model == "gpt-4o-mini"

    def test_custom_pricing(self):
        """CostTracker should use custom pricing."""
        custom = PricingRegistry()
        custom.register("my-model", ModelPricing(input_per_1k=1.0, output_per_1k=2.0))

        tracker = CostTracker(pricing=custom)
        tracker.start_run("run-1", "team-1")

        usage = tracker.record_usage("my-model", 1000, 1000)

        # (1000/1000)*1.0 + (1000/1000)*2.0 = 3.0
        assert abs(usage.cost_usd - 3.0) < 1e-6

    def test_reset_history(self):
        """reset_history should clear completed runs."""
        tracker = CostTracker()
        tracker.start_run("run-1", "team-1")
        tracker.end_run()

        tracker.reset_history()

        assert len(tracker.history) == 0


class TestBudget:
    """Tests for Budget dataclass."""

    def test_creation(self):
        """Budget should be created with max_cost."""
        budget = Budget(max_cost=5.0)

        assert budget.max_cost == 5.0
        assert budget.period == BudgetPeriod.RUN
        assert budget.alert_threshold == 0.8
        assert budget.hard_limit is True

    def test_invalid_max_cost(self):
        """Budget should reject non-positive max_cost."""
        with pytest.raises(ValueError, match="max_cost must be positive"):
            Budget(max_cost=0.0)

    def test_invalid_threshold(self):
        """Budget should reject invalid alert_threshold."""
        with pytest.raises(ValueError, match="alert_threshold"):
            Budget(max_cost=5.0, alert_threshold=1.5)


class TestBudgetManager:
    """Tests for BudgetManager."""

    def test_check_ok(self):
        """check should return OK when under budget."""
        manager = BudgetManager(Budget(max_cost=10.0))

        assert manager.check(5.0) == BudgetStatus.OK

    def test_check_alert(self):
        """check should return ALERT at threshold."""
        manager = BudgetManager(Budget(max_cost=10.0, alert_threshold=0.8))

        assert manager.check(8.5) == BudgetStatus.ALERT

    def test_check_exceeded_hard(self):
        """check should return EXCEEDED with hard_limit."""
        manager = BudgetManager(Budget(max_cost=10.0, hard_limit=True))

        assert manager.check(10.5) == BudgetStatus.EXCEEDED

    def test_check_exceeded_soft(self):
        """check should return WARNING without hard_limit."""
        manager = BudgetManager(Budget(max_cost=10.0, hard_limit=False))

        assert manager.check(10.5) == BudgetStatus.WARNING

    def test_check_or_raise(self):
        """check_or_raise should raise on hard limit exceeded."""
        manager = BudgetManager(Budget(max_cost=5.0, hard_limit=True))

        with pytest.raises(BudgetExceededError):
            manager.check_or_raise(6.0)

    def test_alert_callback(self):
        """Alert callback should fire once at threshold."""
        alerts = []
        manager = BudgetManager(Budget(max_cost=10.0, alert_threshold=0.8))
        manager.on_alert(lambda cost, max_c: alerts.append((cost, max_c)))

        manager.check(8.5)  # Should fire alert
        manager.check(9.0)  # Should NOT fire again

        assert len(alerts) == 1
        assert alerts[0] == (8.5, 10.0)

    def test_remaining(self):
        """remaining should show remaining budget."""
        manager = BudgetManager(Budget(max_cost=10.0))

        assert manager.remaining(3.0) == 7.0
        assert manager.remaining(10.0) == 0.0
        assert manager.remaining(12.0) == -2.0

    def test_reset(self):
        """reset should allow alert to fire again."""
        alerts = []
        manager = BudgetManager(Budget(max_cost=10.0, alert_threshold=0.8))
        manager.on_alert(lambda cost, max_c: alerts.append(cost))

        manager.check(9.0)  # Alert
        manager.reset()
        manager.check(9.0)  # Alert again

        assert len(alerts) == 2

    def test_to_dict(self):
        """to_dict should serialize budget state."""
        manager = BudgetManager(Budget(max_cost=10.0))

        data = manager.to_dict(current_cost=3.0)

        assert data["max_cost"] == 10.0
        assert data["current_cost"] == 3.0
        assert data["remaining"] == 7.0
        assert data["status"] == "ok"


class TestLLMTeamCostIntegration:
    """Tests for LLMTeam cost tracking integration."""

    def test_team_has_cost_tracker(self):
        """LLMTeam should always have a cost tracker."""
        team = LLMTeam(team_id="test")

        assert team.cost_tracker is not None
        assert isinstance(team.cost_tracker, CostTracker)

    def test_team_budget_from_max_cost(self):
        """max_cost_per_run should create budget manager."""
        team = LLMTeam(team_id="test", max_cost_per_run=5.0)

        assert team.budget_manager is not None
        assert team.budget_manager.max_cost == 5.0

    def test_team_no_budget_by_default(self):
        """No budget manager without max_cost_per_run."""
        team = LLMTeam(team_id="test")

        assert team.budget_manager is None

    async def test_run_tracks_cost(self):
        """Team run should track costs in cost_tracker."""
        from llmteam.agents.orchestrator import OrchestratorConfig, OrchestratorMode

        team = LLMTeam(
            team_id="test",
            orchestration=True,
        )
        team.add_agent({
            "type": "llm",
            "role": "agent1",
            "prompt": "test",
        })

        # Mock the orchestrator to not use LLM for routing
        async def mock_decide(current_state, available_agents):
            from llmteam.agents.orchestrator import RoutingDecision
            return RoutingDecision(
                next_agent=available_agents[0] if available_agents else "",
                reason="mock routing",
            )

        team._orchestrator.decide_next_agent = mock_decide

        # Mock agent execution to return tokens
        agent = team.get_agent("agent1")

        async def mock_execute(input_data, context):
            from llmteam import AgentResult
            return AgentResult(
                output="result",
                tokens_used=750,
                model="gpt-4o-mini",
            )

        agent._execute = mock_execute

        result = await team.run({"query": "test"})

        assert result.success is True

        # Cost should have been tracked
        assert len(team.cost_tracker.history) == 1
        run_cost = team.cost_tracker.history[0]
        assert run_cost.total_tokens == 750
        assert run_cost.total_cost > 0
