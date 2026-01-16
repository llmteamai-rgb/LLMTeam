"""
Tests for orchestration strategies.

Tests cover:
- OrchestrationDecision
- OrchestrationContext
- RuleBasedStrategy
- LLMBasedStrategy
"""

import pytest
from datetime import timedelta

from llmteam.roles import (
    OrchestratorRole,
    OrchestrationDecision,
    OrchestrationContext,
    OrchestrationStrategy,
    RuleBasedStrategy,
    LLMBasedStrategy,
)


class TestOrchestratorRole:
    """Tests for OrchestratorRole enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert OrchestratorRole.ORCHESTRATION.value == "orchestration"
        assert OrchestratorRole.PROCESS_MINING.value == "process_mining"


class TestOrchestrationDecision:
    """Tests for OrchestrationDecision."""

    def test_create_decision(self):
        """Test creating a decision."""
        decision = OrchestrationDecision(
            decision_type="route",
            target_agents=["agent_1", "agent_2"],
            reason="test_reason",
            confidence=0.9,
        )

        assert decision.decision_type == "route"
        assert len(decision.target_agents) == 2
        assert decision.reason == "test_reason"
        assert decision.confidence == 0.9


class TestOrchestrationContext:
    """Tests for OrchestrationContext."""

    def test_create_context(self):
        """Test creating orchestration context."""
        context = OrchestrationContext(
            current_step="step_1",
            available_agents=["agent_1", "agent_2"],
            agent_states={"agent_1": {"status": "idle"}},
            execution_history=[],
            global_state={"key": "value"},
        )

        assert context.current_step == "step_1"
        assert len(context.available_agents) == 2
        assert context.error_rate == 0.0


class TestRuleBasedStrategy:
    """Tests for RuleBasedStrategy."""

    @pytest.mark.asyncio
    async def test_default_decision(self):
        """Test default decision when no rules match."""
        strategy = RuleBasedStrategy()

        context = OrchestrationContext(
            current_step="step_1",
            available_agents=["agent_1"],
            agent_states={},
            execution_history=[],
            global_state={},
        )

        decision = await strategy.decide(context)

        assert decision.decision_type == "route"
        assert decision.target_agents == ["agent_1"]
        assert decision.reason == "default_sequence"

    @pytest.mark.asyncio
    async def test_default_decision_no_agents(self):
        """Test default decision when no agents available."""
        strategy = RuleBasedStrategy()

        context = OrchestrationContext(
            current_step="step_1",
            available_agents=[],
            agent_states={},
            execution_history=[],
            global_state={},
        )

        decision = await strategy.decide(context)

        assert decision.decision_type == "end"
        assert decision.target_agents == []

    @pytest.mark.asyncio
    async def test_add_rule(self):
        """Test adding and applying a rule."""
        strategy = RuleBasedStrategy()

        # Add rule: if error_rate > 0.5, retry
        strategy.add_rule(lambda ctx:
            OrchestrationDecision("retry", [ctx.current_step], "high_error_rate", 0.9)
            if ctx.error_rate > 0.5 else None
        )

        context = OrchestrationContext(
            current_step="step_1",
            available_agents=["agent_1"],
            agent_states={},
            execution_history=[],
            global_state={},
            error_rate=0.8,
        )

        decision = await strategy.decide(context)

        assert decision.decision_type == "retry"
        assert decision.reason == "high_error_rate"

    @pytest.mark.asyncio
    async def test_rule_chain(self):
        """Test that rules are checked in order."""
        strategy = RuleBasedStrategy()

        # First rule: high error rate
        strategy.add_rule(lambda ctx:
            OrchestrationDecision("retry", [], "high_error_rate", 0.9)
            if ctx.error_rate > 0.5 else None
        )

        # Second rule: timeout
        strategy.add_rule(lambda ctx:
            OrchestrationDecision("escalate", [], "timeout", 0.8)
            if ctx.step_duration.total_seconds() > 600 else None
        )

        # First rule matches
        context = OrchestrationContext(
            current_step="step_1",
            available_agents=["agent_1"],
            agent_states={},
            execution_history=[],
            global_state={},
            error_rate=0.8,
        )

        decision = await strategy.decide(context)
        assert decision.decision_type == "retry"

        # Second rule matches
        context.error_rate = 0.1
        context.step_duration = timedelta(seconds=700)

        decision = await strategy.decide(context)
        assert decision.decision_type == "escalate"


class TestLLMBasedStrategy:
    """Tests for LLMBasedStrategy."""

    @pytest.mark.asyncio
    async def test_with_mock_llm(self):
        """Test LLM-based strategy with mock LLM."""
        class MockLLM:
            async def generate(self, prompt: str) -> str:
                return '{"decision": "route", "targets": ["agent_1"], "reason": "llm_decision", "confidence": 0.9}'

        llm = MockLLM()
        strategy = LLMBasedStrategy(llm)

        context = OrchestrationContext(
            current_step="step_1",
            available_agents=["agent_1", "agent_2"],
            agent_states={},
            execution_history=[],
            global_state={},
        )

        decision = await strategy.decide(context)

        assert decision.decision_type == "route"
        assert decision.target_agents == ["agent_1"]
        assert decision.reason == "llm_decision"
        assert decision.confidence == 0.9

    @pytest.mark.asyncio
    async def test_with_invalid_json(self):
        """Test LLM-based strategy with invalid JSON response."""
        class MockLLM:
            async def generate(self, prompt: str) -> str:
                return "invalid json"

        llm = MockLLM()
        strategy = LLMBasedStrategy(llm)

        context = OrchestrationContext(
            current_step="step_1",
            available_agents=["agent_1"],
            agent_states={},
            execution_history=[],
            global_state={},
        )

        decision = await strategy.decide(context)

        # Should fallback to default decision
        assert decision.reason == "llm_parse_error"
        assert decision.confidence == 0.5

    @pytest.mark.asyncio
    async def test_custom_prompt_template(self):
        """Test LLM-based strategy with custom prompt."""
        prompt_called_with = []

        class MockLLM:
            async def generate(self, prompt: str) -> str:
                prompt_called_with.append(prompt)
                return '{"decision": "route", "targets": [], "reason": "test"}'

        llm = MockLLM()
        custom_template = "Custom prompt: {current_step}"
        strategy = LLMBasedStrategy(llm, prompt_template=custom_template)

        context = OrchestrationContext(
            current_step="my_step",
            available_agents=[],
            agent_states={},
            execution_history=[],
            global_state={},
        )

        await strategy.decide(context)

        assert "Custom prompt: my_step" in prompt_called_with[0]
