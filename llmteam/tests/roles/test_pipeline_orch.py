"""
Tests for pipeline orchestrator.

Tests cover:
- Agent registration
- Pipeline execution
- Process mining integration
- Orchestration strategies
"""

import pytest
from datetime import datetime

from llmteam.roles import (
    PipelineOrchestrator,
    RuleBasedStrategy,
    OrchestrationDecision,
)


# Mock agent for testing
class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str, result: dict = None):
        self.name = name
        self.result = result or {"processed_by": name}
        self.process_calls = []

    async def process(self, input_data: dict) -> dict:
        """Process input (mock implementation)."""
        self.process_calls.append(input_data)
        return self.result

    def get_state(self) -> dict:
        """Get agent state."""
        return {"status": "available"}


class TestPipelineOrchestrator:
    """Tests for PipelineOrchestrator."""

    def test_create_orchestrator(self):
        """Test creating an orchestrator."""
        orchestrator = PipelineOrchestrator(
            pipeline_id="test_pipeline",
        )

        assert orchestrator.pipeline_id == "test_pipeline"
        assert orchestrator.strategy is not None

    def test_create_with_custom_strategy(self):
        """Test creating orchestrator with custom strategy."""
        strategy = RuleBasedStrategy()
        orchestrator = PipelineOrchestrator(
            pipeline_id="test_pipeline",
            strategy=strategy,
        )

        assert orchestrator.strategy == strategy

    def test_create_with_process_mining_disabled(self):
        """Test creating orchestrator without process mining."""
        orchestrator = PipelineOrchestrator(
            pipeline_id="test_pipeline",
            enable_process_mining=False,
        )

        assert orchestrator.process_mining is None

    def test_register_agent(self):
        """Test registering an agent."""
        orchestrator = PipelineOrchestrator(pipeline_id="test_pipeline")
        agent = MockAgent("agent_1")

        orchestrator.register_agent("agent_1", agent)

        assert "agent_1" in orchestrator._agents

    def test_unregister_agent(self):
        """Test unregistering an agent."""
        orchestrator = PipelineOrchestrator(pipeline_id="test_pipeline")
        agent = MockAgent("agent_1")

        orchestrator.register_agent("agent_1", agent)
        assert "agent_1" in orchestrator._agents

        orchestrator.unregister_agent("agent_1")
        assert "agent_1" not in orchestrator._agents

    @pytest.mark.asyncio
    async def test_orchestrate_simple(self):
        """Test simple orchestration."""
        # Create strategy that routes to agent_1, then ends
        strategy = RuleBasedStrategy()
        call_count = [0]

        def route_rule(ctx):
            call_count[0] += 1
            if call_count[0] == 1:
                return OrchestrationDecision(
                    "route",
                    ["agent_1"],
                    "first_step",
                    1.0,
                )
            else:
                return OrchestrationDecision(
                    "end",
                    [],
                    "finished",
                    1.0,
                )

        strategy.add_rule(route_rule)

        orchestrator = PipelineOrchestrator(
            pipeline_id="test_pipeline",
            strategy=strategy,
        )

        agent = MockAgent("agent_1", {"result": "success"})
        orchestrator.register_agent("agent_1", agent)

        result = await orchestrator.orchestrate("run_1", {"input": "data"})

        assert result["result"] == "success"
        assert len(agent.process_calls) == 1

    @pytest.mark.asyncio
    async def test_orchestrate_with_process_mining(self):
        """Test orchestration with process mining enabled."""
        strategy = RuleBasedStrategy()
        call_count = [0]

        def route_rule(ctx):
            call_count[0] += 1
            if call_count[0] == 1:
                return OrchestrationDecision(
                    "route",
                    ["agent_1"],
                    "process_step",
                    1.0,
                )
            else:
                return OrchestrationDecision(
                    "end",
                    [],
                    "finished",
                    1.0,
                )

        strategy.add_rule(route_rule)

        orchestrator = PipelineOrchestrator(
            pipeline_id="test_pipeline",
            strategy=strategy,
            enable_process_mining=True,
        )

        agent = MockAgent("agent_1")
        orchestrator.register_agent("agent_1", agent)

        await orchestrator.orchestrate("run_1", {})

        # Check process mining recorded events
        assert orchestrator.process_mining.get_event_count() > 0
        assert orchestrator.process_mining.get_case_count() == 1

    @pytest.mark.asyncio
    async def test_get_process_metrics(self):
        """Test getting process metrics."""
        strategy = RuleBasedStrategy()
        strategy.add_rule(lambda ctx: OrchestrationDecision("end", [], "done", 1.0))

        orchestrator = PipelineOrchestrator(
            pipeline_id="test_pipeline",
            strategy=strategy,
            enable_process_mining=True,
        )

        await orchestrator.orchestrate("run_1", {})

        metrics = orchestrator.get_process_metrics()
        assert metrics is not None

    @pytest.mark.asyncio
    async def test_get_process_metrics_disabled(self):
        """Test getting process metrics when disabled."""
        orchestrator = PipelineOrchestrator(
            pipeline_id="test_pipeline",
            enable_process_mining=False,
        )

        metrics = orchestrator.get_process_metrics()
        assert metrics is None

    @pytest.mark.asyncio
    async def test_export_process_model(self):
        """Test exporting process model to XES."""
        strategy = RuleBasedStrategy()
        strategy.add_rule(lambda ctx: OrchestrationDecision("end", [], "done", 1.0))

        orchestrator = PipelineOrchestrator(
            pipeline_id="test_pipeline",
            strategy=strategy,
            enable_process_mining=True,
        )

        await orchestrator.orchestrate("run_1", {})

        xes = orchestrator.export_process_model()
        assert xes is not None
        assert '<?xml version' in xes
        assert '<log>' in xes

    @pytest.mark.asyncio
    async def test_get_execution_history(self):
        """Test getting execution history."""
        strategy = RuleBasedStrategy()
        call_count = [0]

        def route_rule(ctx):
            call_count[0] += 1
            if call_count[0] == 1:
                return OrchestrationDecision(
                    "route",
                    ["agent_1"],
                    "step1",
                    1.0,
                )
            else:
                return OrchestrationDecision(
                    "end",
                    [],
                    "done",
                    1.0,
                )

        strategy.add_rule(route_rule)

        orchestrator = PipelineOrchestrator(
            pipeline_id="test_pipeline",
            strategy=strategy,
        )

        agent = MockAgent("agent_1")
        orchestrator.register_agent("agent_1", agent)

        await orchestrator.orchestrate("run_1", {})

        history = orchestrator.get_execution_history()
        assert len(history) > 0
        assert "step" in history[0]
        assert "decision" in history[0]

    @pytest.mark.asyncio
    async def test_orchestrate_with_missing_agent(self):
        """Test orchestration when agent is not registered."""
        strategy = RuleBasedStrategy()
        strategy.add_rule(lambda ctx: OrchestrationDecision(
            "route",
            ["nonexistent_agent"],
            "test",
            1.0,
        ))

        orchestrator = PipelineOrchestrator(
            pipeline_id="test_pipeline",
            strategy=strategy,
        )

        # This should not raise an error, just skip the agent
        result = await orchestrator.orchestrate("run_1", {"input": "data"})

        # Should still have the input data
        assert "input" in result
