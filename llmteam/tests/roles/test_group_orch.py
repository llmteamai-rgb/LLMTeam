"""
Tests for group orchestrator.

Tests cover:
- Group orchestration strategies
- Pipeline registration
- Load balancing
- Content-based routing
- Parallel execution
- Result aggregation
"""

import pytest

from llmteam.roles import (
    GroupDecisionType,
    GroupOrchestrationDecision,
    PipelineStatus,
    GroupOrchestrator,
    LoadBalancingStrategy,
    ContentBasedRoutingStrategy,
    ParallelFanOutStrategy,
)


# Mock pipeline for testing
class MockPipeline:
    """Mock pipeline for testing."""

    def __init__(self, pipeline_id: str, result: dict = None):
        self.pipeline_id = pipeline_id
        self.result = result or {"processed_by": pipeline_id}
        self.orchestrate_calls = []

    async def orchestrate(self, run_id: str, input_data: dict) -> dict:
        """Execute orchestration (mock implementation)."""
        self.orchestrate_calls.append((run_id, input_data))
        return self.result

    def get_process_metrics(self):
        """Get mock metrics."""
        from llmteam.roles import ProcessMetrics
        from datetime import timedelta

        return ProcessMetrics(
            avg_duration=timedelta(seconds=5),
            min_duration=timedelta(seconds=1),
            max_duration=timedelta(seconds=10),
            cases_per_hour=100.0,
            completion_rate=0.95,
            error_rate=0.05,
            retry_rate=0.02,
            bottleneck_activities=["activity_1"],
            waiting_time_by_activity={},
        )


class TestGroupDecisionType:
    """Tests for GroupDecisionType enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert GroupDecisionType.ROUTE_TO_PIPELINE.value == "route_to_pipeline"
        assert GroupDecisionType.PARALLEL_PIPELINES.value == "parallel_pipelines"


class TestLoadBalancingStrategy:
    """Tests for LoadBalancingStrategy."""

    @pytest.mark.asyncio
    async def test_route_to_idle_pipeline(self):
        """Test routing to idle pipeline."""
        from datetime import datetime

        strategy = LoadBalancingStrategy()

        pipelines = {
            "pipeline_1": PipelineStatus("pipeline_1", "idle", 0.0, "", 0, datetime.now()),
            "pipeline_2": PipelineStatus("pipeline_2", "running", 0.5, "", 0, datetime.now()),
        }

        decision = await strategy.decide(pipelines, {"test": "data"})

        assert decision.decision_type == GroupDecisionType.ROUTE_TO_PIPELINE
        assert "pipeline_1" in decision.target_pipelines
        assert decision.reason == "load_balancing:idle"

    @pytest.mark.asyncio
    async def test_route_to_least_loaded(self):
        """Test routing to least loaded pipeline when all busy."""
        from datetime import datetime

        strategy = LoadBalancingStrategy()

        pipelines = {
            "pipeline_1": PipelineStatus("pipeline_1", "running", 0.8, "", 0, datetime.now()),
            "pipeline_2": PipelineStatus("pipeline_2", "running", 0.3, "", 0, datetime.now()),
        }

        decision = await strategy.decide(pipelines, {"test": "data"})

        assert decision.decision_type == GroupDecisionType.ROUTE_TO_PIPELINE
        assert "pipeline_2" in decision.target_pipelines

    @pytest.mark.asyncio
    async def test_no_pipelines(self):
        """Test behavior with no pipelines."""
        strategy = LoadBalancingStrategy()

        decision = await strategy.decide({}, {"test": "data"})

        assert decision.decision_type == GroupDecisionType.ESCALATE


class TestContentBasedRoutingStrategy:
    """Tests for ContentBasedRoutingStrategy."""

    @pytest.mark.asyncio
    async def test_keyword_match(self):
        """Test routing based on keyword match."""
        from datetime import datetime

        strategy = ContentBasedRoutingStrategy({
            "urgent": ["fast_pipeline"],
            "complex": ["detailed_pipeline"],
        })

        pipelines = {
            "fast_pipeline": PipelineStatus("fast_pipeline", "idle", 0.0, "", 0, datetime.now()),
            "detailed_pipeline": PipelineStatus("detailed_pipeline", "idle", 0.0, "", 0, datetime.now()),
        }

        decision = await strategy.decide(pipelines, {"query": "urgent request"})

        assert decision.decision_type == GroupDecisionType.ROUTE_TO_PIPELINE
        assert "fast_pipeline" in decision.target_pipelines
        assert "urgent" in decision.reason

    @pytest.mark.asyncio
    async def test_default_routing(self):
        """Test default routing when no keywords match."""
        from datetime import datetime

        strategy = ContentBasedRoutingStrategy({
            "urgent": ["fast_pipeline"],
        })

        pipelines = {
            "pipeline_1": PipelineStatus("pipeline_1", "idle", 0.0, "", 0, datetime.now()),
        }

        decision = await strategy.decide(pipelines, {"query": "normal request"})

        assert decision.decision_type == GroupDecisionType.ROUTE_TO_PIPELINE
        assert decision.reason == "default_routing"


class TestParallelFanOutStrategy:
    """Tests for ParallelFanOutStrategy."""

    @pytest.mark.asyncio
    async def test_fan_out_all_pipelines(self):
        """Test parallel execution of all pipelines."""
        from datetime import datetime

        strategy = ParallelFanOutStrategy(aggregation="merge")

        pipelines = {
            "pipeline_1": PipelineStatus("pipeline_1", "idle", 0.0, "", 0, datetime.now()),
            "pipeline_2": PipelineStatus("pipeline_2", "idle", 0.0, "", 0, datetime.now()),
        }

        decision = await strategy.decide(pipelines, {})

        assert decision.decision_type == GroupDecisionType.PARALLEL_PIPELINES
        assert len(decision.target_pipelines) == 2
        assert decision.aggregation_strategy == "merge"


class TestGroupOrchestrator:
    """Tests for GroupOrchestrator."""

    def test_create_orchestrator(self):
        """Test creating group orchestrator."""
        orchestrator = GroupOrchestrator(group_id="test_group")

        assert orchestrator.group_id == "test_group"

    def test_register_pipeline(self):
        """Test registering a pipeline."""
        orchestrator = GroupOrchestrator(group_id="test_group")
        pipeline = MockPipeline("pipeline_1")

        orchestrator.register_pipeline(pipeline)

        assert "pipeline_1" in orchestrator._pipelines
        assert "pipeline_1" in orchestrator._statuses

    def test_unregister_pipeline(self):
        """Test unregistering a pipeline."""
        orchestrator = GroupOrchestrator(group_id="test_group")
        pipeline = MockPipeline("pipeline_1")

        orchestrator.register_pipeline(pipeline)
        assert "pipeline_1" in orchestrator._pipelines

        orchestrator.unregister_pipeline("pipeline_1")
        assert "pipeline_1" not in orchestrator._pipelines

    @pytest.mark.asyncio
    async def test_orchestrate_single_pipeline(self):
        """Test orchestrating with single pipeline routing."""
        orchestrator = GroupOrchestrator(
            group_id="test_group",
            strategy=LoadBalancingStrategy(),
        )

        pipeline = MockPipeline("pipeline_1", {"result": "success"})
        orchestrator.register_pipeline(pipeline)

        result = await orchestrator.orchestrate("run_1", {"input": "data"})

        assert result["result"] == "success"
        assert len(pipeline.orchestrate_calls) == 1

    @pytest.mark.asyncio
    async def test_orchestrate_parallel(self):
        """Test parallel pipeline execution."""
        orchestrator = GroupOrchestrator(
            group_id="test_group",
            strategy=ParallelFanOutStrategy(aggregation="merge"),
        )

        pipeline1 = MockPipeline("pipeline_1", {"result1": "value1"})
        pipeline2 = MockPipeline("pipeline_2", {"result2": "value2"})

        orchestrator.register_pipeline(pipeline1)
        orchestrator.register_pipeline(pipeline2)

        result = await orchestrator.orchestrate("run_1", {"input": "data"})

        # Results should be merged
        assert "result1" in result
        assert "result2" in result

    @pytest.mark.asyncio
    async def test_aggregate_results_merge(self):
        """Test merging results."""
        orchestrator = GroupOrchestrator(group_id="test_group")

        results = [
            {"a": 1, "b": 2},
            {"c": 3, "d": 4},
        ]

        aggregated = orchestrator._aggregate_results(results, "merge")

        assert aggregated["a"] == 1
        assert aggregated["c"] == 3

    @pytest.mark.asyncio
    async def test_aggregate_results_first(self):
        """Test taking first result."""
        orchestrator = GroupOrchestrator(group_id="test_group")

        results = [
            {"result": "first"},
            {"result": "second"},
        ]

        aggregated = orchestrator._aggregate_results(results, "first")

        assert aggregated["result"] == "first"

    @pytest.mark.asyncio
    async def test_aggregate_results_all(self):
        """Test collecting all results."""
        orchestrator = GroupOrchestrator(group_id="test_group")

        results = [
            {"result": "first"},
            {"result": "second"},
        ]

        aggregated = orchestrator._aggregate_results(results, "all")

        assert "results" in aggregated
        assert len(aggregated["results"]) == 2

    def test_get_pipeline_statuses(self):
        """Test getting pipeline statuses."""
        orchestrator = GroupOrchestrator(group_id="test_group")

        pipeline = MockPipeline("pipeline_1")
        orchestrator.register_pipeline(pipeline)

        statuses = orchestrator.get_pipeline_statuses()

        assert "pipeline_1" in statuses
        assert statuses["pipeline_1"].status == "idle"

    def test_get_group_metrics(self):
        """Test getting group metrics."""
        orchestrator = GroupOrchestrator(group_id="test_group")

        pipeline = MockPipeline("pipeline_1")
        orchestrator.register_pipeline(pipeline)

        metrics = orchestrator.get_group_metrics()

        assert "total_pipelines" in metrics
        assert metrics["total_pipelines"] == 1

    def test_get_group_metrics_with_process_mining(self):
        """Test getting group metrics with process mining data."""
        orchestrator = GroupOrchestrator(group_id="test_group")

        pipeline = MockPipeline("pipeline_1")
        orchestrator.register_pipeline(pipeline)

        metrics = orchestrator.get_group_metrics()

        # Should aggregate metrics from pipelines
        assert "avg_completion_rate" in metrics
        assert "avg_error_rate" in metrics
        assert "bottlenecks" in metrics

    @pytest.mark.asyncio
    async def test_orchestrate_with_no_pipelines(self):
        """Test orchestration with no pipelines registered."""
        orchestrator = GroupOrchestrator(group_id="test_group")

        result = await orchestrator.orchestrate("run_1", {"input": "data"})

        assert "error" in result or result == {}
