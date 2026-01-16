"""
Group orchestrator for llmteam.

Manages multiple pipelines with load balancing, content-based routing, and parallel execution.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List


class GroupDecisionType(Enum):
    """
    Types of decisions for group orchestrator.

    Attributes:
        ROUTE_TO_PIPELINE: Route to a single pipeline
        PARALLEL_PIPELINES: Execute multiple pipelines in parallel
        AGGREGATE_RESULTS: Aggregate results from multiple pipelines
        ESCALATE: Escalate to higher level
    """

    ROUTE_TO_PIPELINE = "route_to_pipeline"
    PARALLEL_PIPELINES = "parallel_pipelines"
    AGGREGATE_RESULTS = "aggregate_results"
    ESCALATE = "escalate"


@dataclass
class GroupOrchestrationDecision:
    """
    Decision made by group orchestrator.

    Attributes:
        decision_type: Type of decision
        target_pipelines: List of pipeline IDs to execute
        aggregation_strategy: How to aggregate results (merge, vote, first, all)
        reason: Reasoning behind the decision
        metadata: Additional metadata
    """

    decision_type: GroupDecisionType
    target_pipelines: List[str]
    aggregation_strategy: str = "merge"
    reason: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class PipelineStatus:
    """
    Status of a pipeline.

    Attributes:
        pipeline_id: Pipeline identifier
        status: Current status (idle, running, completed, failed)
        progress: Progress percentage (0.0-1.0)
        current_step: Current step in the pipeline
        error_count: Number of errors encountered
        last_update: Last update timestamp
    """

    pipeline_id: str
    status: str
    progress: float
    current_step: str
    error_count: int
    last_update: datetime


class GroupOrchestrationStrategy:
    """
    Base class for group orchestration strategies.

    Subclasses must implement the decide() method.
    """

    async def decide(
        self,
        pipelines: Dict[str, PipelineStatus],
        input_data: dict,
    ) -> GroupOrchestrationDecision:
        """
        Make a group orchestration decision.

        Args:
            pipelines: Status of all pipelines
            input_data: Input data to process

        Returns:
            GroupOrchestrationDecision
        """
        raise NotImplementedError


class LoadBalancingStrategy(GroupOrchestrationStrategy):
    """
    Load balancing strategy for group orchestrator.

    Routes requests to the least loaded pipeline.

    Example:
        strategy = LoadBalancingStrategy()
        decision = await strategy.decide(pipelines, input_data)
    """

    async def decide(
        self,
        pipelines: Dict[str, PipelineStatus],
        input_data: dict,
    ) -> GroupOrchestrationDecision:
        """
        Find least loaded pipeline and route to it.

        Args:
            pipelines: Pipeline statuses
            input_data: Input data

        Returns:
            Decision to route to least loaded pipeline
        """
        # Find idle pipelines first
        idle_pipelines = [
            p for p in pipelines.values()
            if p.status == "idle"
        ]

        if idle_pipelines:
            return GroupOrchestrationDecision(
                decision_type=GroupDecisionType.ROUTE_TO_PIPELINE,
                target_pipelines=[idle_pipelines[0].pipeline_id],
                reason="load_balancing:idle",
            )

        # All busy - find pipeline with least progress
        if pipelines:
            least_loaded = min(pipelines.values(), key=lambda p: p.progress)
            return GroupOrchestrationDecision(
                decision_type=GroupDecisionType.ROUTE_TO_PIPELINE,
                target_pipelines=[least_loaded.pipeline_id],
                reason="load_balancing:least_progress",
            )

        # No pipelines available
        return GroupOrchestrationDecision(
            decision_type=GroupDecisionType.ESCALATE,
            target_pipelines=[],
            reason="no_pipelines_available",
        )


class ContentBasedRoutingStrategy(GroupOrchestrationStrategy):
    """
    Content-based routing strategy.

    Routes requests based on keywords in the input data.

    Example:
        strategy = ContentBasedRoutingStrategy({
            "urgent": ["fast_pipeline"],
            "complex": ["detailed_pipeline"],
            "financial": ["finance_pipeline"],
        })
        decision = await strategy.decide(pipelines, input_data)
    """

    def __init__(self, routing_rules: Dict[str, List[str]]):
        """
        Initialize content-based routing strategy.

        Args:
            routing_rules: Mapping of keywords to pipeline IDs
        """
        self.routing_rules = routing_rules

    async def decide(
        self,
        pipelines: Dict[str, PipelineStatus],
        input_data: dict,
    ) -> GroupOrchestrationDecision:
        """
        Route based on content keywords.

        Args:
            pipelines: Pipeline statuses
            input_data: Input data

        Returns:
            Decision based on keyword matching
        """
        input_text = str(input_data).lower()

        # Check each routing rule
        for keyword, pipeline_ids in self.routing_rules.items():
            if keyword.lower() in input_text:
                # Filter to available pipelines
                available = [pid for pid in pipeline_ids if pid in pipelines]
                if available:
                    return GroupOrchestrationDecision(
                        decision_type=GroupDecisionType.ROUTE_TO_PIPELINE,
                        target_pipelines=available,
                        reason=f"keyword_match:{keyword}",
                    )

        # Default: first available pipeline
        if pipelines:
            return GroupOrchestrationDecision(
                decision_type=GroupDecisionType.ROUTE_TO_PIPELINE,
                target_pipelines=[list(pipelines.keys())[0]],
                reason="default_routing",
            )

        return GroupOrchestrationDecision(
            decision_type=GroupDecisionType.ESCALATE,
            target_pipelines=[],
            reason="no_pipelines_available",
        )


class ParallelFanOutStrategy(GroupOrchestrationStrategy):
    """
    Parallel fan-out strategy.

    Executes all pipelines in parallel and aggregates results.

    Example:
        strategy = ParallelFanOutStrategy(aggregation="merge")
        decision = await strategy.decide(pipelines, input_data)
    """

    def __init__(self, aggregation: str = "merge"):
        """
        Initialize parallel fan-out strategy.

        Args:
            aggregation: Aggregation strategy (merge, vote, first, all)
        """
        self.aggregation = aggregation

    async def decide(
        self,
        pipelines: Dict[str, PipelineStatus],
        input_data: dict,
    ) -> GroupOrchestrationDecision:
        """
        Execute all pipelines in parallel.

        Args:
            pipelines: Pipeline statuses
            input_data: Input data

        Returns:
            Decision to execute pipelines in parallel
        """
        return GroupOrchestrationDecision(
            decision_type=GroupDecisionType.PARALLEL_PIPELINES,
            target_pipelines=list(pipelines.keys()),
            aggregation_strategy=self.aggregation,
            reason="fan_out",
        )


class GroupOrchestrator:
    """
    Group orchestrator for managing multiple pipelines.

    Features:
    - Load balancing across pipelines
    - Content-based routing
    - Parallel pipeline execution
    - Result aggregation

    Example:
        # Create group orchestrator
        group = GroupOrchestrator(
            group_id="main_group",
            strategy=LoadBalancingStrategy(),
        )

        # Register pipelines
        group.register_pipeline(pipeline_a)
        group.register_pipeline(pipeline_b)

        # Execute
        result = await group.orchestrate("run_1", input_data)

        # Get metrics
        metrics = group.get_group_metrics()
    """

    def __init__(
        self,
        group_id: str,
        strategy: GroupOrchestrationStrategy = None,
    ):
        """
        Initialize group orchestrator.

        Args:
            group_id: Unique identifier for this group
            strategy: Group orchestration strategy (defaults to LoadBalancingStrategy)
        """
        self.group_id = group_id
        self.strategy = strategy or LoadBalancingStrategy()

        self._pipelines: Dict[str, Any] = {}
        self._statuses: Dict[str, PipelineStatus] = {}

    def register_pipeline(self, pipeline: Any) -> None:
        """
        Register a pipeline with this group.

        Args:
            pipeline: Pipeline instance (must have pipeline_id attribute)
        """
        pipeline_id = getattr(pipeline, 'pipeline_id', str(id(pipeline)))
        self._pipelines[pipeline_id] = pipeline
        self._statuses[pipeline_id] = PipelineStatus(
            pipeline_id=pipeline_id,
            status="idle",
            progress=0.0,
            current_step="",
            error_count=0,
            last_update=datetime.now(),
        )

    def unregister_pipeline(self, pipeline_id: str) -> None:
        """
        Unregister a pipeline.

        Args:
            pipeline_id: Pipeline ID to remove
        """
        if pipeline_id in self._pipelines:
            del self._pipelines[pipeline_id]
        if pipeline_id in self._statuses:
            del self._statuses[pipeline_id]

    async def orchestrate(self, run_id: str, input_data: dict) -> dict:
        """
        Execute group orchestration.

        Args:
            run_id: Unique run identifier
            input_data: Input data for the group

        Returns:
            Orchestration result
        """
        decision = await self.strategy.decide(self._statuses, input_data)

        if decision.decision_type == GroupDecisionType.ROUTE_TO_PIPELINE:
            # Single pipeline execution
            if decision.target_pipelines:
                pipeline_id = decision.target_pipelines[0]
                pipeline = self._pipelines.get(pipeline_id)
                if pipeline:
                    self._update_status(pipeline_id, "running")
                    try:
                        result = await pipeline.orchestrate(run_id, input_data)
                        self._update_status(pipeline_id, "completed")
                        return result
                    except Exception as e:
                        self._update_status(pipeline_id, "failed")
                        return {"error": str(e)}

            return {"error": "no_pipeline_available"}

        elif decision.decision_type == GroupDecisionType.PARALLEL_PIPELINES:
            # Parallel execution
            tasks = []
            for pipeline_id in decision.target_pipelines:
                pipeline = self._pipelines.get(pipeline_id)
                if pipeline:
                    self._update_status(pipeline_id, "running")
                    tasks.append(pipeline.orchestrate(f"{run_id}_{pipeline_id}", input_data))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Update statuses
            for pipeline_id in decision.target_pipelines:
                self._update_status(pipeline_id, "completed")

            # Aggregate results
            return self._aggregate_results(
                results,
                decision.aggregation_strategy,
            )

        return {}

    def _update_status(self, pipeline_id: str, status: str) -> None:
        """
        Update pipeline status.

        Args:
            pipeline_id: Pipeline ID
            status: New status
        """
        if pipeline_id in self._statuses:
            self._statuses[pipeline_id].status = status
            self._statuses[pipeline_id].last_update = datetime.now()

    def _aggregate_results(self, results: List, strategy: str) -> dict:
        """
        Aggregate results from multiple pipelines.

        Args:
            results: List of results
            strategy: Aggregation strategy

        Returns:
            Aggregated result
        """
        valid_results = [r for r in results if not isinstance(r, Exception)]

        if strategy == "first":
            return valid_results[0] if valid_results else {}

        elif strategy == "vote":
            # Majority vote (simplified)
            return valid_results[0] if valid_results else {}

        elif strategy == "merge":
            merged = {}
            for r in valid_results:
                if isinstance(r, dict):
                    merged.update(r)
            return merged

        elif strategy == "all":
            return {"results": valid_results}

        return {}

    def get_pipeline_statuses(self) -> Dict[str, PipelineStatus]:
        """
        Get status of all pipelines.

        Returns:
            Dictionary of pipeline statuses
        """
        return self._statuses.copy()

    def get_group_metrics(self) -> dict:
        """
        Get aggregated metrics for the group.

        Returns:
            Dictionary with group metrics
        """
        all_metrics = []
        for pipeline in self._pipelines.values():
            if hasattr(pipeline, 'get_process_metrics'):
                m = pipeline.get_process_metrics()
                if m:
                    all_metrics.append(m)

        if not all_metrics:
            return {
                "total_pipelines": len(self._pipelines),
            }

        return {
            "total_pipelines": len(self._pipelines),
            "avg_completion_rate": sum(m.completion_rate for m in all_metrics) / len(all_metrics),
            "avg_error_rate": sum(m.error_rate for m in all_metrics) / len(all_metrics),
            "bottlenecks": list(set(
                b for m in all_metrics for b in m.bottleneck_activities
            )),
        }
