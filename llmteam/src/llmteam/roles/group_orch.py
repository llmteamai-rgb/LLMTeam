"""
Group orchestrator for llmteam.

Manages multiple pipelines with load balancing, content-based routing, and parallel execution.

v2.3.0: GroupOrchestrator is being transitioned from a Router to a Coordinator/Supervisor role.
Use Canvas with TeamHandler for routing. GroupOrchestrator now focuses on escalation handling
and metrics collection.
"""

import asyncio
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypedDict

from llmteam.observability import get_logger


logger = get_logger(__name__)


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


class GroupOrchestrationDecisionDict(TypedDict):
    """Dictionary representation of GroupOrchestrationDecision."""
    decision_type: str
    target_pipelines: List[str]
    aggregation_strategy: str
    reason: str
    metadata: Dict


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

    def to_dict(self) -> GroupOrchestrationDecisionDict:
        """Convert to dictionary."""
        return {
            "decision_type": self.decision_type.value,
            "target_pipelines": self.target_pipelines,
            "aggregation_strategy": self.aggregation_strategy,
            "reason": self.reason,
            "metadata": self.metadata,
        }


class PipelineStatusDict(TypedDict):
    """Dictionary representation of PipelineStatus."""
    pipeline_id: str
    status: str
    progress: float
    current_step: str
    error_count: int
    last_update: str


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

    def to_dict(self) -> PipelineStatusDict:
        """Convert to dictionary."""
        return {
            "pipeline_id": self.pipeline_id,
            "status": self.status,
            "progress": self.progress,
            "current_step": self.current_step,
            "error_count": self.error_count,
            "last_update": self.last_update.isoformat(),
        }


class EscalationLevel(Enum):
    """Escalation severity levels."""

    INFO = "info"  # Informational escalation
    WARNING = "warning"  # Needs attention
    CRITICAL = "critical"  # Immediate action required
    EMERGENCY = "emergency"  # System-wide issue


class EscalationDict(TypedDict):
    """Dictionary representation of Escalation."""
    escalation_id: str
    level: str
    source_pipeline: str
    reason: str
    context: Dict[str, Any]
    timestamp: str


@dataclass
class Escalation:
    """
    Escalation request from a pipeline.

    Attributes:
        escalation_id: Unique escalation identifier
        level: Severity level
        source_pipeline: Pipeline that raised the escalation
        reason: Reason for escalation
        context: Additional context data
        timestamp: When escalation was raised
    """

    escalation_id: str
    level: EscalationLevel
    source_pipeline: str
    reason: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> EscalationDict:
        """Convert to dictionary."""
        return {
            "escalation_id": self.escalation_id,
            "level": self.level.value,
            "source_pipeline": self.source_pipeline,
            "reason": self.reason,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
        }


class EscalationAction(Enum):
    """Actions that can be taken on escalation."""

    ACKNOWLEDGE = "acknowledge"  # Acknowledge and continue
    RETRY = "retry"  # Retry the operation
    REDIRECT = "redirect"  # Redirect to another pipeline
    ABORT = "abort"  # Abort the operation
    HUMAN_REVIEW = "human_review"  # Request human intervention


@dataclass
class EscalationDecision:
    """
    Decision on how to handle an escalation.

    Attributes:
        action: Action to take
        target_pipeline: Target pipeline for redirect (if applicable)
        message: Message for logging/notification
        metadata: Additional metadata
    """

    action: EscalationAction
    target_pipeline: Optional[str] = None
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action.value,
            "target_pipeline": self.target_pipeline,
            "message": self.message,
            "metadata": self.metadata,
        }


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
        self._escalation_history: List[Escalation] = []

        logger.debug(f"GroupOrchestrator initialized for {group_id}")

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
        logger.debug(f"Registered pipeline '{pipeline_id}' to group '{self.group_id}'")

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
        logger.debug(f"Unregistered pipeline '{pipeline_id}' from group '{self.group_id}'")

    async def orchestrate(self, run_id: str, input_data: dict) -> dict:
        """
        Execute group orchestration.

        .. deprecated:: 2.3.0
            Use Canvas with TeamHandler for routing. GroupOrchestrator is being
            transitioned to a Coordinator/Supervisor role focused on escalation
            handling and metrics collection.

        Args:
            run_id: Unique run identifier
            input_data: Input data for the group

        Returns:
            Orchestration result
        """
        warnings.warn(
            "GroupOrchestrator.orchestrate() is deprecated since v2.3.0. "
            "Use Canvas with TeamHandler for routing instead. "
            "GroupOrchestrator now focuses on escalation handling and metrics.",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.info(f"Starting group orchestration for run {run_id} in group {self.group_id}")
        
        try:
            decision = await self.strategy.decide(self._statuses, input_data)
            logger.debug(f"Group decision: {decision.decision_type.value} -> {decision.target_pipelines}")

            if decision.decision_type == GroupDecisionType.ROUTE_TO_PIPELINE:
                # Single pipeline execution
                if decision.target_pipelines:
                    pipeline_id = decision.target_pipelines[0]
                    pipeline = self._pipelines.get(pipeline_id)
                    if pipeline:
                        logger.debug(f"Routing to pipeline '{pipeline_id}'")
                        self._update_status(pipeline_id, "running")
                        try:
                            result = await pipeline.orchestrate(run_id, input_data)
                            self._update_status(pipeline_id, "completed")
                            return result
                        except Exception as e:
                            logger.error(f"Pipeline '{pipeline_id}' failed: {str(e)}")
                            self._update_status(pipeline_id, "failed")
                            return {"error": str(e)}

                logger.warning("No pipeline available for routing")
                return {"error": "no_pipeline_available"}

            elif decision.decision_type == GroupDecisionType.PARALLEL_PIPELINES:
                # Parallel execution
                logger.info(f"Executing {len(decision.target_pipelines)} pipelines in parallel")
                tasks = []
                for pipeline_id in decision.target_pipelines:
                    pipeline = self._pipelines.get(pipeline_id)
                    if pipeline:
                        self._update_status(pipeline_id, "running")
                        tasks.append(pipeline.orchestrate(f"{run_id}_{pipeline_id}", input_data))

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Update statuses
                for i, pipeline_id in enumerate(decision.target_pipelines):
                    if isinstance(results[i], Exception):
                        logger.error(f"Pipeline '{pipeline_id}' failed in parallel execution: {str(results[i])}")
                        self._update_status(pipeline_id, "failed")
                    else:
                        self._update_status(pipeline_id, "completed")

                # Aggregate results
                return self._aggregate_results(
                    results,
                    decision.aggregation_strategy,
                )
            
            elif decision.decision_type == GroupDecisionType.ESCALATE:
                logger.info("Escalating decision")
                return {"escalated": True, "reason": decision.reason}

            return {}

        except Exception as e:
            logger.error(f"Group orchestration failed: {str(e)}")
            raise

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

    # === v2.3.0: Coordinator/Supervisor Role ===

    async def handle_escalation(
        self,
        escalation: Escalation,
        handler: Optional[Callable[[Escalation], EscalationDecision]] = None,
    ) -> EscalationDecision:
        """
        Handle an escalation from a pipeline.

        This is the primary entry point for the new Coordinator role.
        Escalations can come from pipelines when they encounter issues
        that require supervisor intervention.

        Args:
            escalation: The escalation to handle
            handler: Optional custom handler function. If not provided,
                     uses default handling based on escalation level.

        Returns:
            EscalationDecision indicating how to proceed

        Example:
            escalation = Escalation(
                escalation_id="esc-123",
                level=EscalationLevel.WARNING,
                source_pipeline="analysis_pipeline",
                reason="Model confidence below threshold",
                context={"confidence": 0.3},
            )
            decision = await group.handle_escalation(escalation)
        """
        logger.info(
            f"Handling escalation '{escalation.escalation_id}' "
            f"from '{escalation.source_pipeline}' (level={escalation.level.value})"
        )

        # Track escalation
        self._escalation_history.append(escalation)

        # Use custom handler if provided
        if handler:
            decision = handler(escalation)
            logger.debug(f"Custom handler decision: {decision.action.value}")
            return decision

        # Default handling based on level
        decision = self._default_escalation_handler(escalation)
        logger.debug(f"Default handler decision: {decision.action.value}")
        return decision

    def _default_escalation_handler(self, escalation: Escalation) -> EscalationDecision:
        """
        Default escalation handling based on severity level.

        Args:
            escalation: The escalation to handle

        Returns:
            EscalationDecision
        """
        if escalation.level == EscalationLevel.INFO:
            return EscalationDecision(
                action=EscalationAction.ACKNOWLEDGE,
                message=f"Acknowledged info escalation: {escalation.reason}",
            )

        elif escalation.level == EscalationLevel.WARNING:
            # Try to find an alternative pipeline
            available = [
                pid for pid, status in self._statuses.items()
                if status.status == "idle" and pid != escalation.source_pipeline
            ]
            if available:
                return EscalationDecision(
                    action=EscalationAction.REDIRECT,
                    target_pipeline=available[0],
                    message=f"Redirecting to {available[0]} due to: {escalation.reason}",
                )
            return EscalationDecision(
                action=EscalationAction.RETRY,
                message=f"Retrying due to warning: {escalation.reason}",
            )

        elif escalation.level == EscalationLevel.CRITICAL:
            return EscalationDecision(
                action=EscalationAction.HUMAN_REVIEW,
                message=f"Critical issue requires human review: {escalation.reason}",
                metadata={"escalation": escalation.to_dict()},
            )

        elif escalation.level == EscalationLevel.EMERGENCY:
            return EscalationDecision(
                action=EscalationAction.ABORT,
                message=f"Emergency abort: {escalation.reason}",
                metadata={"escalation": escalation.to_dict()},
            )

        return EscalationDecision(
            action=EscalationAction.ACKNOWLEDGE,
            message="Unknown escalation level, acknowledging",
        )

    def collect_metrics(self) -> Dict[str, Any]:
        """
        Collect comprehensive metrics for process mining and monitoring.

        This method aggregates metrics from all pipelines and includes
        escalation statistics for the Coordinator role.

        Returns:
            Dictionary with comprehensive metrics including:
            - Pipeline metrics (completion rates, error rates, bottlenecks)
            - Escalation statistics
            - Group health indicators

        Example:
            metrics = group.collect_metrics()
            print(f"Health score: {metrics['health_score']}")
            print(f"Escalations today: {metrics['escalations']['total']}")
        """
        # Get base group metrics
        base_metrics = self.get_group_metrics()

        # Calculate escalation metrics
        now = datetime.now()
        recent_escalations = [
            e for e in self._escalation_history
            if (now - e.timestamp).total_seconds() < 86400  # Last 24 hours
        ]

        escalation_by_level = {}
        for level in EscalationLevel:
            count = sum(1 for e in recent_escalations if e.level == level)
            escalation_by_level[level.value] = count

        escalation_by_pipeline = {}
        for e in recent_escalations:
            if e.source_pipeline not in escalation_by_pipeline:
                escalation_by_pipeline[e.source_pipeline] = 0
            escalation_by_pipeline[e.source_pipeline] += 1

        # Calculate health score (0-100)
        health_score = 100
        if base_metrics.get("avg_error_rate", 0) > 0:
            health_score -= min(30, base_metrics["avg_error_rate"] * 100)
        critical_escalations = escalation_by_level.get("critical", 0) + escalation_by_level.get("emergency", 0)
        health_score -= min(40, critical_escalations * 10)
        health_score = max(0, health_score)

        return {
            **base_metrics,
            "escalations": {
                "total": len(recent_escalations),
                "by_level": escalation_by_level,
                "by_pipeline": escalation_by_pipeline,
            },
            "health_score": health_score,
            "collected_at": now.isoformat(),
        }

    def get_escalation_history(
        self,
        pipeline_id: Optional[str] = None,
        level: Optional[EscalationLevel] = None,
        limit: int = 100,
    ) -> List[Escalation]:
        """
        Get escalation history with optional filtering.

        Args:
            pipeline_id: Filter by source pipeline
            level: Filter by escalation level
            limit: Maximum number of records to return

        Returns:
            List of Escalation records
        """
        history = self._escalation_history

        if pipeline_id:
            history = [e for e in history if e.source_pipeline == pipeline_id]

        if level:
            history = [e for e in history if e.level == level]

        return list(reversed(history[-limit:]))
