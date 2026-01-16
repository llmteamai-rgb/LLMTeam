"""
Orchestration roles for llmteam.

This module provides orchestration capabilities for pipeline and group management:
- Orchestration strategies (rule-based, LLM-based)
- Process mining for workflow analysis
- Pipeline orchestrators
- Group orchestrators

Quick Start:
    from llmteam.roles import (
        PipelineOrchestrator,
        RuleBasedStrategy,
        GroupOrchestrator,
    )

    # Create pipeline orchestrator
    orchestrator = PipelineOrchestrator(
        pipeline_id="my_pipeline",
        strategy=RuleBasedStrategy(),
        enable_process_mining=True,
    )

    # Execute
    result = await orchestrator.orchestrate("run_1", input_data)

    # Get metrics
    metrics = orchestrator.get_process_metrics()
"""

from llmteam.roles.orchestration import (
    OrchestratorRole,
    OrchestrationDecision,
    OrchestrationContext,
    OrchestrationStrategy,
    RuleBasedStrategy,
    LLMBasedStrategy,
)

from llmteam.roles.process_mining import (
    ProcessEvent,
    ProcessMetrics,
    ProcessModel,
    ProcessMiningEngine,
)

from llmteam.roles.pipeline_orch import (
    PipelineOrchestrator,
)

from llmteam.roles.group_orch import (
    GroupDecisionType,
    GroupOrchestrationDecision,
    PipelineStatus,
    GroupOrchestrationStrategy,
    LoadBalancingStrategy,
    ContentBasedRoutingStrategy,
    ParallelFanOutStrategy,
    GroupOrchestrator,
)

__all__ = [
    # Orchestration
    "OrchestratorRole",
    "OrchestrationDecision",
    "OrchestrationContext",
    "OrchestrationStrategy",
    "RuleBasedStrategy",
    "LLMBasedStrategy",

    # Process Mining
    "ProcessEvent",
    "ProcessMetrics",
    "ProcessModel",
    "ProcessMiningEngine",

    # Pipeline Orchestrator
    "PipelineOrchestrator",

    # Group Orchestrator
    "GroupDecisionType",
    "GroupOrchestrationDecision",
    "PipelineStatus",
    "GroupOrchestrationStrategy",
    "LoadBalancingStrategy",
    "ContentBasedRoutingStrategy",
    "ParallelFanOutStrategy",
    "GroupOrchestrator",
]
