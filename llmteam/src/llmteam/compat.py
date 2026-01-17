"""
Compatibility module for deprecated names.

This module provides backward-compatible aliases with deprecation warnings
for classes that have been renamed in v2.0.0.

Renames in v2.0.0:
    - Pipeline -> LLMTeam
    - PipelineOrchestrator -> TeamOrchestrator
"""

import warnings
from typing import Any, Optional

from llmteam.roles.orchestration import OrchestrationStrategy
from llmteam.roles.pipeline_orch import PipelineOrchestrator


class LLMTeam(PipelineOrchestrator):
    """
    A team of AI agents working together.

    This is the recommended class name as of v2.0.0.
    Formerly known as PipelineOrchestrator/Pipeline.

    Example:
        team = LLMTeam(
            team_id="content_creation",
            strategy=RuleBasedStrategy(),
        )

        team.register_agent("writer", writer_agent)
        team.register_agent("editor", editor_agent)

        result = await team.orchestrate("run_123", input_data)
    """

    def __init__(
        self,
        team_id: str,
        strategy: Optional[OrchestrationStrategy] = None,
        enable_process_mining: bool = True,
    ):
        """
        Initialize LLMTeam.

        Args:
            team_id: Unique identifier for this team (maps to pipeline_id)
            strategy: Orchestration strategy (defaults to RuleBasedStrategy)
            enable_process_mining: Whether to enable process mining
        """
        super().__init__(
            pipeline_id=team_id,
            strategy=strategy,
            enable_process_mining=enable_process_mining,
        )

    @property
    def team_id(self) -> str:
        """Get the team ID (alias for pipeline_id)."""
        return self.pipeline_id


# Alias for semantic naming
TeamOrchestrator = LLMTeam


class Pipeline(LLMTeam):
    """
    Deprecated. Use LLMTeam instead.

    This class is provided for backward compatibility only.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        warnings.warn(
            "Pipeline is deprecated, use LLMTeam instead. "
            "Pipeline will be removed in v3.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Handle old 'pipeline_id' kwarg -> 'team_id'
        if "pipeline_id" in kwargs and "team_id" not in kwargs:
            kwargs["team_id"] = kwargs.pop("pipeline_id")
        super().__init__(*args, **kwargs)


def _create_deprecated_orchestrator(*args: Any, **kwargs: Any) -> PipelineOrchestrator:
    """
    Factory function that creates PipelineOrchestrator with deprecation warning.

    Use TeamOrchestrator or LLMTeam instead.
    """
    warnings.warn(
        "PipelineOrchestrator is deprecated, use TeamOrchestrator or LLMTeam instead. "
        "PipelineOrchestrator deprecation warnings will be enabled in v2.1.0.",
        PendingDeprecationWarning,
        stacklevel=2,
    )
    return PipelineOrchestrator(*args, **kwargs)
