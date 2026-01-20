"""
Compatibility module for deprecated names.

DEPRECATED: This module will be removed in v5.0.0.
Use llmteam.team.LLMTeam directly.

Migration guide:
    # Old (deprecated)
    from llmteam.compat import LLMTeam

    # New (recommended)
    from llmteam import LLMTeam
"""

import warnings
from typing import Any, Optional

from llmteam.team import LLMTeam as _LLMTeam
from llmteam.team import LLMGroup as _LLMGroup


class LLMTeam(_LLMTeam):
    """
    Deprecated. Use llmteam.team.LLMTeam instead.

    This wrapper exists only for backwards compatibility and will be
    removed in v5.0.0.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        warnings.warn(
            "Importing LLMTeam from llmteam.compat is deprecated. "
            "Use 'from llmteam import LLMTeam' instead. "
            "This compatibility layer will be removed in v5.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class LLMGroup(_LLMGroup):
    """
    Deprecated. Use llmteam.team.LLMGroup instead.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        warnings.warn(
            "Importing LLMGroup from llmteam.compat is deprecated. "
            "Use 'from llmteam import LLMGroup' instead. "
            "This compatibility layer will be removed in v5.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


# Legacy aliases (all deprecated)
TeamOrchestrator = LLMTeam
Pipeline = LLMTeam
PipelineOrchestrator = LLMTeam


# Stub for old OrchestrationStrategy
class OrchestrationStrategy:
    """
    Deprecated. Orchestration is now handled via the 'flow' parameter.

    In v4.0.0, orchestration strategies are replaced by:
    - flow="sequential" (default)
    - flow="a -> b -> c" (string syntax)
    - flow={"edges": [...]} (DAG)
    - orchestration=True (adds orchestrator agent)
    """

    def __init__(self):
        warnings.warn(
            "OrchestrationStrategy is deprecated. "
            "Use LLMTeam(flow=...) parameter instead.",
            DeprecationWarning,
            stacklevel=2,
        )


__all__ = [
    "LLMTeam",
    "LLMGroup",
    "TeamOrchestrator",
    "Pipeline",
    "PipelineOrchestrator",
    "OrchestrationStrategy",
]
