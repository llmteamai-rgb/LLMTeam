"""
LLMTeam Container.

Re-exports from llmteam.team package for backwards compatibility.
"""

# Re-export everything from the team package
from llmteam.team.team import LLMTeam
from llmteam.team.group import LLMGroup
from llmteam.team.result import (
    RunResult,
    RunStatus,
    ContextMode,
    TeamResult,
)
from llmteam.team.snapshot import TeamSnapshot

# Backwards compatibility - TeamConfig is now just constructor args
from dataclasses import dataclass
from typing import Optional


@dataclass
class TeamConfig:
    """
    Team configuration.

    Can be passed to LLMTeam constructor for advanced settings.

    Args:
        strict_validation: Validate agent configs strictly
        timeout: Default execution timeout in seconds
        enforce_lifecycle: Enable lifecycle state enforcement (RFC-014).
                          When True, team must be configured before running.
    """

    strict_validation: bool = True
    timeout: int = 30
    enforce_lifecycle: bool = False


__all__ = [
    "LLMTeam",
    "LLMGroup",
    "RunResult",
    "RunStatus",
    "ContextMode",
    "TeamResult",
    "TeamSnapshot",
    "TeamConfig",
]
