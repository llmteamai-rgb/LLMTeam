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
    Team configuration (deprecated).

    Use LLMTeam constructor arguments directly instead.
    """

    strict_validation: bool = True
    timeout: int = 30


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
