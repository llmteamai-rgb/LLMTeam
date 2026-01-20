"""
Team package.

Provides LLMTeam and LLMGroup for orchestrating AI agents.
"""

from llmteam.team.result import (
    RunResult,
    RunStatus,
    ContextMode,
    TeamResult,  # Backwards compatibility alias
)
from llmteam.team.snapshot import TeamSnapshot
from llmteam.team.team import LLMTeam
from llmteam.team.group import LLMGroup

# Backwards compatibility
TeamConfig = None  # Will be removed, use LLMTeam constructor args directly

__all__ = [
    # Main classes
    "LLMTeam",
    "LLMGroup",
    # Result types
    "RunResult",
    "RunStatus",
    "ContextMode",
    "TeamResult",  # Alias for RunResult
    # Snapshot
    "TeamSnapshot",
]
