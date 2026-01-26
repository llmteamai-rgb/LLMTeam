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

# RFC-022: Interactive session
from llmteam.team.interactive import (
    InteractiveSession,
    SessionState as InteractiveSessionState,
    Question as InteractiveQuestion,
    TeamProposal,
)

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
    # RFC-022: Interactive session
    "InteractiveSession",
    "InteractiveSessionState",
    "InteractiveQuestion",
    "TeamProposal",
]
