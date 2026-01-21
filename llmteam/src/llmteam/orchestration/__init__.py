"""
Orchestration module.

RFC-004: GroupOrchestrator for multi-team coordination.

Contains:
    - GroupOrchestrator: Coordinates multiple LLMTeams
    - GroupRole: Roles for GroupOrchestrator (MVP: REPORT_COLLECTOR)
    - TeamReport, GroupReport, GroupResult: Report dataclasses

Legacy (deprecated in v4.0.0):
    - OrchestrationMode, OrchestrationStrategy, etc.

Use instead:
    - llmteam.team.LLMTeam: Main team container with agents
    - llmteam.orchestration.GroupOrchestrator: Multi-team coordination
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional

# RFC-004: GroupOrchestrator
from llmteam.orchestration.group import GroupOrchestrator, GroupRole
from llmteam.orchestration.reports import TeamReport, GroupReport, GroupResult


class OrchestrationMode(Enum):
    """Orchestration execution mode."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"


class OrchestrationStrategy:
    """Base orchestration strategy (deprecated)."""
    pass


class SequentialStrategy(OrchestrationStrategy):
    """Sequential execution strategy (deprecated)."""
    pass


class RoundRobinStrategy(OrchestrationStrategy):
    """Round-robin execution strategy (deprecated)."""
    pass


class LLMRoutingStrategy(OrchestrationStrategy):
    """LLM-based routing strategy (deprecated)."""
    pass


@dataclass
class OrchestrationContext:
    """Orchestration context (deprecated)."""
    run_id: str


@dataclass
class OrchestrationDecision:
    """Orchestration decision (deprecated)."""
    next_step: str


__all__ = [
    # RFC-004: GroupOrchestrator
    "GroupOrchestrator",
    "GroupRole",
    "TeamReport",
    "GroupReport",
    "GroupResult",
    # Legacy (deprecated)
    "OrchestrationMode",
    "OrchestrationStrategy",
    "SequentialStrategy",
    "RoundRobinStrategy",
    "LLMRoutingStrategy",
    "OrchestrationContext",
    "OrchestrationDecision",
]
