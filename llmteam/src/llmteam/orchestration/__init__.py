"""
Orchestration module.

DEPRECATED: In v4.0.0, orchestration is handled internally by LLMTeam.

Use instead:
    - llmteam.team.LLMTeam: Main team container with agents
    - llmteam.team.LLMGroup: Multi-team coordination
    - llmteam.agents.create_orchestrator_config: Create orchestrator agent

Orchestration is now just an LLMAgent with a specialized prompt.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


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
    "OrchestrationMode",
    "OrchestrationStrategy",
    "SequentialStrategy",
    "RoundRobinStrategy",
    "LLMRoutingStrategy",
    "OrchestrationContext",
    "OrchestrationDecision",
]
