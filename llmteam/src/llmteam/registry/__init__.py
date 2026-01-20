"""
Registry module for LLMTeam.

Provides generic registries for managing agents and teams.
"""

from llmteam.registry.base import (
    BaseRegistry,
    RegistryError,
    NotFoundError,
    AlreadyExistsError,
)
from llmteam.registry.agent_registry import AgentRegistry
from llmteam.registry.team_registry import TeamRegistry

__all__ = [
    # Base
    "BaseRegistry",
    "RegistryError",
    "NotFoundError",
    "AlreadyExistsError",
    # Specialized
    "AgentRegistry",
    "TeamRegistry",
]
