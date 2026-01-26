"""
LLMTeam Routing - Adaptive routing for hybrid workflows.

RFC-022: Provides AdaptiveStep for rule-based routing with LLM fallback.
"""

from llmteam.routing.adaptive import (
    # Configuration
    AdaptiveStepConfig,
    RoutingRule,
    LLMFallbackConfig,
    RouteOption,
    # Results
    RoutingDecision,
    RoutingMethod,
    # Events
    AdaptiveDecisionEvent,
    # Checkpoint
    CheckpointConfig,
)

__all__ = [
    # Configuration
    "AdaptiveStepConfig",
    "RoutingRule",
    "LLMFallbackConfig",
    "RouteOption",
    # Results
    "RoutingDecision",
    "RoutingMethod",
    # Events
    "AdaptiveDecisionEvent",
    # Checkpoint
    "CheckpointConfig",
]
