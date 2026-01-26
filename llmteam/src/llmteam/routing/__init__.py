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
    AdaptiveRoutingDecision,
    RoutingMethod,
    # Events
    AdaptiveDecisionEvent,
    # Checkpoint
    CheckpointConfig,
)

# Backward compatibility alias (deprecated)
RoutingDecision = AdaptiveRoutingDecision

__all__ = [
    # Configuration
    "AdaptiveStepConfig",
    "RoutingRule",
    "LLMFallbackConfig",
    "RouteOption",
    # Results
    "AdaptiveRoutingDecision",
    "RoutingDecision",  # Deprecated alias
    "RoutingMethod",
    # Events
    "AdaptiveDecisionEvent",
    # Checkpoint
    "CheckpointConfig",
]
