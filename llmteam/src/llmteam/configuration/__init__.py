"""
Configuration module for CONFIGURATOR mode (RFC-005, RFC-023).

Provides interactive team configuration via LLM assistance.

RFC-023: Added Configurator class and decision point handling.

Usage:
    from llmteam import LLMTeam

    # L1: One-call solve
    result = await LLMTeam.solve(
        task="Write an article about AI",
        quality=70,
        routing_mode="hybrid",
    )

    # L1: Interactive session
    session = await LLMTeam.start(
        task="Create a marketing campaign",
        quality=70,
    )
    print(session.question)
    await session.answer("Fitness app for millennials")
    result = await session.execute()

    # L2: Configure and modify
    team = await LLMTeam.create_configured(task="Write article")
    team.remove_agent("editor")
    result = await team.run({"topic": "AI trends"})
"""

from llmteam.configuration.models import (
    # Session state
    SessionState,
    # Suggestions
    AgentSuggestion,
    TestRunResult,
    TaskAnalysis,
    PipelinePreview,
    # RFC-023: Decision points
    DecisionPointAnalysis,
    DecisionPointConfig,
    RoutingRuleConfig,
    RouteConfig,
    LLMFallbackConfigData,
    ConfiguratorCostEstimate,
    ConfiguratorOutput,
)

from llmteam.configuration.prompts import ConfiguratorPrompts

from llmteam.configuration.session import ConfigurationSession

# RFC-023: Standalone Configurator
from llmteam.configuration.configurator import Configurator

__all__ = [
    # Models
    "SessionState",
    "AgentSuggestion",
    "TestRunResult",
    "TaskAnalysis",
    "PipelinePreview",
    # RFC-023: Decision points
    "DecisionPointAnalysis",
    "DecisionPointConfig",
    "RoutingRuleConfig",
    "RouteConfig",
    "LLMFallbackConfigData",
    "ConfiguratorCostEstimate",
    "ConfiguratorOutput",
    # Prompts
    "ConfiguratorPrompts",
    # Session
    "ConfigurationSession",
    # RFC-023: Configurator
    "Configurator",
]
