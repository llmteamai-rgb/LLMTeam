"""
Configuration module for CONFIGURATOR mode (RFC-005).

Provides interactive team configuration via LLM assistance.

Usage:
    from llmteam import LLMTeam

    team = LLMTeam(team_id="content")

    # Start configuration session
    session = await team.configure(
        task="Generate LinkedIn posts from press releases",
        constraints={"tone": "professional", "length": "<300"}
    )

    # Review suggestions
    print(session.suggested_agents)
    print(session.suggested_flow)

    # Modify if needed
    session.modify_agent("writer", prompt="Write in first person...")

    # Test
    test = await session.test_run({"press_release": "..."})
    print(test.analysis)
    print(test.recommendations)

    # Apply when ready
    await session.apply()

    # Use the configured team
    result = await team.run({"press_release": "..."})
"""

from llmteam.configuration.models import (
    SessionState,
    AgentSuggestion,
    TestRunResult,
    TaskAnalysis,
    PipelinePreview,
)

from llmteam.configuration.prompts import ConfiguratorPrompts

from llmteam.configuration.session import ConfigurationSession

__all__ = [
    # Models
    "SessionState",
    "AgentSuggestion",
    "TestRunResult",
    "TaskAnalysis",
    "PipelinePreview",
    # Prompts
    "ConfiguratorPrompts",
    # Session
    "ConfigurationSession",
]
