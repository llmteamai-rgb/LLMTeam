"""
Builder module for LLMTeam.

RFC-021: DynamicTeamBuilder — Automatic team creation from task descriptions.

Usage:
    from llmteam.builder import DynamicTeamBuilder, TeamBlueprint, AgentBlueprint

    builder = DynamicTeamBuilder(model="gpt-4o-mini")
    blueprint = await builder.analyze_task("Research AI trends and summarize")
    team = builder.build_team(blueprint)
    await builder.execute(team, {"query": "LLM breakthroughs"})
"""

from llmteam.builder.dynamic import (
    DynamicTeamBuilder,
    TeamBlueprint,
    AgentBlueprint,
    TOOL_MAP,
    BuilderError,
    BuilderParseError,
    BuilderValidationError,
)

__all__ = [
    "DynamicTeamBuilder",
    "TeamBlueprint",
    "AgentBlueprint",
    "TOOL_MAP",
    "BuilderError",
    "BuilderParseError",
    "BuilderValidationError",
]
