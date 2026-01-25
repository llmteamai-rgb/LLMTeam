"""
API Routes for LLMTeam.

RFC-020: Teams API Extension
RFC-021: Quality Simplification (quality only in ConfigurationSession)

Organized by resource:
- teams: Team CRUD and execution
- agents: Agent management within teams
- sessions: Configuration sessions (CONFIGURATOR mode)
- groups: Multi-team group orchestration
"""

from llmteam.api.routes.teams import router as teams_router
from llmteam.api.routes.agents import router as agents_router
from llmteam.api.routes.sessions import router as sessions_router
from llmteam.api.routes.groups import router as groups_router

__all__ = [
    "teams_router",
    "agents_router",
    "sessions_router",
    "groups_router",
]
