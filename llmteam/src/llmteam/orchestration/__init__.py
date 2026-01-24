"""
Orchestration module.

RFC-004: GroupOrchestrator for multi-team coordination.
RFC-009: Group Architecture Unification with bi-directional context.

Contains:
    - GroupOrchestrator: Coordinates multiple LLMTeams
    - GroupRole: Roles for GroupOrchestrator (REPORT_COLLECTOR, COORDINATOR, ROUTER, AGGREGATOR, ARBITER)
    - TeamRole: Roles for teams within a group (LEADER, MEMBER, SPECIALIST, FALLBACK)
    - GroupContext: Bi-directional context passed to teams
    - EscalationRequest, EscalationResponse: Escalation handling
    - TeamReport, GroupReport, GroupResult: Report dataclasses
"""

# RFC-004 + RFC-009: GroupOrchestrator
from llmteam.orchestration.group import GroupOrchestrator, GroupRole
from llmteam.orchestration.reports import TeamReport, GroupReport, GroupResult

# RFC-009: New models
from llmteam.orchestration.models import (
    TeamRole,
    GroupContext,
    EscalationRequest,
    EscalationResponse,
    GroupEscalationAction,
)

__all__ = [
    # RFC-004 + RFC-009: GroupOrchestrator
    "GroupOrchestrator",
    "GroupRole",
    # RFC-009: Team roles and context
    "TeamRole",
    "GroupContext",
    # RFC-009: Group escalation
    "EscalationRequest",
    "EscalationResponse",
    "GroupEscalationAction",
    # Reports
    "TeamReport",
    "GroupReport",
    "GroupResult",
]
