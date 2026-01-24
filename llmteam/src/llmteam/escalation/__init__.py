"""
Escalation module for LLMTeam.

Provides escalation levels, actions, and handlers for managing
error recovery and human-in-the-loop workflows.

Usage:
    from llmteam.escalation import (
        EscalationLevel,
        EscalationAction,
        Escalation,
        EscalationDecision,
        EscalationRecord,
        EscalationHandler,
        DefaultHandler,
        ThresholdHandler,
        ChainHandler,
    )
"""

# Models
from llmteam.escalation.models import (
    EscalationLevel,
    EscalationAction,
    Escalation,
    EscalationDecision,
    EscalationRecord,
)

# Handlers
from llmteam.escalation.handlers import (
    EscalationHandler,
    DefaultHandler,
    ThresholdHandler,
    FunctionHandler,
    ChainHandler,
    LevelFilterHandler,
)

__all__ = [
    # Models
    "EscalationLevel",
    "EscalationAction",
    "Escalation",
    "EscalationDecision",
    "EscalationRecord",
    # Handlers
    "EscalationHandler",
    "DefaultHandler",
    "ThresholdHandler",
    "FunctionHandler",
    "ChainHandler",
    "LevelFilterHandler",
]
