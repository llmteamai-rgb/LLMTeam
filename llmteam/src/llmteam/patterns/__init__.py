"""
Patterns Module.

This module provides reusable patterns for multi-agent workflows:
- CriticLoop: Generator-Critic feedback loop for recursive improvement
"""

from llmteam.patterns.critic_loop import (
    CriticVerdict,
    CriticLoopConfig,
    CriticFeedback,
    IterationRecord,
    CriticLoopResult,
    CriticLoop,
)

__all__ = [
    "CriticVerdict",
    "CriticLoopConfig",
    "CriticFeedback",
    "IterationRecord",
    "CriticLoopResult",
    "CriticLoop",
]
