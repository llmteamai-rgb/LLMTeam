"""
Escalation handlers for LLMTeam.

Provides built-in handlers for common escalation patterns.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Any

from llmteam.escalation.models import (
    Escalation,
    EscalationAction,
    EscalationDecision,
    EscalationLevel,
)


class EscalationHandler(ABC):
    """
    Abstract base class for escalation handlers.

    Implement this to create custom escalation handling logic.
    """

    @abstractmethod
    def handle(self, escalation: Escalation) -> EscalationDecision:
        """
        Handle an escalation.

        Args:
            escalation: The escalation to handle.

        Returns:
            Decision on what action to take.
        """
        ...

    def can_handle(self, escalation: Escalation) -> bool:
        """
        Check if this handler can handle the escalation.

        Override to implement filtering logic.

        Args:
            escalation: The escalation to check.

        Returns:
            True if this handler can handle it.
        """
        return True


class DefaultHandler(EscalationHandler):
    """
    Default escalation handler based on level.

    Maps escalation levels to default actions.
    """

    def __init__(
        self,
        level_actions: Optional[Dict[EscalationLevel, EscalationAction]] = None,
    ):
        """
        Initialize with level-to-action mapping.

        Args:
            level_actions: Custom mapping of levels to actions.
        """
        self._level_actions = level_actions or {
            EscalationLevel.INFO: EscalationAction.ACKNOWLEDGE,
            EscalationLevel.WARNING: EscalationAction.RETRY,
            EscalationLevel.CRITICAL: EscalationAction.HUMAN_REVIEW,
            EscalationLevel.EMERGENCY: EscalationAction.ABORT,
        }

    def handle(self, escalation: Escalation) -> EscalationDecision:
        """Handle based on level."""
        action = self._level_actions.get(
            escalation.level,
            EscalationAction.ACKNOWLEDGE,
        )

        messages = {
            EscalationAction.ACKNOWLEDGE: "Acknowledged",
            EscalationAction.RETRY: "Will retry operation",
            EscalationAction.REDIRECT: "Redirecting to alternative",
            EscalationAction.HUMAN_REVIEW: "Escalated for human review",
            EscalationAction.ABORT: "Operation aborted",
        }

        return EscalationDecision(
            action=action,
            message=messages.get(action, ""),
        )


class ThresholdHandler(EscalationHandler):
    """
    Handler that takes action based on thresholds.

    Useful for rate limiting or count-based escalation.
    """

    def __init__(
        self,
        threshold: int = 3,
        window_seconds: float = 300.0,
        threshold_action: EscalationAction = EscalationAction.HUMAN_REVIEW,
        under_threshold_action: EscalationAction = EscalationAction.ACKNOWLEDGE,
    ):
        """
        Initialize threshold handler.

        Args:
            threshold: Number of escalations before triggering.
            window_seconds: Time window for counting.
            threshold_action: Action when threshold exceeded.
            under_threshold_action: Action when under threshold.
        """
        self._threshold = threshold
        self._window_seconds = window_seconds
        self._threshold_action = threshold_action
        self._under_threshold_action = under_threshold_action
        self._counts: Dict[str, int] = {}

    def handle(self, escalation: Escalation) -> EscalationDecision:
        """Handle based on count threshold."""
        key = f"{escalation.source_team}:{escalation.level.value}"
        self._counts[key] = self._counts.get(key, 0) + 1

        if self._counts[key] >= self._threshold:
            return EscalationDecision(
                action=self._threshold_action,
                message=f"Threshold of {self._threshold} exceeded",
                metadata={"count": self._counts[key]},
            )

        return EscalationDecision(
            action=self._under_threshold_action,
            message=f"Count: {self._counts[key]}/{self._threshold}",
        )

    def reset(self, key: Optional[str] = None) -> None:
        """Reset counts."""
        if key:
            self._counts.pop(key, None)
        else:
            self._counts.clear()


class FunctionHandler(EscalationHandler):
    """
    Handler that wraps a callable function.

    Useful for quick custom handlers.
    """

    def __init__(
        self,
        fn: Callable[[Escalation], EscalationDecision],
        filter_fn: Optional[Callable[[Escalation], bool]] = None,
    ):
        """
        Initialize with function.

        Args:
            fn: Function to call for handling.
            filter_fn: Optional function to filter escalations.
        """
        self._fn = fn
        self._filter_fn = filter_fn

    def handle(self, escalation: Escalation) -> EscalationDecision:
        """Handle by calling function."""
        return self._fn(escalation)

    def can_handle(self, escalation: Escalation) -> bool:
        """Check using filter function."""
        if self._filter_fn:
            return self._filter_fn(escalation)
        return True


class ChainHandler(EscalationHandler):
    """
    Handler that chains multiple handlers.

    Tries each handler in order until one succeeds.
    """

    def __init__(self, handlers: List[EscalationHandler]):
        """
        Initialize with list of handlers.

        Args:
            handlers: Handlers to chain.
        """
        self._handlers = handlers

    def handle(self, escalation: Escalation) -> EscalationDecision:
        """Try each handler in order."""
        for handler in self._handlers:
            if handler.can_handle(escalation):
                return handler.handle(escalation)

        # Fallback
        return EscalationDecision(
            action=EscalationAction.ACKNOWLEDGE,
            message="No handler matched, acknowledging",
        )

    def add_handler(self, handler: EscalationHandler) -> None:
        """Add a handler to the chain."""
        self._handlers.append(handler)


class LevelFilterHandler(EscalationHandler):
    """
    Handler that only handles specific levels.
    """

    def __init__(
        self,
        levels: List[EscalationLevel],
        handler: EscalationHandler,
    ):
        """
        Initialize with levels and wrapped handler.

        Args:
            levels: Levels to handle.
            handler: Handler to delegate to.
        """
        self._levels = set(levels)
        self._handler = handler

    def handle(self, escalation: Escalation) -> EscalationDecision:
        """Handle by delegating to wrapped handler."""
        return self._handler.handle(escalation)

    def can_handle(self, escalation: Escalation) -> bool:
        """Check if level matches."""
        return escalation.level in self._levels
