"""
Engine module exceptions.

This module provides exceptions for the ExecutionEngine (workflow runtime).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from llmteam.engine.validation import ValidationResult


class EngineError(Exception):
    """Base exception for engine module."""

    pass


class WorkflowValidationError(EngineError):
    """Raised when workflow validation fails."""

    def __init__(
        self,
        errors: Union[list[str], str],
        result: "ValidationResult | None" = None,
    ) -> None:
        if isinstance(errors, str):
            self.errors = [errors]
            message = errors
        else:
            self.errors = errors
            message = f"Workflow validation failed: {', '.join(errors)}"

        self.result = result
        super().__init__(message)


class StepTypeNotFoundError(EngineError):
    """Raised when step type is not found in catalog."""

    def __init__(self, type_id: str) -> None:
        self.type_id = type_id
        super().__init__(f"Step type '{type_id}' not found in catalog")


class InvalidStepConfigError(EngineError):
    """Raised when step configuration is invalid."""

    def __init__(self, type_id: str, errors: list[str]) -> None:
        self.type_id = type_id
        self.errors = errors
        super().__init__(f"Invalid config for step type '{type_id}': {', '.join(errors)}")


class InvalidConditionError(EngineError):
    """Raised when edge condition cannot be evaluated."""

    def __init__(self, condition: str, reason: str) -> None:
        self.condition = condition
        self.reason = reason
        super().__init__(f"Invalid condition '{condition}': {reason}")


# =============================================================================
# Backward compatibility aliases (deprecated)
# =============================================================================

def _deprecated_alias(old_name: str, new_class: type) -> type:
    """Create a deprecated alias that warns on use."""
    class DeprecatedAlias(new_class):
        def __init__(self, *args, **kwargs):
            warnings.warn(
                f"{old_name} is deprecated, use {new_class.__name__} instead",
                DeprecationWarning,
                stacklevel=2,
            )
            super().__init__(*args, **kwargs)
    DeprecatedAlias.__name__ = old_name
    DeprecatedAlias.__qualname__ = old_name
    return DeprecatedAlias


# Deprecated aliases
CanvasError = EngineError  # Direct alias (base exception)
SegmentValidationError = WorkflowValidationError  # Direct alias
