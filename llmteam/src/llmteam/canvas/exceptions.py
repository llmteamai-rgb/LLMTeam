"""
Canvas module exceptions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from llmteam.canvas.validation import ValidationResult


class CanvasError(Exception):
    """Base exception for canvas module."""

    pass


class SegmentValidationError(CanvasError):
    """Raised when segment validation fails."""

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
            message = f"Segment validation failed: {', '.join(errors)}"

        self.result = result
        super().__init__(message)


class StepTypeNotFoundError(CanvasError):
    """Raised when step type is not found in catalog."""

    def __init__(self, type_id: str) -> None:
        self.type_id = type_id
        super().__init__(f"Step type '{type_id}' not found in catalog")


class InvalidStepConfigError(CanvasError):
    """Raised when step configuration is invalid."""

    def __init__(self, type_id: str, errors: list[str]) -> None:
        self.type_id = type_id
        self.errors = errors
        super().__init__(f"Invalid config for step type '{type_id}': {', '.join(errors)}")


class InvalidConditionError(CanvasError):
    """Raised when edge condition cannot be evaluated."""

    def __init__(self, condition: str, reason: str) -> None:
        self.condition = condition
        self.reason = reason
        super().__init__(f"Invalid condition '{condition}': {reason}")
