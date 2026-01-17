"""
Canvas module exceptions.
"""


class CanvasError(Exception):
    """Base exception for canvas module."""

    pass


class SegmentValidationError(CanvasError):
    """Raised when segment validation fails."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Segment validation failed: {', '.join(errors)}")


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
