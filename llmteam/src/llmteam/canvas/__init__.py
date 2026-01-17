"""
Canvas Integration Module.

This module provides the JSON contract and APIs for integrating
with KorpOS Worktrail Canvas.

Main components:
- SegmentDefinition: JSON contract for workflow segments
- StepCatalog: Registry of available step types
- SegmentRunner: Execution engine for segments
- Models for steps, edges, and ports
"""

from llmteam.canvas.models import (
    PortDefinition,
    StepPosition,
    StepUIMetadata,
    StepDefinition,
    EdgeDefinition,
    SegmentParams,
    SegmentDefinition,
)

from llmteam.canvas.catalog import (
    StepCategory,
    PortSpec,
    StepTypeMetadata,
    StepCatalog,
)

from llmteam.canvas.runner import (
    SegmentStatus,
    SegmentResult,
    RunConfig,
    SegmentRunner,
    SegmentSnapshot,
    SegmentSnapshotStore,
)

from llmteam.canvas.handlers import (
    LLMAgentHandler,
    HTTPActionHandler,
    TransformHandler,
    ConditionHandler,
    ParallelSplitHandler,
    ParallelJoinHandler,
    HumanTaskHandler,
    create_human_task_handler,
)

from llmteam.canvas.exceptions import (
    CanvasError,
    SegmentValidationError,
    StepTypeNotFoundError,
    InvalidStepConfigError,
    InvalidConditionError,
)

from llmteam.canvas.validation import (
    ValidationSeverity,
    ValidationMessage,
    ValidationResult,
    SegmentValidator,
    validate_segment,
    validate_segment_dict,
)

__all__ = [
    # Models
    "PortDefinition",
    "StepPosition",
    "StepUIMetadata",
    "StepDefinition",
    "EdgeDefinition",
    "SegmentParams",
    "SegmentDefinition",
    # Catalog
    "StepCategory",
    "PortSpec",
    "StepTypeMetadata",
    "StepCatalog",
    # Runner
    "SegmentStatus",
    "SegmentResult",
    "RunConfig",
    "SegmentRunner",
    "SegmentSnapshot",
    "SegmentSnapshotStore",
    # Handlers
    "LLMAgentHandler",
    "HTTPActionHandler",
    "TransformHandler",
    "ConditionHandler",
    "ParallelSplitHandler",
    "ParallelJoinHandler",
    "HumanTaskHandler",
    "create_human_task_handler",
    # Exceptions
    "CanvasError",
    "SegmentValidationError",
    "StepTypeNotFoundError",
    "InvalidStepConfigError",
    "InvalidConditionError",
    # Validation
    "ValidationSeverity",
    "ValidationMessage",
    "ValidationResult",
    "SegmentValidator",
    "validate_segment",
    "validate_segment_dict",
]
