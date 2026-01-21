"""
Engine Module - Workflow Execution Runtime.

This module provides the execution engine for workflows.
Formerly known as the Canvas module.

Main components:
- WorkflowDefinition: JSON contract for workflows (formerly SegmentDefinition)
- StepCatalog: Registry of available step types
- ExecutionEngine: Execution engine for workflows (formerly SegmentRunner)
- Models for steps, edges, and ports
"""

from llmteam.engine.models import (
    PortDefinition,
    StepPosition,
    StepUIMetadata,
    StepDefinition,
    EdgeDefinition,
    # New names
    WorkflowParams,
    WorkflowDefinition,
    # Backward compatibility aliases
    SegmentParams,
    SegmentDefinition,
)

from llmteam.engine.catalog import (
    StepCategory,
    PortSpec,
    StepTypeMetadata,
    StepCatalog,
)

from llmteam.engine.engine import (
    # New names
    ExecutionStatus,
    ExecutionResult,
    ExecutionEngine,
    ExecutionSnapshot,
    ExecutionSnapshotStore,
    RunConfig,
    # Backward compatibility aliases
    SegmentStatus,
    SegmentResult,
    SegmentRunner,
    SegmentSnapshot,
    SegmentSnapshotStore,
)

from llmteam.engine.handlers import (
    LLMAgentHandler,
    HTTPActionHandler,
    TransformHandler,
    ConditionHandler,
    ParallelSplitHandler,
    ParallelJoinHandler,
    HumanTaskHandler,
    create_human_task_handler,
)

from llmteam.engine.exceptions import (
    # New names
    EngineError,
    WorkflowValidationError,
    # Backward compatibility aliases
    CanvasError,
    SegmentValidationError,
    # Other exceptions
    StepTypeNotFoundError,
    InvalidStepConfigError,
    InvalidConditionError,
)

from llmteam.engine.validation import (
    ValidationSeverity,
    ValidationMessage,
    ValidationResult,
    SegmentValidator,
    validate_segment,
    validate_segment_dict,
)

__all__ = [
    # Models - New names
    "PortDefinition",
    "StepPosition",
    "StepUIMetadata",
    "StepDefinition",
    "EdgeDefinition",
    "WorkflowParams",
    "WorkflowDefinition",
    # Models - Backward compatibility
    "SegmentParams",
    "SegmentDefinition",
    # Catalog
    "StepCategory",
    "PortSpec",
    "StepTypeMetadata",
    "StepCatalog",
    # Engine - New names
    "ExecutionStatus",
    "ExecutionResult",
    "ExecutionEngine",
    "ExecutionSnapshot",
    "ExecutionSnapshotStore",
    "RunConfig",
    # Engine - Backward compatibility
    "SegmentStatus",
    "SegmentResult",
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
    # Exceptions - New names
    "EngineError",
    "WorkflowValidationError",
    # Exceptions - Backward compatibility
    "CanvasError",
    "SegmentValidationError",
    # Exceptions - Other
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
