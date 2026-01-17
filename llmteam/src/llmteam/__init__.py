"""
llmteam - Enterprise AI Workflow Runtime

A library for building multi-agent LLM pipelines with enterprise-grade
security, orchestration, and workflow capabilities.

Version: 2.0.4 (Middleware + Observability + New Handlers)

License Tiers:
    - COMMUNITY (free): Basic features, memory stores
    - PROFESSIONAL ($99/mo): Process mining, PostgreSQL, Human-in-the-loop
    - ENTERPRISE (custom): Multi-tenant, Audit trail, SSO

Quick Start:
    import llmteam

    # Check tier
    print(llmteam.get_tier())  # LicenseTier.COMMUNITY

    # Activate license
    llmteam.activate("LLMT-PRO-XXXX-20261231")

Documentation: https://docs.llmteam.ai
"""

__version__ = "2.0.4"
__author__ = "llmteam contributors"
__email__ = "LLMTeamai@gmail.com"

# Core exports
from llmteam.tenancy import (
    TenantConfig,
    TenantContext,
    TenantManager,
    TenantTier,
    TenantLimits,
    current_tenant,
)

from llmteam.audit import (
    AuditTrail,
    AuditRecord,
    AuditQuery,
    AuditEventType,
    AuditSeverity,
)

from llmteam.context import (
    SecureAgentContext,
    ContextAccessPolicy,
    SealedData,
    VisibilityLevel,
    SensitivityLevel,
    # v1.8.0: Hierarchical Context
    ContextScope,
    HierarchicalContext,
    ContextManager,
    ContextPropagationConfig,
)

from llmteam.ratelimit import (
    RateLimiter,
    RateLimitConfig,
    RateLimitStrategy,
    CircuitBreakerConfig,
    CircuitState,
    RateLimitedExecutor,
)

# v1.8.0 + v2.0.0: Licensing (Open Core)
from llmteam.licensing import (
    LicenseTier,
    LicenseLimits,
    License,
    LicenseManager,
    # Activation functions
    activate,
    get_tier,
    has_feature,
    print_license_status,
    get_license_manager,
    # Exceptions
    LicenseValidationError,
    LicenseExpiredError,
    FeatureNotLicensedError,
    # Decorators
    professional_only,
    enterprise_only,
)

# v1.8.0: Execution
from llmteam.execution import (
    ExecutionMode,
    ExecutorConfig,
    TaskResult,
    ExecutionStats,
    PipelineExecutor,
)

# v1.8.0: Orchestration Roles
from llmteam.roles import (
    # Orchestration
    OrchestratorRole,
    OrchestrationDecision,
    OrchestrationContext,
    OrchestrationStrategy,
    RuleBasedStrategy,
    LLMBasedStrategy,
    # Process Mining
    ProcessEvent,
    ProcessMetrics,
    ProcessModel,
    ProcessMiningEngine,
    # Pipeline Orchestrator
    PipelineOrchestrator,
    # Group Orchestrator
    GroupDecisionType,
    GroupOrchestrationDecision,
    PipelineStatus,
    GroupOrchestrationStrategy,
    LoadBalancingStrategy,
    ContentBasedRoutingStrategy,
    ParallelFanOutStrategy,
    GroupOrchestrator,
)

# v1.9.0: External Actions
from llmteam.actions import (
    ActionType,
    ActionStatus,
    ActionConfig,
    ActionContext,
    ActionResult,
    ActionRegistry,
    ActionExecutor,
)

# v1.9.0: Human Interaction
from llmteam.human import (
    InteractionType,
    InteractionStatus,
    InteractionPriority,
    InteractionRequest,
    InteractionResponse,
    HumanInteractionManager,
    MemoryInteractionStore,
)

# v1.9.0: Persistence
from llmteam.persistence import (
    SnapshotType,
    PipelinePhase,
    AgentSnapshot,
    PipelineSnapshot,
    RestoreResult,
    SnapshotManager,
    MemorySnapshotStore,
)

# v2.0.0: Runtime Context
from llmteam.runtime import (
    Store,
    Client,
    SecretsProvider,
    LLMProvider,
    StoreRegistry,
    ClientRegistry,
    LLMRegistry,
    RuntimeContext,
    StepContext,
    RuntimeContextManager,
    RuntimeContextFactory,
    current_runtime,
    get_current_runtime,
    ResourceNotFoundError,
    SecretAccessDeniedError,
    RuntimeContextError,
)

# v2.0.0: Worktrail Events
from llmteam.events import (
    EventType,
    EventSeverity,
    ErrorInfo,
    WorktrailEvent,
    EventEmitter,
    EventStore,
    MemoryEventStore,
    EventStream,
)

# v2.0.0: Canvas Integration
from llmteam.canvas import (
    # Models
    PortDefinition,
    StepPosition,
    StepUIMetadata,
    StepDefinition,
    EdgeDefinition,
    SegmentParams,
    SegmentDefinition,
    # Catalog
    StepCategory,
    PortSpec,
    StepTypeMetadata,
    StepCatalog,
    # Runner
    SegmentStatus,
    SegmentResult,
    RunConfig,
    SegmentRunner,
    SegmentSnapshot,
    SegmentSnapshotStore,
    # Handlers
    HumanTaskHandler,
    create_human_task_handler,
    # Exceptions
    CanvasError,
    SegmentValidationError,
    StepTypeNotFoundError,
    InvalidStepConfigError,
    InvalidConditionError,
)

# v2.0.0: Patterns
from llmteam.patterns import (
    CriticVerdict,
    CriticLoopConfig,
    CriticFeedback,
    IterationRecord,
    CriticLoopResult,
    CriticLoop,
)

# v2.0.0: New naming (LLMTeam) + Compatibility aliases
from llmteam.compat import (
    LLMTeam,
    TeamOrchestrator,
    Pipeline,  # Deprecated, use LLMTeam
)

# v2.0.0: Three-Level Ports (RFC #7)
from llmteam.ports import (
    PortLevel,
    PortDirection,
    PortDataType,
    TypedPort,
    StepPorts,
    workflow_input,
    workflow_output,
    agent_input,
    agent_output,
    human_input,
    human_output,
    llm_agent_ports,
    human_task_ports,
    transform_ports,
    http_action_ports,
)

# v2.0.0: Observability
from llmteam.observability import (
    configure_logging,
    get_logger,
    LogConfig,
    LogFormat,
)

# v2.0.0: Canvas Validation
from llmteam.canvas.validation import (
    ValidationSeverity,
    ValidationMessage,
    ValidationResult,
    SegmentValidator,
    validate_segment,
    validate_segment_dict,
)

# v2.0.3: Providers (lazy import to avoid optional dependency issues)
# Use: from llmteam.providers import OpenAIProvider

# v2.0.3: Testing utilities (lazy import)
# Use: from llmteam.testing import MockLLMProvider, SegmentTestRunner

# v2.0.3: Event transports (lazy import)
# Use: from llmteam.events.transports import WebSocketTransport, SSETransport

__all__ = [
    # Version
    "__version__",

    # Tenancy
    "TenantConfig",
    "TenantContext",
    "TenantManager",
    "TenantTier",
    "TenantLimits",
    "current_tenant",

    # Audit
    "AuditTrail",
    "AuditRecord",
    "AuditQuery",
    "AuditEventType",
    "AuditSeverity",

    # Context Security (v1.7.0)
    "SecureAgentContext",
    "ContextAccessPolicy",
    "SealedData",
    "VisibilityLevel",
    "SensitivityLevel",

    # Hierarchical Context (v1.8.0)
    "ContextScope",
    "HierarchicalContext",
    "ContextManager",
    "ContextPropagationConfig",

    # Rate Limiting (v1.7.0)
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitStrategy",
    "CircuitBreakerConfig",
    "CircuitState",
    "RateLimitedExecutor",

    # Licensing (v1.8.0 + v2.0.0 Open Core)
    "LicenseTier",
    "LicenseLimits",
    "License",
    "LicenseManager",
    "activate",
    "get_tier",
    "has_feature",
    "print_license_status",
    "get_license_manager",
    "LicenseValidationError",
    "LicenseExpiredError",
    "FeatureNotLicensedError",
    "professional_only",
    "enterprise_only",

    # Execution (v1.8.0)
    "ExecutionMode",
    "ExecutorConfig",
    "TaskResult",
    "ExecutionStats",
    "PipelineExecutor",

    # Orchestration (v1.8.0)
    "OrchestratorRole",
    "OrchestrationDecision",
    "OrchestrationContext",
    "OrchestrationStrategy",
    "RuleBasedStrategy",
    "LLMBasedStrategy",

    # Process Mining (v1.8.0)
    "ProcessEvent",
    "ProcessMetrics",
    "ProcessModel",
    "ProcessMiningEngine",

    # Pipeline Orchestrator (v1.8.0)
    "PipelineOrchestrator",

    # Group Orchestrator (v1.8.0)
    "GroupDecisionType",
    "GroupOrchestrationDecision",
    "PipelineStatus",
    "GroupOrchestrationStrategy",
    "LoadBalancingStrategy",
    "ContentBasedRoutingStrategy",
    "ParallelFanOutStrategy",
    "GroupOrchestrator",

    # External Actions (v1.9.0)
    "ActionType",
    "ActionStatus",
    "ActionConfig",
    "ActionContext",
    "ActionResult",
    "ActionRegistry",
    "ActionExecutor",

    # Human Interaction (v1.9.0)
    "InteractionType",
    "InteractionStatus",
    "InteractionPriority",
    "InteractionRequest",
    "InteractionResponse",
    "HumanInteractionManager",
    "MemoryInteractionStore",

    # Persistence (v1.9.0)
    "SnapshotType",
    "PipelinePhase",
    "AgentSnapshot",
    "PipelineSnapshot",
    "RestoreResult",
    "SnapshotManager",
    "MemorySnapshotStore",

    # Runtime Context (v2.0.0)
    "Store",
    "Client",
    "SecretsProvider",
    "LLMProvider",
    "StoreRegistry",
    "ClientRegistry",
    "LLMRegistry",
    "RuntimeContext",
    "StepContext",
    "RuntimeContextManager",
    "RuntimeContextFactory",
    "current_runtime",
    "get_current_runtime",
    "ResourceNotFoundError",
    "SecretAccessDeniedError",
    "RuntimeContextError",

    # Worktrail Events (v2.0.0)
    "EventType",
    "EventSeverity",
    "ErrorInfo",
    "WorktrailEvent",
    "EventEmitter",
    "EventStore",
    "MemoryEventStore",
    "EventStream",

    # Canvas Integration (v2.0.0)
    "PortDefinition",
    "StepPosition",
    "StepUIMetadata",
    "StepDefinition",
    "EdgeDefinition",
    "SegmentParams",
    "SegmentDefinition",
    "StepCategory",
    "PortSpec",
    "StepTypeMetadata",
    "StepCatalog",
    "SegmentStatus",
    "SegmentResult",
    "RunConfig",
    "SegmentRunner",
    "SegmentSnapshot",
    "SegmentSnapshotStore",
    "HumanTaskHandler",
    "create_human_task_handler",
    "CanvasError",
    "SegmentValidationError",
    "StepTypeNotFoundError",
    "InvalidStepConfigError",
    "InvalidConditionError",

    # Patterns (v2.0.0)
    "CriticVerdict",
    "CriticLoopConfig",
    "CriticFeedback",
    "IterationRecord",
    "CriticLoopResult",
    "CriticLoop",

    # Team (v2.0.0 naming)
    "LLMTeam",
    "TeamOrchestrator",
    "Pipeline",  # Deprecated

    # Three-Level Ports (v2.0.0)
    "PortLevel",
    "PortDirection",
    "PortDataType",
    "TypedPort",
    "StepPorts",
    "workflow_input",
    "workflow_output",
    "agent_input",
    "agent_output",
    "human_input",
    "human_output",
    "llm_agent_ports",
    "human_task_ports",
    "transform_ports",
    "http_action_ports",

    # Observability (v2.0.0)
    "configure_logging",
    "get_logger",
    "LogConfig",
    "LogFormat",

    # Validation (v2.0.0)
    "ValidationSeverity",
    "ValidationMessage",
    "ValidationResult",
    "SegmentValidator",
    "validate_segment",
    "validate_segment_dict",
]
