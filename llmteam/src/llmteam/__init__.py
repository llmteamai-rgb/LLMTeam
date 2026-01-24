"""
llmteam - Enterprise AI Workflow Runtime

A library for building multi-agent LLM pipelines with enterprise-grade
security, orchestration, and workflow capabilities.

Version: 5.3.0 (RFC-010/011/012/013/014: Enterprise Features)
    - New: RFC-012 Per-agent RetryPolicy & CircuitBreakerPolicy
    - New: RFC-010 CostTracker, BudgetManager, PricingRegistry
    - New: RFC-011 StreamEventType, StreamEvent, LLMTeam.stream()
    - New: RFC-013 ToolDefinition, @tool decorator, ToolExecutor (per-agent)
    - New: RFC-014 TeamState, TeamLifecycle, enforce_lifecycle opt-in

Version: 5.2.0 (RFC-009: Group Architecture Unification)
    - New: TeamRole enum (LEADER, MEMBER, SPECIALIST, FALLBACK)
    - New: GroupRole extensions (COORDINATOR, ROUTER, AGGREGATOR, ARBITER)
    - New: GroupContext for bi-directional team-group communication
    - New: EscalationRequest/EscalationResponse for group-level escalation
    - New: LLMTeam.is_in_group, group_id, group_role properties
    - New: LLMTeam.escalate_to_group(), request_team() methods
    - New: GroupOrchestrator role-based execution strategies

Version: 5.1.0 (RFC-008: Quality Slider)
    - New: QualityManager for quality/cost tradeoff control
    - New: Quality presets (draft, economy, balanced, production, best)
    - New: CostEstimate and CostEstimator for cost prediction
    - New: LLMTeam.quality parameter (0-100 or preset name)
    - New: LLMTeam.estimate_cost() method
    - New: run(quality=, importance=) parameters
    - New: Auto mode with budget-based quality adjustment

Version: 4.1.0 (Orchestrator Architecture Refactoring)
    - New: TeamOrchestrator as separate supervisor (not agent)
    - New: OrchestratorMode (SUPERVISOR, REPORTER, ROUTER, RECOVERY)
    - New: AgentReport for automatic agent reporting
    - New: ROUTER mode enables orchestrator to select agents
    - New: RunResult.report and RunResult.summary

Version: 4.0.0 (Agent Architecture Refactoring)
    - New: Typed agents (LLMAgent, RAGAgent, KAGAgent)
    - New: AgentFactory for creating agents
    - New: LLMTeam uses ExecutionEngine internally
    - New: LLMGroup for multi-team coordination

License Tiers:
    - COMMUNITY (free): Basic features, memory stores
    - PROFESSIONAL ($99/mo): Process mining, PostgreSQL, Human-in-the-loop
    - ENTERPRISE (custom): Multi-tenant, Audit trail, SSO

Quick Start:
    from llmteam import LLMTeam

    team = LLMTeam(
        team_id="content",
        agents=[
            {"type": "rag", "role": "retriever", "collection": "docs"},
            {"type": "llm", "role": "writer", "prompt": "Write about: {query}"}
        ]
    )
    result = await team.run({"query": "AI trends"})

Documentation: https://docs.llmteam.ai
"""

__version__ = "5.3.0"
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

# v3.0.0: Process Mining (new location)
from llmteam.mining import (
    ProcessEvent,
    ProcessMetrics,
    ProcessModel,
    ProcessMiningEngine,
)

# v3.0.0: Team Contract (new location)
from llmteam.contract import (
    TeamContract,
    ContractValidationResult,
    ContractValidationError,
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
    # RFC-011: Streaming
    StreamEventType,
    StreamEvent,
)

# v2.0.0: Canvas Integration (now Engine)
# v5.0.0: Renamed canvas → engine, made optional (pip install llmteam-ai[engine])
try:
    from llmteam.engine import (
        # Models - New names (v5.0.0)
        PortDefinition,
        StepPosition,
        StepUIMetadata,
        StepDefinition,
        EdgeDefinition,
        WorkflowParams,
        WorkflowDefinition,
        # Models - Backward compatibility aliases
        SegmentParams,
        SegmentDefinition,
        # Catalog
        StepCategory,
        PortSpec,
        StepTypeMetadata,
        StepCatalog,
        # Engine - New names (v5.0.0)
        ExecutionStatus,
        ExecutionResult,
        ExecutionEngine,
        ExecutionSnapshot,
        ExecutionSnapshotStore,
        RunConfig,
        # Engine - Backward compatibility aliases
        SegmentStatus,
        SegmentResult,
        SegmentRunner,
        SegmentSnapshot,
        SegmentSnapshotStore,
        # Handlers
        HumanTaskHandler,
        create_human_task_handler,
        # Exceptions - New names (v5.0.0)
        EngineError,
        WorkflowValidationError,
        # Exceptions - Backward compatibility aliases
        CanvasError,
        SegmentValidationError,
        # Other exceptions
        StepTypeNotFoundError,
        InvalidStepConfigError,
        InvalidConditionError,
    )
    _ENGINE_AVAILABLE = True
except ImportError:
    _ENGINE_AVAILABLE = False
    # Engine module not installed - these will be None
    # Users should use: from llmteam.engine import ... directly after pip install llmteam-ai[engine]
    PortDefinition = None  # type: ignore
    StepPosition = None  # type: ignore
    StepUIMetadata = None  # type: ignore
    StepDefinition = None  # type: ignore
    EdgeDefinition = None  # type: ignore
    WorkflowParams = None  # type: ignore
    WorkflowDefinition = None  # type: ignore
    SegmentParams = None  # type: ignore
    SegmentDefinition = None  # type: ignore
    StepCategory = None  # type: ignore
    PortSpec = None  # type: ignore
    StepTypeMetadata = None  # type: ignore
    StepCatalog = None  # type: ignore
    ExecutionStatus = None  # type: ignore
    ExecutionResult = None  # type: ignore
    ExecutionEngine = None  # type: ignore
    ExecutionSnapshot = None  # type: ignore
    ExecutionSnapshotStore = None  # type: ignore
    RunConfig = None  # type: ignore
    SegmentStatus = None  # type: ignore
    SegmentResult = None  # type: ignore
    SegmentRunner = None  # type: ignore
    SegmentSnapshot = None  # type: ignore
    SegmentSnapshotStore = None  # type: ignore
    HumanTaskHandler = None  # type: ignore
    create_human_task_handler = None  # type: ignore
    EngineError = None  # type: ignore
    WorkflowValidationError = None  # type: ignore
    CanvasError = None  # type: ignore
    SegmentValidationError = None  # type: ignore
    StepTypeNotFoundError = None  # type: ignore
    InvalidStepConfigError = None  # type: ignore
    InvalidConditionError = None  # type: ignore

# v2.0.0: Patterns
from llmteam.patterns import (
    CriticVerdict,
    CriticLoopConfig,
    CriticFeedback,
    IterationRecord,
    CriticLoopResult,
    CriticLoop,
)

# v4.0.0: New Agent Architecture
from llmteam.agents import (
    # Types
    AgentType,
    AgentMode,
    AgentStatus,
    # Config
    AgentConfig,
    LLMAgentConfig,
    RAGAgentConfig,
    KAGAgentConfig,
    # State & Result
    AgentState,
    AgentResult,
    RAGResult,
    KAGResult,
    # Factory
    AgentFactory,
    # Presets
    create_orchestrator_config,
    create_group_orchestrator_config,
    create_summarizer_config,
    create_reviewer_config,
    create_rag_config,
    create_kag_config,
    # v4.1.0: Orchestrator
    AgentReport,
    TeamOrchestrator,
    OrchestratorMode,
    OrchestratorScope,
    OrchestratorConfig,
    RoutingDecision,
    RecoveryDecision,
    RecoveryAction,
    # RFC-012: Per-agent Retry & Circuit Breaker
    RetryPolicy,
    CircuitBreakerPolicy,
    RetryMetrics,
    AgentRetryExecutor,
)

# v4.0.0: LLMTeam and LLMGroup
from llmteam.team import (
    LLMTeam,
    LLMGroup,
    RunResult,
    RunStatus,
    ContextMode,
    TeamResult,
    TeamSnapshot,
    TeamConfig,
)

# v3.0.0: Registries
from llmteam.registry import (
    BaseRegistry,
    TeamRegistry,
)

# v3.0.0: Escalation subsystem
from llmteam.escalation import (
    EscalationLevel,
    EscalationAction,
    Escalation,
    EscalationDecision,
    EscalationRecord,
    EscalationHandler,
    DefaultHandler,
    ThresholdHandler,
    FunctionHandler,
    ChainHandler,
    LevelFilterHandler,
)

# RFC-004: GroupOrchestrator
# RFC-009: Group Architecture Unification
from llmteam.orchestration import (
    GroupOrchestrator,
    GroupRole,
    TeamReport,
    GroupReport,
    GroupResult,
    # RFC-009: Team roles and context
    TeamRole,
    GroupContext,
    EscalationRequest,
    EscalationResponse,
    GroupEscalationAction,  # Group-level escalation actions
)

# RFC-005: Configuration (CONFIGURATOR mode)
from llmteam.configuration import (
    SessionState,
    AgentSuggestion,
    TestRunResult,
    TaskAnalysis,
    PipelinePreview,
    ConfiguratorPrompts,
    ConfigurationSession,
)

# RFC-010: Cost Tracking & Budget Management
from llmteam.cost import (
    ModelPricing,
    PricingRegistry,
    TokenUsage,
    RunCost,
    CostTracker,
    Budget,
    BudgetPeriod,
    BudgetStatus,
    BudgetManager,
    BudgetExceededError,
)

# RFC-014: Enhanced Configurator Mode (opt-in lifecycle)
from llmteam.team.lifecycle import (
    TeamState,
    ProposalStatus,
    ConfigurationProposal,
    TeamLifecycle,
    LifecycleError,
)

# RFC-013: Tool/Function Calling
from llmteam.tools import (
    ParamType,
    ToolParameter,
    ToolDefinition,
    ToolResult,
    tool,
    ToolExecutor,
)

# RFC-008: Quality Slider
from llmteam.quality import (
    QualityManager,
    QualityPreset,
    TaskComplexity,
    PipelineDepth,
    CostEstimate,
    CostEstimator,
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

# v2.0.0: Workflow Validation (formerly Canvas Validation)
# v5.0.0: Made optional with engine module
if _ENGINE_AVAILABLE:
    from llmteam.engine.validation import (
        ValidationSeverity,
        ValidationMessage,
        ValidationResult,
        SegmentValidator,
        validate_segment,
        validate_segment_dict,
    )
else:
    ValidationSeverity = None  # type: ignore
    ValidationMessage = None  # type: ignore
    ValidationResult = None  # type: ignore
    SegmentValidator = None  # type: ignore
    validate_segment = None  # type: ignore
    validate_segment_dict = None  # type: ignore

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
    # RFC-011: Streaming
    "StreamEventType",
    "StreamEvent",

    # Engine Integration (v5.0.0, formerly Canvas v2.0.0)
    "PortDefinition",
    "StepPosition",
    "StepUIMetadata",
    "StepDefinition",
    "EdgeDefinition",
    # New names (v5.0.0)
    "WorkflowParams",
    "WorkflowDefinition",
    "ExecutionStatus",
    "ExecutionResult",
    "ExecutionEngine",
    "ExecutionSnapshot",
    "ExecutionSnapshotStore",
    "EngineError",
    "WorkflowValidationError",
    # Backward compatibility (deprecated)
    "SegmentParams",
    "SegmentDefinition",
    "SegmentStatus",
    "SegmentResult",
    "SegmentRunner",
    "SegmentSnapshot",
    "SegmentSnapshotStore",
    "CanvasError",
    "SegmentValidationError",
    # Catalog
    "StepCategory",
    "PortSpec",
    "StepTypeMetadata",
    "StepCatalog",
    "RunConfig",
    "HumanTaskHandler",
    "create_human_task_handler",
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

    # v4.0.0: Agent Architecture
    "AgentType",
    "AgentMode",
    "AgentStatus",
    "AgentConfig",
    "LLMAgentConfig",
    "RAGAgentConfig",
    "KAGAgentConfig",
    "AgentState",
    "AgentResult",
    "RAGResult",
    "KAGResult",
    "AgentFactory",
    "create_orchestrator_config",
    "create_group_orchestrator_config",
    "create_summarizer_config",
    "create_reviewer_config",
    "create_rag_config",
    "create_kag_config",

    # v4.1.0: Orchestrator
    "AgentReport",
    "TeamOrchestrator",
    "OrchestratorMode",
    "OrchestratorScope",
    "OrchestratorConfig",
    "RoutingDecision",
    "RecoveryDecision",
    "RecoveryAction",

    # RFC-012: Per-agent Retry & Circuit Breaker
    "RetryPolicy",
    "CircuitBreakerPolicy",
    "RetryMetrics",
    "AgentRetryExecutor",

    # v4.0.0: Team
    "LLMTeam",
    "LLMGroup",
    "RunResult",
    "RunStatus",
    "ContextMode",
    "TeamResult",
    "TeamSnapshot",
    "TeamConfig",

    # Registries (v3.0.0)
    "BaseRegistry",
    "TeamRegistry",

    # Escalation (v3.0.0)
    "EscalationLevel",
    "EscalationAction",
    "Escalation",
    "EscalationDecision",
    "EscalationRecord",
    "EscalationHandler",
    "DefaultHandler",
    "ThresholdHandler",
    "FunctionHandler",
    "ChainHandler",
    "LevelFilterHandler",

    # RFC-004: GroupOrchestrator
    # RFC-009: Group Architecture Unification
    "GroupOrchestrator",
    "GroupRole",
    "TeamReport",
    "GroupReport",
    "GroupResult",
    # RFC-009: Team roles and context
    "TeamRole",
    "GroupContext",
    "EscalationRequest",
    "EscalationResponse",
    "GroupEscalationAction",  # Group-level escalation (different from EscalationAction)

    # RFC-005: Configuration (CONFIGURATOR mode)
    "SessionState",
    "AgentSuggestion",
    "TestRunResult",
    "TaskAnalysis",
    "PipelinePreview",
    "ConfiguratorPrompts",
    "ConfigurationSession",

    # RFC-010: Cost Tracking & Budget Management
    "ModelPricing",
    "PricingRegistry",
    "TokenUsage",
    "RunCost",
    "CostTracker",
    "Budget",
    "BudgetPeriod",
    "BudgetStatus",
    "BudgetManager",
    "BudgetExceededError",

    # RFC-014: Enhanced Configurator Mode
    "TeamState",
    "ProposalStatus",
    "ConfigurationProposal",
    "TeamLifecycle",
    "LifecycleError",

    # RFC-013: Tool/Function Calling
    "ParamType",
    "ToolParameter",
    "ToolDefinition",
    "ToolResult",
    "tool",
    "ToolExecutor",

    # RFC-008: Quality Slider
    "QualityManager",
    "QualityPreset",
    "TaskComplexity",
    "PipelineDepth",
    "CostEstimate",
    "CostEstimator",

    # Contract (v3.0.0)
    "TeamContract",
    "ContractValidationResult",
    "ContractValidationError",

    # Mining (v3.0.0)
    "ProcessEvent",
    "ProcessMetrics",
    "ProcessModel",
    "ProcessMiningEngine",

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
