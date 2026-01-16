"""
llmteam - Enterprise AI Workflow Runtime

A library for building multi-agent LLM pipelines with enterprise-grade
security, orchestration, and workflow capabilities.

Version: 1.9.0 (Workflow Runtime)
"""

__version__ = "1.9.0"
__author__ = "llmteam contributors"

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

# v1.8.0: Licensing
from llmteam.licensing import (
    LicenseTier,
    LicenseLimits,
    LicenseManager,
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

    # Licensing (v1.8.0)
    "LicenseTier",
    "LicenseLimits",
    "LicenseManager",

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
]
