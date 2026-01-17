# src/llmteam/__init__.py
"""
LLMTeam - Hierarchical Multi-Agent Orchestration Platform

Build enterprise AI workforces, not just agents.

Quick Start:
    from llmteam import LLMTeam, Agent
    
    team = LLMTeam(
        name="My Team",
        agents=[
            Agent(name="Writer", llm_ref="gpt4"),
            Agent(name="Reviewer", llm_ref="gpt4"),
        ],
    )
    
    result = await team.run(ctx, {"task": "Write a report"})

License Tiers:
    - COMMUNITY (free): Basic features, memory stores
    - PROFESSIONAL ($99/mo): Process mining, PostgreSQL, Human-in-the-loop
    - ENTERPRISE (custom): Multi-tenant, Audit trail, SSO

Activate License:
    import llmteam
    llmteam.activate("LLMT-PRO-XXXX-20261231")

Documentation: https://docs.llmteam.ai
"""

__version__ = "1.9.0"
__author__ = "KirilinVS"
__email__ = "LLMTeamai@gmail.com"

# === Licensing (always available) ===
from .licensing import (
    # Activation
    activate,
    get_tier,
    has_feature,
    print_license_status,
    
    # Models
    LicenseTier,
    License,
    LicenseManager,
    
    # Exceptions
    FeatureNotLicensedError,
    LicenseValidationError,
    LicenseExpiredError,
)

# === Core (COMMUNITY) ===
# These will be available after Pipeline→LLMTeam rename in v2.0.0
# For now, export existing classes

# Tenancy models (models are free, manager is enterprise)
from .tenancy.models import (
    TenantConfig,
    TenantTier,
    TenantLimits,
    TenantNotFoundError,
    TenantLimitExceededError,
)

# Context (COMMUNITY)
from .context.security import (
    SecureAgentContext,
    ContextAccessPolicy,
    SealedData,
    VisibilityLevel,
    SensitivityLevel,
)

from .context.hierarchical import (
    ContextScope,
    ContextPropagationConfig,
)

# Rate limiting - basic (COMMUNITY)
from .ratelimit.config import (
    RateLimitConfig,
    CircuitBreakerConfig,
    RateLimitStrategy,
)
from .ratelimit.limiter import RateLimiter
from .ratelimit.circuit import CircuitBreaker

# Orchestration strategies (COMMUNITY)
from .roles.orchestration import (
    OrchestrationStrategy,
    OrchestrationDecision,
    RuleBasedStrategy,
    LLMBasedStrategy,
)

# Execution (COMMUNITY)
from .execution.config import (
    ExecutorConfig,
    ExecutionMode,
)

# Actions models (COMMUNITY - models only)
from .actions.models import (
    ActionType,
    ActionConfig,
    ActionResult,
    ActionStatus,
)
from .actions.registry import ActionRegistry

# Human models (COMMUNITY - models only)
from .human.models import (
    InteractionType,
    InteractionStatus,
    InteractionRequest,
    InteractionResponse,
)

# Persistence models (COMMUNITY - models only)
from .persistence.models import (
    SnapshotType,
    PipelineSnapshot,
    AgentSnapshot,
)

# Memory stores (COMMUNITY)
from .persistence.stores.memory import MemorySnapshotStore
from .tenancy.stores.memory import MemoryTenantStore
from .audit.stores.memory import MemoryAuditStore

# === PROFESSIONAL Features ===
# These require Professional license

# Process Mining
from .roles.process_mining import (
    ProcessMiningEngine,  # @professional_only
    ProcessEvent,
    ProcessMetrics,
    ProcessModel,
)

# Pipeline Orchestrator
from .roles.pipeline_orch import PipelineOrchestrator
from .roles.group_orch import GroupOrchestrator

# Human Interaction
from .human.manager import HumanInteractionManager  # @professional_only

# External Actions
from .actions.executor import ActionExecutor  # @professional_only

# Rate Limited Executor
from .ratelimit.executor import RateLimitedExecutor  # @professional_only

# PostgreSQL stores
from .persistence.stores.postgres import PostgresSnapshotStore  # @professional_only

# === ENTERPRISE Features ===
# These require Enterprise license

# Multi-tenant
from .tenancy.manager import TenantManager  # @enterprise_only
from .tenancy.context import TenantContext, current_tenant  # @enterprise_only
from .tenancy.stores.postgres import PostgresTenantStore  # @enterprise_only

# Audit
from .audit.models import (
    AuditRecord,
    AuditQuery,
    AuditEventType,
)
from .audit.trail import AuditTrail  # @enterprise_only
from .audit.stores.postgres import PostgresAuditStore  # @enterprise_only


# === Backward Compatibility ===
# Pipeline alias (deprecated, use LLMTeam in v2.0.0)
Pipeline = PipelineOrchestrator  # Temporary alias


# === Public API ===
__all__ = [
    # Version
    "__version__",
    "__author__",
    
    # Licensing
    "activate",
    "get_tier",
    "has_feature",
    "print_license_status",
    "LicenseTier",
    "License",
    "LicenseManager",
    "FeatureNotLicensedError",
    "LicenseValidationError",
    "LicenseExpiredError",
    
    # === COMMUNITY ===
    # Tenancy models
    "TenantConfig",
    "TenantTier",
    "TenantLimits",
    "TenantNotFoundError",
    "TenantLimitExceededError",
    
    # Context
    "SecureAgentContext",
    "ContextAccessPolicy",
    "SealedData",
    "VisibilityLevel",
    "SensitivityLevel",
    "ContextScope",
    "ContextPropagationConfig",
    
    # Rate limiting
    "RateLimitConfig",
    "CircuitBreakerConfig",
    "RateLimitStrategy",
    "RateLimiter",
    "CircuitBreaker",
    
    # Orchestration
    "OrchestrationStrategy",
    "OrchestrationDecision",
    "RuleBasedStrategy",
    "LLMBasedStrategy",
    
    # Execution
    "ExecutorConfig",
    "ExecutionMode",
    
    # Actions models
    "ActionType",
    "ActionConfig",
    "ActionResult",
    "ActionStatus",
    "ActionRegistry",
    
    # Human models
    "InteractionType",
    "InteractionStatus",
    "InteractionRequest",
    "InteractionResponse",
    
    # Persistence models
    "SnapshotType",
    "PipelineSnapshot",
    "AgentSnapshot",
    
    # Memory stores
    "MemorySnapshotStore",
    "MemoryTenantStore",
    "MemoryAuditStore",
    
    # === PROFESSIONAL ===
    "ProcessMiningEngine",
    "ProcessEvent",
    "ProcessMetrics",
    "ProcessModel",
    "PipelineOrchestrator",
    "GroupOrchestrator",
    "HumanInteractionManager",
    "ActionExecutor",
    "RateLimitedExecutor",
    "PostgresSnapshotStore",
    
    # === ENTERPRISE ===
    "TenantManager",
    "TenantContext",
    "current_tenant",
    "PostgresTenantStore",
    "AuditRecord",
    "AuditQuery",
    "AuditEventType",
    "AuditTrail",
    "PostgresAuditStore",
    
    # Aliases
    "Pipeline",
]


# === Startup message ===
def _show_startup_info():
    """Show startup info in development mode."""
    import os
    if os.environ.get("LLMTEAM_SHOW_LICENSE_INFO", "0") == "1":
        print_license_status()

# Uncomment to show license status on import
# _show_startup_info()
