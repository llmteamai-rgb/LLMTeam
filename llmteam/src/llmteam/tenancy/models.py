"""
Tenant models and data structures.

This module defines the core data structures for multi-tenant support:
- TenantTier: Subscription levels
- TenantLimits: Resource limits per tier
- TenantConfig: Tenant configuration
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Set, TypedDict, List


class TenantTier(Enum):
    """
    Subscription tier for tenants.
    
    Each tier has different resource limits and feature access.
    """
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class TenantLimitsDict(TypedDict):
    """Dictionary representation of TenantLimits."""
    max_concurrent_pipelines: int
    max_agents_per_pipeline: int
    max_requests_per_minute: int
    max_storage_gb: float
    max_runs_per_day: int
    features: List[str]


@dataclass
class TenantLimits:
    """
    Resource limits for a tenant tier.
    
    Attributes:
        max_concurrent_pipelines: Maximum number of pipelines running simultaneously
        max_agents_per_pipeline: Maximum agents in a single pipeline
        max_requests_per_minute: API rate limit
        max_storage_gb: Storage quota in GB
        max_runs_per_day: Maximum pipeline runs per day
        features: Set of enabled feature names
    """
    max_concurrent_pipelines: int
    max_agents_per_pipeline: int
    max_requests_per_minute: int
    max_storage_gb: float
    max_runs_per_day: int
    features: Set[str]
    
    def has_feature(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        return "*" in self.features or feature in self.features
    
    def to_dict(self) -> TenantLimitsDict:
        """Convert to dictionary."""
        return {
            "max_concurrent_pipelines": self.max_concurrent_pipelines,
            "max_agents_per_pipeline": self.max_agents_per_pipeline,
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_storage_gb": self.max_storage_gb,
            "max_runs_per_day": self.max_runs_per_day,
            "features": list(self.features),
        }


# Default limits per tier
TIER_LIMITS: Dict[TenantTier, TenantLimits] = {
    TenantTier.FREE: TenantLimits(
        max_concurrent_pipelines=1,
        max_agents_per_pipeline=5,
        max_requests_per_minute=10,
        max_storage_gb=1.0,
        max_runs_per_day=100,
        features={"basic_agents", "simple_pipelines"},
    ),
    TenantTier.STARTER: TenantLimits(
        max_concurrent_pipelines=2,
        max_agents_per_pipeline=10,
        max_requests_per_minute=60,
        max_storage_gb=10.0,
        max_runs_per_day=1000,
        features={"basic_agents", "simple_pipelines", "parallel_execution"},
    ),
    TenantTier.PROFESSIONAL: TenantLimits(
        max_concurrent_pipelines=10,
        max_agents_per_pipeline=50,
        max_requests_per_minute=300,
        max_storage_gb=100.0,
        max_runs_per_day=10000,
        features={
            "basic_agents", 
            "simple_pipelines", 
            "parallel_execution",
            "external_actions", 
            "human_interaction", 
            "persistence",
        },
    ),
    TenantTier.ENTERPRISE: TenantLimits(
        max_concurrent_pipelines=999999,
        max_agents_per_pipeline=999999,
        max_requests_per_minute=999999,
        max_storage_gb=999999.0,
        max_runs_per_day=999999,
        features={"*"},  # All features
    ),
}


def get_tier_limits(tier: TenantTier) -> TenantLimits:
    """Get default limits for a tier."""
    return TIER_LIMITS[tier]


class TenantConfigDict(TypedDict):
    """Dictionary representation of TenantConfig."""
    tenant_id: str
    name: str
    tier: str
    max_concurrent_pipelines: Optional[int]
    max_agents_per_pipeline: Optional[int]
    max_requests_per_minute: Optional[int]
    features_enabled: List[str]
    features_disabled: List[str]
    allowed_actions: List[str]
    blocked_actions: List[str]
    data_region: str
    encryption_key_id: str
    audit_retention_days: int
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str
    is_active: bool


@dataclass
class TenantConfig:
    """
    Configuration for a tenant.
    
    Attributes:
        tenant_id: Unique identifier for the tenant
        name: Human-readable name
        tier: Subscription tier
        max_concurrent_pipelines: Override for concurrent pipeline limit
        max_agents_per_pipeline: Override for agents per pipeline limit
        max_requests_per_minute: Override for rate limit
        features_enabled: Additional features to enable
        features_disabled: Features to disable (override tier defaults)
        allowed_actions: Whitelist of allowed external actions
        blocked_actions: Blacklist of blocked external actions
        data_region: Data residency region
        encryption_key_id: Custom encryption key ID
        audit_retention_days: How long to retain audit logs
        metadata: Custom metadata
        created_at: Creation timestamp
        updated_at: Last update timestamp
        is_active: Whether tenant is active
    """
    tenant_id: str
    name: str
    tier: TenantTier = TenantTier.FREE
    
    # Limit overrides (None = use tier default)
    max_concurrent_pipelines: Optional[int] = None
    max_agents_per_pipeline: Optional[int] = None
    max_requests_per_minute: Optional[int] = None
    
    # Feature overrides
    features_enabled: Set[str] = field(default_factory=set)
    features_disabled: Set[str] = field(default_factory=set)
    
    # Security
    allowed_actions: Set[str] = field(default_factory=set)
    blocked_actions: Set[str] = field(default_factory=set)
    
    # Data residency
    data_region: str = "default"
    encryption_key_id: str = ""
    
    # Audit
    audit_retention_days: int = 90
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    
    def get_effective_limits(self) -> TenantLimits:
        """
        Get effective limits combining tier defaults with overrides.
        
        Returns:
            TenantLimits with overrides applied
        """
        base = TIER_LIMITS[self.tier]
        
        # Resolve features
        features = base.features.copy()
        features |= self.features_enabled
        features -= self.features_disabled
        
        return TenantLimits(
            max_concurrent_pipelines=self.max_concurrent_pipelines or base.max_concurrent_pipelines,
            max_agents_per_pipeline=self.max_agents_per_pipeline or base.max_agents_per_pipeline,
            max_requests_per_minute=self.max_requests_per_minute or base.max_requests_per_minute,
            max_storage_gb=base.max_storage_gb,
            max_runs_per_day=base.max_runs_per_day,
            features=features,
        )
    
    def is_action_allowed(self, action_name: str) -> bool:
        """
        Check if an external action is allowed for this tenant.
        
        Args:
            action_name: Name of the action to check
            
        Returns:
            True if action is allowed
        """
        # Explicit block takes precedence
        if action_name in self.blocked_actions:
            return False
        
        # If whitelist is defined, action must be in it
        if self.allowed_actions:
            return action_name in self.allowed_actions
        
        # Default: allow all
        return True
    
    def to_dict(self) -> TenantConfigDict:
        """Convert to dictionary for serialization."""
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "tier": self.tier.value,
            "max_concurrent_pipelines": self.max_concurrent_pipelines,
            "max_agents_per_pipeline": self.max_agents_per_pipeline,
            "max_requests_per_minute": self.max_requests_per_minute,
            "features_enabled": list(self.features_enabled),
            "features_disabled": list(self.features_disabled),
            "allowed_actions": list(self.allowed_actions),
            "blocked_actions": list(self.blocked_actions),
            "data_region": self.data_region,
            "encryption_key_id": self.encryption_key_id,
            "audit_retention_days": self.audit_retention_days,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_active": self.is_active,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TenantConfig":
        """Create from dictionary."""
        return cls(
            tenant_id=data["tenant_id"],
            name=data["name"],
            tier=TenantTier(data["tier"]),
            max_concurrent_pipelines=data.get("max_concurrent_pipelines"),
            max_agents_per_pipeline=data.get("max_agents_per_pipeline"),
            max_requests_per_minute=data.get("max_requests_per_minute"),
            features_enabled=set(data.get("features_enabled", [])),
            features_disabled=set(data.get("features_disabled", [])),
            allowed_actions=set(data.get("allowed_actions", [])),
            blocked_actions=set(data.get("blocked_actions", [])),
            data_region=data.get("data_region", "default"),
            encryption_key_id=data.get("encryption_key_id", ""),
            audit_retention_days=data.get("audit_retention_days", 90),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(),
            is_active=data.get("is_active", True),
        )


# Exceptions

class TenantError(Exception):
    """Base exception for tenant-related errors."""
    pass


class TenantNotFoundError(TenantError):
    """Raised when a tenant is not found."""
    pass


class TenantLimitExceededError(TenantError):
    """Raised when a tenant limit is exceeded."""
    
    def __init__(self, tenant_id: str, limit_type: str, current: int, maximum: int):
        self.tenant_id = tenant_id
        self.limit_type = limit_type
        self.current = current
        self.maximum = maximum
        super().__init__(
            f"Tenant '{tenant_id}' exceeded {limit_type} limit: {current}/{maximum}"
        )


class TenantFeatureDisabledError(TenantError):
    """Raised when a feature is not available for the tenant."""
    
    def __init__(self, tenant_id: str, feature: str):
        self.tenant_id = tenant_id
        self.feature = feature
        super().__init__(
            f"Feature '{feature}' is not available for tenant '{tenant_id}'"
        )


class TenantContextError(TenantError):
    """Raised when there's no tenant context."""
    pass
