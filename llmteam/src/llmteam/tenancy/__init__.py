"""
Multi-tenant support for llmteam.

This module provides complete multi-tenant isolation including:
- Tenant configuration and management
- Context variables for tenant tracking
- Tenant-isolated storage
- Limit and feature checking

Quick Start:
    from llmteam.tenancy import TenantManager, TenantConfig, TenantTier, TenantContext
    from llmteam.tenancy.stores import MemoryTenantStore
    
    # Create manager
    store = MemoryTenantStore()
    manager = TenantManager(store)
    
    # Create tenant
    await manager.create_tenant(TenantConfig(
        tenant_id="acme",
        name="Acme Corporation",
        tier=TenantTier.PROFESSIONAL,
    ))
    
    # Use tenant context
    async with manager.context("acme"):
        # All operations here are in context of "acme"
        pass
"""

from llmteam.tenancy.models import (
    TenantTier,
    TenantLimits,
    TenantConfig,
    TenantError,
    TenantNotFoundError,
    TenantLimitExceededError,
    TenantFeatureDisabledError,
    TenantContextError,
    TIER_LIMITS,
    get_tier_limits,
)

from llmteam.tenancy.context import (
    current_tenant,
    get_current_tenant,
    require_tenant_context,
    TenantContext,
    TenantContextStack,
)

from llmteam.tenancy.manager import (
    TenantStore,
    TenantManager,
)

from llmteam.tenancy.isolation import (
    KeyValueStore,
    TenantIsolatedStore,
    CrossTenantStore,
)

__all__ = [
    # Models
    "TenantTier",
    "TenantLimits",
    "TenantConfig",
    "TIER_LIMITS",
    "get_tier_limits",
    
    # Exceptions
    "TenantError",
    "TenantNotFoundError",
    "TenantLimitExceededError",
    "TenantFeatureDisabledError",
    "TenantContextError",
    
    # Context
    "current_tenant",
    "get_current_tenant",
    "require_tenant_context",
    "TenantContext",
    "TenantContextStack",
    
    # Manager
    "TenantStore",
    "TenantManager",
    
    # Isolation
    "KeyValueStore",
    "TenantIsolatedStore",
    "CrossTenantStore",
]
