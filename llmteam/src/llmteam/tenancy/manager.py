"""
Tenant management functionality.

This module provides TenantManager for CRUD operations on tenants
and limit/feature checking.
"""

from datetime import datetime
from typing import Dict, List, Optional, Protocol, runtime_checkable

from llmteam.tenancy.models import (
    TenantConfig,
    TenantLimits,
    TenantTier,
    TenantNotFoundError,
    TenantLimitExceededError,
    TenantFeatureDisabledError,
)
from llmteam.tenancy.context import TenantContext


@runtime_checkable
class TenantStore(Protocol):
    """
    Protocol for tenant storage backends.
    
    Implementations must provide async CRUD operations for tenant configs.
    """
    
    async def get(self, tenant_id: str) -> Optional[TenantConfig]:
        """Get a tenant by ID."""
        ...
    
    async def create(self, config: TenantConfig) -> None:
        """Create a new tenant."""
        ...
    
    async def update(self, config: TenantConfig) -> None:
        """Update an existing tenant."""
        ...
    
    async def delete(self, tenant_id: str) -> None:
        """Delete a tenant."""
        ...
    
    async def list(self, limit: int = 100, offset: int = 0) -> List[TenantConfig]:
        """List tenants with pagination."""
        ...


from llmteam.licensing import enterprise_only


@enterprise_only
class TenantManager:
    """
    Manager for tenant operations.
    
    Provides:
    - CRUD operations for tenants
    - Limit checking and enforcement
    - Feature availability checking
    - Tenant context creation
    
    Example:
        store = MemoryTenantStore()
        manager = TenantManager(store)
        
        # Create tenant
        await manager.create_tenant(TenantConfig(
            tenant_id="acme",
            name="Acme Corp",
            tier=TenantTier.PROFESSIONAL,
        ))
        
        # Use tenant context
        async with manager.context("acme"):
            # Operations are now in context of "acme"
            if await manager.check_feature("acme", "parallel_execution"):
                # Feature is available
                pass
    """
    
    def __init__(self, store: TenantStore, cache_enabled: bool = True):
        """
        Initialize TenantManager.
        
        Args:
            store: Storage backend for tenant data
            cache_enabled: Whether to cache tenant configs in memory
        """
        self.store = store
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, TenantConfig] = {}
    
    # CRUD Operations
    
    async def get_tenant(self, tenant_id: str) -> TenantConfig:
        """
        Get a tenant configuration.
        
        Args:
            tenant_id: The tenant ID to look up
            
        Returns:
            TenantConfig for the tenant
            
        Raises:
            TenantNotFoundError: If tenant doesn't exist
        """
        # Check cache first
        if self.cache_enabled and tenant_id in self._cache:
            return self._cache[tenant_id]
        
        # Load from store
        config = await self.store.get(tenant_id)
        if config is None:
            raise TenantNotFoundError(f"Tenant '{tenant_id}' not found")
        
        # Update cache
        if self.cache_enabled:
            self._cache[tenant_id] = config
        
        return config
    
    async def create_tenant(self, config: TenantConfig) -> TenantConfig:
        """
        Create a new tenant.
        
        Args:
            config: Tenant configuration
            
        Returns:
            The created TenantConfig
            
        Raises:
            TenantError: If tenant already exists
        """
        # Check if exists
        existing = await self.store.get(config.tenant_id)
        if existing is not None:
            raise TenantNotFoundError(f"Tenant '{config.tenant_id}' already exists")
        
        # Set timestamps
        config.created_at = datetime.now()
        config.updated_at = datetime.now()
        
        # Save
        await self.store.create(config)
        
        # Update cache
        if self.cache_enabled:
            self._cache[config.tenant_id] = config
        
        return config
    
    async def update_tenant(self, config: TenantConfig) -> TenantConfig:
        """
        Update an existing tenant.
        
        Args:
            config: Updated tenant configuration
            
        Returns:
            The updated TenantConfig
            
        Raises:
            TenantNotFoundError: If tenant doesn't exist
        """
        # Check exists
        existing = await self.store.get(config.tenant_id)
        if existing is None:
            raise TenantNotFoundError(f"Tenant '{config.tenant_id}' not found")
        
        # Update timestamp
        config.updated_at = datetime.now()
        
        # Save
        await self.store.update(config)
        
        # Update cache
        if self.cache_enabled:
            self._cache[config.tenant_id] = config
        
        return config
    
    async def delete_tenant(self, tenant_id: str) -> None:
        """
        Delete a tenant.
        
        Args:
            tenant_id: The tenant ID to delete
            
        Raises:
            TenantNotFoundError: If tenant doesn't exist
        """
        # Check exists
        existing = await self.store.get(tenant_id)
        if existing is None:
            raise TenantNotFoundError(f"Tenant '{tenant_id}' not found")
        
        # Delete
        await self.store.delete(tenant_id)
        
        # Remove from cache
        self._cache.pop(tenant_id, None)
    
    async def list_tenants(self, limit: int = 100, offset: int = 0) -> List[TenantConfig]:
        """
        List all tenants with pagination.
        
        Args:
            limit: Maximum number of tenants to return
            offset: Number of tenants to skip
            
        Returns:
            List of TenantConfig objects
        """
        return await self.store.list(limit=limit, offset=offset)
    
    # Limits and Features
    
    def get_limits(self, config: TenantConfig) -> TenantLimits:
        """
        Get effective limits for a tenant.
        
        Args:
            config: Tenant configuration
            
        Returns:
            TenantLimits with all overrides applied
        """
        return config.get_effective_limits()
    
    async def check_limit(
        self,
        tenant_id: str,
        limit_type: str,
        current_value: int,
    ) -> bool:
        """
        Check if a limit would be exceeded.
        
        Args:
            tenant_id: The tenant to check
            limit_type: Type of limit (e.g., "concurrent_pipelines")
            current_value: Current usage value
            
        Returns:
            True if within limits, False if would exceed
        """
        config = await self.get_tenant(tenant_id)
        limits = self.get_limits(config)
        
        limit_map = {
            "concurrent_pipelines": limits.max_concurrent_pipelines,
            "agents_per_pipeline": limits.max_agents_per_pipeline,
            "requests_per_minute": limits.max_requests_per_minute,
            "runs_per_day": limits.max_runs_per_day,
        }
        
        max_value = limit_map.get(limit_type)
        if max_value is None:
            return True  # Unknown limit type, allow
        
        return current_value < max_value
    
    async def enforce_limit(
        self,
        tenant_id: str,
        limit_type: str,
        current_value: int,
    ) -> None:
        """
        Enforce a limit, raising an exception if exceeded.
        
        Args:
            tenant_id: The tenant to check
            limit_type: Type of limit
            current_value: Current usage value
            
        Raises:
            TenantLimitExceededError: If limit is exceeded
        """
        config = await self.get_tenant(tenant_id)
        limits = self.get_limits(config)
        
        limit_map = {
            "concurrent_pipelines": limits.max_concurrent_pipelines,
            "agents_per_pipeline": limits.max_agents_per_pipeline,
            "requests_per_minute": limits.max_requests_per_minute,
            "runs_per_day": limits.max_runs_per_day,
        }
        
        max_value = limit_map.get(limit_type)
        if max_value is not None and current_value >= max_value:
            raise TenantLimitExceededError(
                tenant_id=tenant_id,
                limit_type=limit_type,
                current=current_value,
                maximum=max_value,
            )
    
    async def check_feature(self, tenant_id: str, feature: str) -> bool:
        """
        Check if a feature is available for a tenant.
        
        Args:
            tenant_id: The tenant to check
            feature: Feature name to check
            
        Returns:
            True if feature is available
        """
        config = await self.get_tenant(tenant_id)
        limits = self.get_limits(config)
        return limits.has_feature(feature)
    
    async def require_feature(self, tenant_id: str, feature: str) -> None:
        """
        Require a feature, raising an exception if not available.
        
        Args:
            tenant_id: The tenant to check
            feature: Feature name to require
            
        Raises:
            TenantFeatureDisabledError: If feature is not available
        """
        if not await self.check_feature(tenant_id, feature):
            raise TenantFeatureDisabledError(tenant_id=tenant_id, feature=feature)
    
    # Context Management
    
    def context(self, tenant_id: str) -> TenantContext:
        """
        Create a tenant context.
        
        Args:
            tenant_id: The tenant ID to set
            
        Returns:
            TenantContext for use with `with` or `async with`
        """
        config = self._cache.get(tenant_id)
        return TenantContext(tenant_id=tenant_id, config=config)
    
    # Cache Management
    
    def invalidate_cache(self, tenant_id: str = None) -> None:
        """
        Invalidate cached tenant data.
        
        Args:
            tenant_id: Specific tenant to invalidate, or None for all
        """
        if tenant_id is None:
            self._cache.clear()
        else:
            self._cache.pop(tenant_id, None)
    
    def preload_cache(self, configs: List[TenantConfig]) -> None:
        """
        Preload the cache with tenant configurations.
        
        Args:
            configs: List of TenantConfig objects to cache
        """
        if self.cache_enabled:
            for config in configs:
                self._cache[config.tenant_id] = config
