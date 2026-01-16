"""
In-memory tenant store implementation.

This store is useful for testing and development. Data is lost when
the process exits.
"""

from typing import Dict, List, Optional

from llmteam.tenancy.models import TenantConfig


class MemoryTenantStore:
    """
    In-memory implementation of TenantStore.
    
    Stores tenant configurations in a dictionary. Useful for:
    - Unit testing
    - Development and prototyping
    - Single-instance deployments without persistence needs
    
    Example:
        store = MemoryTenantStore()
        manager = TenantManager(store)
        
        await manager.create_tenant(TenantConfig(
            tenant_id="test",
            name="Test Tenant",
        ))
    """
    
    def __init__(self):
        """Initialize empty store."""
        self._data: Dict[str, TenantConfig] = {}
    
    async def get(self, tenant_id: str) -> Optional[TenantConfig]:
        """
        Get a tenant by ID.
        
        Args:
            tenant_id: The tenant ID to look up
            
        Returns:
            TenantConfig if found, None otherwise
        """
        return self._data.get(tenant_id)
    
    async def create(self, config: TenantConfig) -> None:
        """
        Create a new tenant.
        
        Args:
            config: Tenant configuration to store
        """
        self._data[config.tenant_id] = config
    
    async def update(self, config: TenantConfig) -> None:
        """
        Update an existing tenant.
        
        Args:
            config: Updated tenant configuration
        """
        self._data[config.tenant_id] = config
    
    async def delete(self, tenant_id: str) -> None:
        """
        Delete a tenant.
        
        Args:
            tenant_id: The tenant ID to delete
        """
        self._data.pop(tenant_id, None)
    
    async def list(self, limit: int = 100, offset: int = 0) -> List[TenantConfig]:
        """
        List tenants with pagination.
        
        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of TenantConfig objects
        """
        all_configs = list(self._data.values())
        return all_configs[offset:offset + limit]
    
    async def count(self) -> int:
        """
        Get total number of tenants.
        
        Returns:
            Number of tenants
        """
        return len(self._data)
    
    def clear(self) -> None:
        """Clear all tenants (useful for testing)."""
        self._data.clear()


class MemoryKeyValueStore:
    """
    In-memory key-value store for use with TenantIsolatedStore.
    
    Example:
        kv_store = MemoryKeyValueStore()
        isolated_store = TenantIsolatedStore(kv_store)
    """
    
    def __init__(self):
        """Initialize empty store."""
        self._data: Dict[str, any] = {}
    
    async def get(self, key: str) -> Optional[any]:
        """Get a value by key."""
        return self._data.get(key)
    
    async def set(self, key: str, value: any) -> None:
        """Set a value for a key."""
        self._data[key] = value
    
    async def delete(self, key: str) -> None:
        """Delete a key."""
        self._data.pop(key, None)
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        return key in self._data
    
    async def list(self, prefix: str = "") -> List[str]:
        """List keys with optional prefix filter."""
        if not prefix:
            return list(self._data.keys())
        return [k for k in self._data.keys() if k.startswith(prefix)]
    
    def clear(self) -> None:
        """Clear all data."""
        self._data.clear()
