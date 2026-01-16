"""
Tenant-isolated storage wrapper.

This module provides TenantIsolatedStore which automatically namespaces
all storage operations by the current tenant context.
"""

from typing import Any, List, Optional, Protocol, runtime_checkable

from llmteam.tenancy.context import current_tenant
from llmteam.tenancy.models import TenantContextError


@runtime_checkable
class KeyValueStore(Protocol):
    """
    Protocol for a key-value storage backend.
    
    This is the interface that TenantIsolatedStore wraps.
    """
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value by key."""
        ...
    
    async def set(self, key: str, value: Any) -> None:
        """Set a value for a key."""
        ...
    
    async def delete(self, key: str) -> None:
        """Delete a key."""
        ...
    
    async def list(self, prefix: str = "") -> List[str]:
        """List keys with optional prefix filter."""
        ...
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        ...


class TenantIsolatedStore:
    """
    Storage wrapper that automatically isolates data by tenant.
    
    All keys are automatically prefixed with the current tenant ID,
    ensuring complete data isolation between tenants.
    
    Example:
        inner_store = RedisStore()
        store = TenantIsolatedStore(inner_store)
        
        # With tenant context "acme"
        async with TenantContext("acme"):
            await store.set("key1", "value1")  # Actually stores "acme:key1"
            value = await store.get("key1")    # Actually gets "acme:key1"
        
        # With tenant context "beta"
        async with TenantContext("beta"):
            await store.get("key1")  # Actually gets "beta:key1" - different!
    """
    
    def __init__(
        self, 
        inner_store: KeyValueStore,
        separator: str = ":",
        require_context: bool = True,
    ):
        """
        Initialize TenantIsolatedStore.
        
        Args:
            inner_store: The underlying storage backend
            separator: Character(s) to separate tenant ID from key
            require_context: Whether to require a tenant context (raise error if missing)
        """
        self.inner = inner_store
        self.separator = separator
        self.require_context = require_context
    
    def _get_tenant_id(self) -> str:
        """
        Get the current tenant ID.
        
        Returns:
            Current tenant ID
            
        Raises:
            TenantContextError: If no context is set and require_context is True
        """
        tenant_id = current_tenant.get()
        
        if not tenant_id and self.require_context:
            raise TenantContextError(
                "No tenant context set. Use TenantContext or TenantManager.context() "
                "to set a tenant context before accessing tenant-isolated storage."
            )
        
        return tenant_id or "default"
    
    def _namespace_key(self, key: str) -> str:
        """
        Namespace a key with the current tenant ID.
        
        Args:
            key: The original key
            
        Returns:
            Namespaced key in format "{tenant_id}{separator}{key}"
        """
        tenant_id = self._get_tenant_id()
        return f"{tenant_id}{self.separator}{key}"
    
    def _strip_namespace(self, key: str) -> str:
        """
        Strip the tenant namespace from a key.
        
        Args:
            key: The namespaced key
            
        Returns:
            The original key without namespace
        """
        tenant_id = self._get_tenant_id()
        prefix = f"{tenant_id}{self.separator}"
        
        if key.startswith(prefix):
            return key[len(prefix):]
        return key
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value for the current tenant.
        
        Args:
            key: The key to retrieve
            
        Returns:
            The value or None if not found
        """
        namespaced_key = self._namespace_key(key)
        return await self.inner.get(namespaced_key)
    
    async def set(self, key: str, value: Any) -> None:
        """
        Set a value for the current tenant.
        
        Args:
            key: The key to set
            value: The value to store
        """
        namespaced_key = self._namespace_key(key)
        await self.inner.set(namespaced_key, value)
    
    async def delete(self, key: str) -> None:
        """
        Delete a key for the current tenant.
        
        Args:
            key: The key to delete
        """
        namespaced_key = self._namespace_key(key)
        await self.inner.delete(namespaced_key)
    
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists for the current tenant.
        
        Args:
            key: The key to check
            
        Returns:
            True if key exists
        """
        namespaced_key = self._namespace_key(key)
        return await self.inner.exists(namespaced_key)
    
    async def list(self, prefix: str = "") -> List[str]:
        """
        List keys for the current tenant.
        
        Args:
            prefix: Optional prefix to filter keys
            
        Returns:
            List of keys (with namespace stripped)
        """
        tenant_id = self._get_tenant_id()
        namespaced_prefix = f"{tenant_id}{self.separator}{prefix}"
        
        keys = await self.inner.list(namespaced_prefix)
        
        # Strip the tenant namespace from returned keys
        return [self._strip_namespace(k) for k in keys]
    
    async def get_many(self, keys: List[str]) -> dict[str, Any]:
        """
        Get multiple values for the current tenant.
        
        Args:
            keys: List of keys to retrieve
            
        Returns:
            Dictionary of key -> value (only includes found keys)
        """
        result = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                result[key] = value
        return result
    
    async def set_many(self, items: dict[str, Any]) -> None:
        """
        Set multiple values for the current tenant.
        
        Args:
            items: Dictionary of key -> value to set
        """
        for key, value in items.items():
            await self.set(key, value)
    
    async def delete_many(self, keys: List[str]) -> None:
        """
        Delete multiple keys for the current tenant.
        
        Args:
            keys: List of keys to delete
        """
        for key in keys:
            await self.delete(key)
    
    async def clear_tenant(self) -> int:
        """
        Clear all data for the current tenant.
        
        Returns:
            Number of keys deleted
        """
        keys = await self.list()
        await self.delete_many(keys)
        return len(keys)


class CrossTenantStore:
    """
    Store that allows explicit cross-tenant access for admin operations.
    
    This should only be used for administrative tasks that need to
    access data across multiple tenants (e.g., migrations, analytics).
    
    Example:
        admin_store = CrossTenantStore(inner_store)
        
        # Access specific tenant's data
        value = await admin_store.get_for_tenant("acme", "key1")
        
        # List all tenants' keys
        all_keys = await admin_store.list_all_tenants()
    """
    
    def __init__(self, inner_store: KeyValueStore, separator: str = ":"):
        self.inner = inner_store
        self.separator = separator
    
    async def get_for_tenant(self, tenant_id: str, key: str) -> Optional[Any]:
        """Get a value for a specific tenant."""
        namespaced_key = f"{tenant_id}{self.separator}{key}"
        return await self.inner.get(namespaced_key)
    
    async def set_for_tenant(self, tenant_id: str, key: str, value: Any) -> None:
        """Set a value for a specific tenant."""
        namespaced_key = f"{tenant_id}{self.separator}{key}"
        await self.inner.set(namespaced_key, value)
    
    async def delete_for_tenant(self, tenant_id: str, key: str) -> None:
        """Delete a key for a specific tenant."""
        namespaced_key = f"{tenant_id}{self.separator}{key}"
        await self.inner.delete(namespaced_key)
    
    async def list_for_tenant(self, tenant_id: str, prefix: str = "") -> List[str]:
        """List keys for a specific tenant."""
        namespaced_prefix = f"{tenant_id}{self.separator}{prefix}"
        keys = await self.inner.list(namespaced_prefix)
        
        # Strip namespace
        strip_prefix = f"{tenant_id}{self.separator}"
        return [k[len(strip_prefix):] for k in keys if k.startswith(strip_prefix)]
    
    async def list_all(self, prefix: str = "") -> List[tuple[str, str]]:
        """
        List all keys across all tenants.
        
        Returns:
            List of (tenant_id, key) tuples
        """
        all_keys = await self.inner.list(prefix)
        result = []
        
        for key in all_keys:
            if self.separator in key:
                tenant_id, actual_key = key.split(self.separator, 1)
                result.append((tenant_id, actual_key))
        
        return result
    
    async def copy_between_tenants(
        self, 
        source_tenant: str, 
        target_tenant: str, 
        key: str,
    ) -> bool:
        """
        Copy a value from one tenant to another.
        
        Returns:
            True if copy was successful
        """
        value = await self.get_for_tenant(source_tenant, key)
        if value is None:
            return False
        
        await self.set_for_tenant(target_tenant, key, value)
        return True
