"""
Tenant context management using context variables.

This module provides:
- current_tenant: A context variable holding the current tenant ID
- TenantContext: A context manager for setting the tenant context

Usage:
    from llmteam.tenancy import TenantContext, current_tenant
    
    # Set tenant context
    async with TenantContext("acme_corp"):
        # All operations here are in context of acme_corp
        tenant_id = current_tenant.get()  # Returns "acme_corp"
        
    # Outside context
    tenant_id = current_tenant.get()  # Returns "" (default)
"""

from contextvars import ContextVar, Token
from typing import Optional, Any
from dataclasses import dataclass

from llmteam.tenancy.models import TenantConfig


# Context variable for current tenant ID
# This is thread-safe and async-safe
current_tenant: ContextVar[str] = ContextVar("current_tenant", default="")


def get_current_tenant() -> str:
    """
    Get the current tenant ID.
    
    Returns:
        Current tenant ID or empty string if not set.
    """
    return current_tenant.get()


def require_tenant_context() -> str:
    """
    Get the current tenant ID, raising an error if not set.
    
    Returns:
        Current tenant ID
        
    Raises:
        TenantContextError: If no tenant context is set
    """
    from llmteam.tenancy.models import TenantContextError
    
    tenant_id = current_tenant.get()
    if not tenant_id:
        raise TenantContextError("No tenant context is set. Use TenantContext to set one.")
    return tenant_id


@dataclass
class TenantContext:
    """
    Context manager for setting the current tenant.
    
    This ensures that all operations within the context are associated
    with the specified tenant. Works with both sync and async code.
    
    Attributes:
        tenant_id: The tenant ID to set
        config: Optional tenant configuration (for caching)
    
    Example:
        # Sync usage
        with TenantContext("acme"):
            do_something()
        
        # Async usage
        async with TenantContext("acme"):
            await do_something_async()
    """
    
    tenant_id: str
    config: Optional[TenantConfig] = None
    
    _token: Optional[Token[str]] = None
    
    def __post_init__(self):
        if not self.tenant_id:
            raise ValueError("tenant_id cannot be empty")
    
    def __enter__(self) -> "TenantContext":
        """Enter the context (sync)."""
        self._token = current_tenant.set(self.tenant_id)
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context (sync)."""
        if self._token is not None:
            current_tenant.reset(self._token)
            self._token = None
    
    async def __aenter__(self) -> "TenantContext":
        """Enter the context (async)."""
        return self.__enter__()
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context (async)."""
        self.__exit__(exc_type, exc_val, exc_tb)
    
    def get_config(self) -> Optional[TenantConfig]:
        """Get the cached tenant configuration."""
        return self.config
    
    def is_active(self) -> bool:
        """Check if this context is currently active."""
        return current_tenant.get() == self.tenant_id


class TenantContextStack:
    """
    A stack of tenant contexts for nested operations.
    
    This is useful when you need to temporarily switch tenants
    while maintaining the ability to return to the previous context.
    
    Example:
        stack = TenantContextStack()
        
        stack.push("tenant_a")
        # Operations for tenant_a
        
        stack.push("tenant_b")
        # Operations for tenant_b
        
        stack.pop()  # Back to tenant_a
        stack.pop()  # Back to no tenant
    """
    
    def __init__(self):
        self._stack: list[TenantContext] = []
    
    def push(self, tenant_id: str, config: Optional[TenantConfig] = None) -> TenantContext:
        """
        Push a new tenant context onto the stack.
        
        Args:
            tenant_id: The tenant ID to set
            config: Optional tenant configuration
            
        Returns:
            The created TenantContext
        """
        ctx = TenantContext(tenant_id=tenant_id, config=config)
        ctx.__enter__()
        self._stack.append(ctx)
        return ctx
    
    def pop(self) -> Optional[TenantContext]:
        """
        Pop the current tenant context from the stack.
        
        Returns:
            The popped TenantContext or None if stack is empty
        """
        if not self._stack:
            return None
        
        ctx = self._stack.pop()
        ctx.__exit__(None, None, None)
        return ctx
    
    def current(self) -> Optional[TenantContext]:
        """Get the current tenant context without removing it."""
        return self._stack[-1] if self._stack else None
    
    def depth(self) -> int:
        """Get the current stack depth."""
        return len(self._stack)
    
    def clear(self) -> None:
        """Clear all contexts from the stack."""
        while self._stack:
            self.pop()
