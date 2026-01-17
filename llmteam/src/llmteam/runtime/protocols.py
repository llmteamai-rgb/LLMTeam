"""
Protocols for runtime registries.

These protocols define the interfaces for resources that can be
registered and resolved through RuntimeContext.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Store(Protocol):
    """Protocol for storage backends."""

    async def get(self, key: str) -> Any:
        """Get value by key."""
        ...

    async def set(self, key: str, value: Any) -> None:
        """Set value by key."""
        ...

    async def delete(self, key: str) -> None:
        """Delete value by key."""
        ...


@runtime_checkable
class Client(Protocol):
    """Protocol for external clients (HTTP, gRPC, etc)."""

    async def request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Make a request to the external service."""
        ...


@runtime_checkable
class SecretsProvider(Protocol):
    """Protocol for secrets access."""

    async def get_secret(self, secret_id: str) -> str:
        """Get secret value by ID."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        """Generate completion for prompt."""
        ...
