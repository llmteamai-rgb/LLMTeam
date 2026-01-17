"""
Secrets Manager with caching and multi-provider support.

Provides a high-level interface for managing secrets across
multiple backends with optional caching.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from llmteam.secrets.base import (
    SecretsProvider,
    SecretValue,
    SecretMetadata,
    SecretsError,
    SecretNotFoundError,
)


@dataclass
class SecretsCacheConfig:
    """Configuration for secrets caching."""

    enabled: bool = True
    ttl_seconds: int = 300  # 5 minutes default
    max_entries: int = 1000
    refresh_on_access: bool = False


@dataclass
class CachedSecret:
    """A cached secret entry."""

    secret: SecretValue
    cached_at: datetime
    expires_at: datetime
    access_count: int = 0


class SecretsManager:
    """
    High-level secrets manager.

    Provides a unified interface for accessing secrets from
    one or more providers.

    Usage:
        manager = SecretsManager(provider=vault_provider)

        # Get a secret
        api_key = await manager.get("openai-api-key")

        # Get with default
        value = await manager.get("optional-key", default="fallback")

        # Check existence
        if await manager.exists("my-secret"):
            ...
    """

    def __init__(
        self,
        provider: SecretsProvider,
        fallback_providers: Optional[list[SecretsProvider]] = None,
    ):
        """
        Initialize secrets manager.

        Args:
            provider: Primary secrets provider
            fallback_providers: Optional fallback providers to try
        """
        self.provider = provider
        self.fallback_providers = fallback_providers or []

    async def get(
        self,
        name: str,
        version: Optional[str] = None,
        default: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get a secret value.

        Args:
            name: Secret name/path
            version: Optional version
            default: Default value if not found

        Returns:
            Secret value or default
        """
        try:
            secret = await self.get_secret(name, version)
            return secret.value
        except SecretNotFoundError:
            return default

    async def get_secret(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> SecretValue:
        """
        Get a secret with metadata.

        Args:
            name: Secret name/path
            version: Optional version

        Returns:
            SecretValue with value and metadata

        Raises:
            SecretNotFoundError: If not found in any provider
        """
        # Try primary provider
        try:
            return await self.provider.get_secret(name, version)
        except SecretNotFoundError:
            pass
        except SecretsError:
            # Log but continue to fallbacks
            pass

        # Try fallback providers
        for fallback in self.fallback_providers:
            try:
                return await fallback.get_secret(name, version)
            except SecretNotFoundError:
                continue
            except SecretsError:
                continue

        raise SecretNotFoundError(name, self.provider.provider_name)

    async def set(
        self,
        name: str,
        value: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SecretMetadata:
        """
        Store a secret.

        Args:
            name: Secret name/path
            value: Secret value
            metadata: Optional metadata

        Returns:
            Metadata about stored secret
        """
        return await self.provider.set_secret(name, value, metadata)

    async def delete(self, name: str) -> bool:
        """
        Delete a secret.

        Args:
            name: Secret name/path

        Returns:
            True if deleted
        """
        return await self.provider.delete_secret(name)

    async def exists(self, name: str) -> bool:
        """
        Check if a secret exists.

        Args:
            name: Secret name/path

        Returns:
            True if exists in any provider
        """
        if await self.provider.secret_exists(name):
            return True

        for fallback in self.fallback_providers:
            if await fallback.secret_exists(name):
                return True

        return False

    async def list(
        self,
        prefix: Optional[str] = None,
    ) -> list[SecretMetadata]:
        """
        List secrets from primary provider.

        Args:
            prefix: Optional prefix filter

        Returns:
            List of secret metadata
        """
        return await self.provider.list_secrets(prefix)


class CachingSecretsManager(SecretsManager):
    """
    Secrets manager with caching support.

    Caches secrets in memory to reduce backend calls.
    Supports TTL-based expiration.

    Usage:
        config = SecretsCacheConfig(ttl_seconds=600)
        manager = CachingSecretsManager(
            provider=vault_provider,
            cache_config=config,
        )

        # First call hits backend
        value = await manager.get("api-key")

        # Second call uses cache
        value = await manager.get("api-key")

        # Force refresh
        manager.invalidate("api-key")
    """

    def __init__(
        self,
        provider: SecretsProvider,
        cache_config: Optional[SecretsCacheConfig] = None,
        fallback_providers: Optional[list[SecretsProvider]] = None,
    ):
        """
        Initialize caching secrets manager.

        Args:
            provider: Primary secrets provider
            cache_config: Cache configuration
            fallback_providers: Optional fallback providers
        """
        super().__init__(provider, fallback_providers)
        self.cache_config = cache_config or SecretsCacheConfig()
        self._cache: dict[str, CachedSecret] = {}
        self._lock = asyncio.Lock()

    async def get_secret(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> SecretValue:
        """Get a secret, using cache if enabled."""
        if not self.cache_config.enabled:
            return await super().get_secret(name, version)

        cache_key = f"{name}:{version or 'latest'}"

        async with self._lock:
            # Check cache
            cached = self._cache.get(cache_key)
            if cached and datetime.utcnow() < cached.expires_at:
                cached.access_count += 1
                return cached.secret

        # Cache miss or expired - fetch from backend
        secret = await super().get_secret(name, version)

        async with self._lock:
            # Update cache
            now = datetime.utcnow()
            self._cache[cache_key] = CachedSecret(
                secret=secret,
                cached_at=now,
                expires_at=now + timedelta(seconds=self.cache_config.ttl_seconds),
            )

            # Evict if over limit
            if len(self._cache) > self.cache_config.max_entries:
                self._evict_oldest()

        return secret

    async def set(
        self,
        name: str,
        value: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SecretMetadata:
        """Store a secret and invalidate cache."""
        result = await super().set(name, value, metadata)

        # Invalidate cache for this secret
        async with self._lock:
            keys_to_remove = [k for k in self._cache if k.startswith(f"{name}:")]
            for key in keys_to_remove:
                del self._cache[key]

        return result

    async def delete(self, name: str) -> bool:
        """Delete a secret and invalidate cache."""
        result = await super().delete(name)

        # Invalidate cache
        async with self._lock:
            keys_to_remove = [k for k in self._cache if k.startswith(f"{name}:")]
            for key in keys_to_remove:
                del self._cache[key]

        return result

    def invalidate(self, name: Optional[str] = None) -> int:
        """
        Invalidate cached secrets.

        Args:
            name: Specific secret to invalidate, or None for all

        Returns:
            Number of entries invalidated
        """
        if name is None:
            count = len(self._cache)
            self._cache.clear()
            return count

        keys_to_remove = [k for k in self._cache if k.startswith(f"{name}:")]
        for key in keys_to_remove:
            del self._cache[key]
        return len(keys_to_remove)

    def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        now = datetime.utcnow()
        valid_entries = sum(1 for c in self._cache.values() if now < c.expires_at)
        total_accesses = sum(c.access_count for c in self._cache.values())

        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self._cache) - valid_entries,
            "total_accesses": total_accesses,
            "max_entries": self.cache_config.max_entries,
            "ttl_seconds": self.cache_config.ttl_seconds,
        }

    def _evict_oldest(self) -> None:
        """Evict the oldest cache entry."""
        if not self._cache:
            return

        oldest_key = min(self._cache, key=lambda k: self._cache[k].cached_at)
        del self._cache[oldest_key]
