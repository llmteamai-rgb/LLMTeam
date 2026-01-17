"""
API Key Authentication.

Provides API key validation and management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, Protocol
import asyncio
import hashlib
import secrets


@dataclass
class APIKeyConfig:
    """Configuration for API key validation."""

    # Header/query parameter names
    header_name: str = "X-API-Key"
    query_param: str = "api_key"

    # Hash algorithm for stored keys
    hash_algorithm: str = "sha256"

    # Key prefix for identification
    key_prefix: str = "llmt_"

    # Key length (excluding prefix)
    key_length: int = 32


@dataclass
class APIKeyInfo:
    """Information about an API key."""

    key_id: str
    name: str
    tenant_id: str

    # Permissions
    scopes: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None

    # Rate limiting
    rate_limit: Optional[int] = None  # requests per minute

    # Custom metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if key is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() >= self.expires_at


class APIKeyStore(Protocol):
    """Protocol for API key storage."""

    async def get_key_info(self, key_hash: str) -> Optional[APIKeyInfo]:
        """Get key info by hash."""
        ...

    async def save_key_info(self, key_hash: str, info: APIKeyInfo) -> None:
        """Save key info."""
        ...

    async def delete_key(self, key_hash: str) -> bool:
        """Delete a key."""
        ...

    async def update_last_used(self, key_hash: str) -> None:
        """Update last used timestamp."""
        ...


class MemoryAPIKeyStore:
    """In-memory API key store for testing."""

    def __init__(self) -> None:
        self._keys: dict[str, APIKeyInfo] = {}
        self._lock = asyncio.Lock()

    async def get_key_info(self, key_hash: str) -> Optional[APIKeyInfo]:
        """Get key info by hash."""
        return self._keys.get(key_hash)

    async def save_key_info(self, key_hash: str, info: APIKeyInfo) -> None:
        """Save key info."""
        async with self._lock:
            self._keys[key_hash] = info

    async def delete_key(self, key_hash: str) -> bool:
        """Delete a key."""
        async with self._lock:
            if key_hash in self._keys:
                del self._keys[key_hash]
                return True
            return False

    async def update_last_used(self, key_hash: str) -> None:
        """Update last used timestamp."""
        async with self._lock:
            if key_hash in self._keys:
                self._keys[key_hash].last_used_at = datetime.now()


class APIKeyValidator:
    """
    API key validator.

    Validates API keys and retrieves associated permissions.

    Usage:
        store = MemoryAPIKeyStore()
        validator = APIKeyValidator(store=store)

        # Create a key
        key, info = await validator.create_key(
            name="My API Key",
            tenant_id="tenant-1",
            scopes=["read", "write"],
        )
        print(f"API Key: {key}")  # llmt_abc123...

        # Validate a key
        info = await validator.validate(key)
        if info:
            print(f"Valid key for tenant: {info.tenant_id}")
    """

    def __init__(
        self,
        store: Optional[APIKeyStore] = None,
        config: Optional[APIKeyConfig] = None,
    ) -> None:
        """
        Initialize API key validator.

        Args:
            store: API key storage (default: in-memory)
            config: API key configuration
        """
        self.store = store or MemoryAPIKeyStore()
        self.config = config or APIKeyConfig()

    def _hash_key(self, key: str) -> str:
        """Hash an API key for storage."""
        if self.config.hash_algorithm == "sha256":
            return hashlib.sha256(key.encode()).hexdigest()
        elif self.config.hash_algorithm == "sha512":
            return hashlib.sha512(key.encode()).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {self.config.hash_algorithm}")

    def generate_key(self) -> str:
        """Generate a new API key."""
        random_part = secrets.token_urlsafe(self.config.key_length)
        return f"{self.config.key_prefix}{random_part}"

    async def create_key(
        self,
        name: str,
        tenant_id: str,
        scopes: Optional[list[str]] = None,
        permissions: Optional[list[str]] = None,
        expires_at: Optional[datetime] = None,
        rate_limit: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[str, APIKeyInfo]:
        """
        Create a new API key.

        Args:
            name: Human-readable name for the key
            tenant_id: Tenant this key belongs to
            scopes: OAuth-style scopes
            permissions: Fine-grained permissions
            expires_at: Optional expiration datetime
            rate_limit: Optional rate limit (requests/minute)
            metadata: Optional custom metadata

        Returns:
            Tuple of (api_key, APIKeyInfo)
            Note: The raw API key is only returned once!
        """
        key = self.generate_key()
        key_hash = self._hash_key(key)

        info = APIKeyInfo(
            key_id=key_hash[:16],  # Short ID for reference
            name=name,
            tenant_id=tenant_id,
            scopes=scopes or [],
            permissions=permissions or [],
            created_at=datetime.now(),
            expires_at=expires_at,
            rate_limit=rate_limit,
            metadata=metadata or {},
        )

        await self.store.save_key_info(key_hash, info)

        return key, info

    async def validate(
        self,
        key: str,
        update_last_used: bool = True,
    ) -> Optional[APIKeyInfo]:
        """
        Validate an API key.

        Args:
            key: API key to validate
            update_last_used: Whether to update last used timestamp

        Returns:
            APIKeyInfo if valid, None if invalid
        """
        # Check prefix
        if not key.startswith(self.config.key_prefix):
            return None

        key_hash = self._hash_key(key)
        info = await self.store.get_key_info(key_hash)

        if info is None:
            return None

        # Check expiration
        if info.is_expired:
            return None

        # Update last used
        if update_last_used:
            await self.store.update_last_used(key_hash)

        return info

    async def revoke(self, key: str) -> bool:
        """
        Revoke an API key.

        Args:
            key: API key to revoke

        Returns:
            True if revoked, False if not found
        """
        key_hash = self._hash_key(key)
        return await self.store.delete_key(key_hash)

    async def revoke_by_id(self, key_id: str) -> bool:
        """
        Revoke an API key by its ID.

        Note: This requires iterating through all keys.
        For production, use a store that supports lookup by ID.

        Args:
            key_id: Key ID to revoke

        Returns:
            True if revoked, False if not found
        """
        # This is inefficient for large key sets
        # Production implementations should use a proper store
        if isinstance(self.store, MemoryAPIKeyStore):
            for key_hash, info in list(self.store._keys.items()):
                if info.key_id == key_id:
                    return await self.store.delete_key(key_hash)
        return False

    def extract_key_from_request(
        self,
        headers: dict[str, str],
        query_params: Optional[dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Extract API key from request headers or query params.

        Args:
            headers: Request headers (case-insensitive)
            query_params: Optional query parameters

        Returns:
            API key if found, None otherwise
        """
        # Check header (case-insensitive)
        header_lower = self.config.header_name.lower()
        for name, value in headers.items():
            if name.lower() == header_lower:
                return value

        # Check query params
        if query_params and self.config.query_param in query_params:
            return query_params[self.config.query_param]

        return None
