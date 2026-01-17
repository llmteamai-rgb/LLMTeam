"""
Base classes for secrets management.

Defines the abstract interface that all secret providers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


class SecretsError(Exception):
    """Base exception for secrets-related errors."""

    def __init__(self, message: str, provider: Optional[str] = None):
        self.provider = provider
        super().__init__(f"[{provider}] {message}" if provider else message)


class SecretNotFoundError(SecretsError):
    """Raised when a secret is not found."""

    def __init__(self, secret_name: str, provider: Optional[str] = None):
        self.secret_name = secret_name
        super().__init__(f"Secret not found: {secret_name}", provider)


class SecretAccessDeniedError(SecretsError):
    """Raised when access to a secret is denied."""

    def __init__(self, secret_name: str, provider: Optional[str] = None):
        self.secret_name = secret_name
        super().__init__(f"Access denied for secret: {secret_name}", provider)


@dataclass
class SecretMetadata:
    """Metadata about a secret."""

    name: str
    version: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    tags: dict[str, str] = field(default_factory=dict)
    provider: Optional[str] = None


@dataclass
class SecretValue:
    """A secret value with its metadata."""

    value: str
    metadata: SecretMetadata
    binary: bool = False
    raw_data: Optional[bytes] = None

    @property
    def is_expired(self) -> bool:
        """Check if the secret has expired."""
        if self.metadata.expires_at is None:
            return False
        return datetime.utcnow() > self.metadata.expires_at


class SecretsProvider(ABC):
    """
    Abstract base class for secret providers.

    All secret backends must implement this interface.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider."""
        pass

    @abstractmethod
    async def get_secret(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> SecretValue:
        """
        Retrieve a secret by name.

        Args:
            name: The secret name/path
            version: Optional specific version

        Returns:
            SecretValue containing the secret and metadata

        Raises:
            SecretNotFoundError: If secret doesn't exist
            SecretAccessDeniedError: If access is denied
            SecretsError: For other errors
        """
        pass

    @abstractmethod
    async def set_secret(
        self,
        name: str,
        value: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SecretMetadata:
        """
        Store or update a secret.

        Args:
            name: The secret name/path
            value: The secret value
            metadata: Optional metadata (tags, expiration, etc.)

        Returns:
            Metadata about the stored secret

        Raises:
            SecretAccessDeniedError: If write access is denied
            SecretsError: For other errors
        """
        pass

    @abstractmethod
    async def delete_secret(self, name: str) -> bool:
        """
        Delete a secret.

        Args:
            name: The secret name/path

        Returns:
            True if deleted, False if not found

        Raises:
            SecretAccessDeniedError: If delete access is denied
            SecretsError: For other errors
        """
        pass

    @abstractmethod
    async def list_secrets(
        self,
        prefix: Optional[str] = None,
    ) -> list[SecretMetadata]:
        """
        List available secrets.

        Args:
            prefix: Optional prefix/path to filter by

        Returns:
            List of secret metadata (values not included)

        Raises:
            SecretAccessDeniedError: If list access is denied
            SecretsError: For other errors
        """
        pass

    async def secret_exists(self, name: str) -> bool:
        """
        Check if a secret exists.

        Args:
            name: The secret name/path

        Returns:
            True if secret exists
        """
        try:
            await self.get_secret(name)
            return True
        except SecretNotFoundError:
            return False

    async def get_secret_value(
        self,
        name: str,
        default: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get just the secret value, with optional default.

        Args:
            name: The secret name/path
            default: Default value if secret not found

        Returns:
            The secret value or default
        """
        try:
            secret = await self.get_secret(name)
            return secret.value
        except SecretNotFoundError:
            return default
