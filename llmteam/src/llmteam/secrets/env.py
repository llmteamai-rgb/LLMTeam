"""
Environment Variables Secrets Provider.

A simple provider that reads secrets from environment variables.
Useful as a fallback or for local development.
"""

import os
from datetime import datetime
from typing import Any, Optional

from llmteam.secrets.base import (
    SecretsProvider,
    SecretValue,
    SecretMetadata,
    SecretNotFoundError,
    SecretsError,
)


class EnvSecretsProvider(SecretsProvider):
    """
    Environment variables secrets provider.

    Reads secrets from environment variables with optional prefix.

    Usage:
        provider = EnvSecretsProvider(prefix="APP_")

        # Reads from APP_API_KEY environment variable
        secret = await provider.get_secret("API_KEY")

        # Without prefix
        provider = EnvSecretsProvider()
        secret = await provider.get_secret("OPENAI_API_KEY")
    """

    def __init__(
        self,
        prefix: str = "",
        case_transform: Optional[str] = "upper",
    ):
        """
        Initialize environment provider.

        Args:
            prefix: Prefix to add to secret names when looking up
            case_transform: Transform secret names: "upper", "lower", or None
        """
        self.prefix = prefix
        self.case_transform = case_transform

    @property
    def provider_name(self) -> str:
        return "Environment"

    def _get_env_name(self, name: str) -> str:
        """Convert secret name to environment variable name."""
        env_name = f"{self.prefix}{name}"

        # Replace common path separators with underscores
        env_name = env_name.replace("/", "_").replace("-", "_").replace(".", "_")

        if self.case_transform == "upper":
            env_name = env_name.upper()
        elif self.case_transform == "lower":
            env_name = env_name.lower()

        return env_name

    async def get_secret(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> SecretValue:
        """Get secret from environment variable."""
        if version is not None:
            # Environment variables don't support versioning
            raise SecretsError(
                "Environment provider does not support versioning",
                provider=self.provider_name,
            )

        env_name = self._get_env_name(name)
        value = os.environ.get(env_name)

        if value is None:
            raise SecretNotFoundError(name, self.provider_name)

        return SecretValue(
            value=value,
            metadata=SecretMetadata(
                name=name,
                provider=self.provider_name,
            ),
        )

    async def set_secret(
        self,
        name: str,
        value: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SecretMetadata:
        """
        Set environment variable.

        Note: This only sets the variable in the current process.
        It does not persist across restarts.
        """
        env_name = self._get_env_name(name)
        os.environ[env_name] = value

        return SecretMetadata(
            name=name,
            updated_at=datetime.utcnow(),
            provider=self.provider_name,
        )

    async def delete_secret(self, name: str) -> bool:
        """Remove environment variable from current process."""
        env_name = self._get_env_name(name)

        if env_name in os.environ:
            del os.environ[env_name]
            return True

        return False

    async def list_secrets(
        self,
        prefix: Optional[str] = None,
    ) -> list[SecretMetadata]:
        """
        List environment variables.

        Args:
            prefix: Additional prefix to filter by
        """
        results = []
        full_prefix = self.prefix + (prefix or "")

        for env_name in os.environ:
            # Check if matches our prefix
            check_name = env_name
            if self.case_transform == "upper":
                full_prefix_check = full_prefix.upper()
            elif self.case_transform == "lower":
                full_prefix_check = full_prefix.lower()
            else:
                full_prefix_check = full_prefix

            if check_name.startswith(full_prefix_check):
                # Convert back to secret name
                secret_name = env_name[len(self.prefix) :]
                results.append(
                    SecretMetadata(
                        name=secret_name,
                        provider=self.provider_name,
                    )
                )

        return results
