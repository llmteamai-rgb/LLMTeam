"""
HashiCorp Vault Secrets Provider.

Provides secure access to secrets stored in HashiCorp Vault.

Requires:
    pip install hvac

Environment Variables:
    VAULT_ADDR - Vault server URL
    VAULT_TOKEN - Vault authentication token
    VAULT_NAMESPACE - Optional Vault namespace (Enterprise)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from llmteam.secrets.base import (
    SecretsProvider,
    SecretValue,
    SecretMetadata,
    SecretsError,
    SecretNotFoundError,
    SecretAccessDeniedError,
)


@dataclass
class VaultConfig:
    """Configuration for HashiCorp Vault provider."""

    url: str = "http://localhost:8200"
    token: Optional[str] = None
    namespace: Optional[str] = None
    mount_point: str = "secret"
    verify_ssl: bool = True
    timeout: int = 30
    # AppRole auth
    role_id: Optional[str] = None
    secret_id: Optional[str] = None
    # Kubernetes auth
    kubernetes_role: Optional[str] = None
    kubernetes_jwt_path: str = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    # Additional options
    headers: dict[str, str] = field(default_factory=dict)


class VaultProvider(SecretsProvider):
    """
    HashiCorp Vault secrets provider.

    Supports multiple authentication methods:
    - Token authentication
    - AppRole authentication
    - Kubernetes authentication

    Usage:
        # Token auth (simplest)
        config = VaultConfig(
            url="https://vault.example.com:8200",
            token="hvs.xxxxx",
        )
        provider = VaultProvider(config)

        # AppRole auth (recommended for services)
        config = VaultConfig(
            url="https://vault.example.com:8200",
            role_id="xxx",
            secret_id="yyy",
        )
        provider = VaultProvider(config)

        # Get a secret
        secret = await provider.get_secret("myapp/api-key")
    """

    def __init__(self, config: Optional[VaultConfig] = None):
        """
        Initialize Vault provider.

        Args:
            config: Vault configuration (uses environment vars if not provided)
        """
        import os

        self.config = config or VaultConfig()

        # Override with environment variables if not set
        if not self.config.url:
            self.config.url = os.environ.get("VAULT_ADDR", "http://localhost:8200")
        if not self.config.token:
            self.config.token = os.environ.get("VAULT_TOKEN")
        if not self.config.namespace:
            self.config.namespace = os.environ.get("VAULT_NAMESPACE")

        self._client: Any = None

    @property
    def provider_name(self) -> str:
        return "Vault"

    def _get_client(self) -> Any:
        """Get or create the Vault client."""
        if self._client is None:
            try:
                import hvac
            except ImportError:
                raise SecretsError(
                    "hvac is required for Vault. Install with: pip install hvac",
                    provider=self.provider_name,
                )

            self._client = hvac.Client(
                url=self.config.url,
                token=self.config.token,
                namespace=self.config.namespace,
                verify=self.config.verify_ssl,
                timeout=self.config.timeout,
            )

            # Authenticate if needed
            if not self.config.token:
                self._authenticate()

        return self._client

    def _authenticate(self) -> None:
        """Authenticate to Vault using configured method."""
        client = self._client

        # AppRole authentication
        if self.config.role_id and self.config.secret_id:
            response = client.auth.approle.login(
                role_id=self.config.role_id,
                secret_id=self.config.secret_id,
            )
            client.token = response["auth"]["client_token"]

        # Kubernetes authentication
        elif self.config.kubernetes_role:
            try:
                with open(self.config.kubernetes_jwt_path) as f:
                    jwt = f.read().strip()
                response = client.auth.kubernetes.login(
                    role=self.config.kubernetes_role,
                    jwt=jwt,
                )
                client.token = response["auth"]["client_token"]
            except FileNotFoundError:
                raise SecretsError(
                    f"Kubernetes JWT not found at {self.config.kubernetes_jwt_path}",
                    provider=self.provider_name,
                )

    async def get_secret(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> SecretValue:
        """
        Get a secret from Vault.

        Args:
            name: Secret path (e.g., "myapp/api-key")
            version: Optional version number for KV v2

        Returns:
            SecretValue with the secret data
        """
        try:
            client = self._get_client()

            # Read from KV v2 secrets engine
            read_kwargs: dict[str, Any] = {
                "path": name,
                "mount_point": self.config.mount_point,
            }
            if version is not None:
                read_kwargs["version"] = int(version)

            response = client.secrets.kv.v2.read_secret_version(**read_kwargs)

            if response is None or "data" not in response:
                raise SecretNotFoundError(name, self.provider_name)

            data = response["data"]["data"]
            metadata = response["data"]["metadata"]

            # Get the primary value (prefer "value" key, fallback to first key)
            if "value" in data:
                value = data["value"]
            elif len(data) == 1:
                value = list(data.values())[0]
            else:
                # Return as JSON for multiple keys
                import json

                value = json.dumps(data)

            return SecretValue(
                value=str(value),
                metadata=SecretMetadata(
                    name=name,
                    version=str(metadata.get("version")),
                    created_at=_parse_vault_time(metadata.get("created_time")),
                    updated_at=_parse_vault_time(metadata.get("created_time")),
                    provider=self.provider_name,
                ),
            )

        except SecretNotFoundError:
            raise
        except Exception as e:
            error_str = str(e).lower()
            if "permission denied" in error_str or "403" in error_str:
                raise SecretAccessDeniedError(name, self.provider_name)
            elif "not found" in error_str or "404" in error_str:
                raise SecretNotFoundError(name, self.provider_name)
            raise SecretsError(str(e), provider=self.provider_name)

    async def set_secret(
        self,
        name: str,
        value: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SecretMetadata:
        """
        Store a secret in Vault.

        Args:
            name: Secret path
            value: Secret value
            metadata: Optional metadata (stored as custom_metadata in KV v2)

        Returns:
            Metadata about the stored secret
        """
        try:
            client = self._get_client()

            secret_data = {"value": value}
            if metadata and "data" in metadata:
                secret_data.update(metadata["data"])

            response = client.secrets.kv.v2.create_or_update_secret(
                path=name,
                secret=secret_data,
                mount_point=self.config.mount_point,
            )

            version_info = response.get("data", {})

            return SecretMetadata(
                name=name,
                version=str(version_info.get("version")),
                created_at=_parse_vault_time(version_info.get("created_time")),
                provider=self.provider_name,
            )

        except Exception as e:
            error_str = str(e).lower()
            if "permission denied" in error_str or "403" in error_str:
                raise SecretAccessDeniedError(name, self.provider_name)
            raise SecretsError(str(e), provider=self.provider_name)

    async def delete_secret(self, name: str) -> bool:
        """
        Delete a secret from Vault.

        This performs a soft delete (the secret can be recovered).
        Use delete_metadata for permanent deletion.
        """
        try:
            client = self._get_client()

            client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=name,
                mount_point=self.config.mount_point,
            )
            return True

        except Exception as e:
            error_str = str(e).lower()
            if "not found" in error_str or "404" in error_str:
                return False
            if "permission denied" in error_str:
                raise SecretAccessDeniedError(name, self.provider_name)
            raise SecretsError(str(e), provider=self.provider_name)

    async def list_secrets(
        self,
        prefix: Optional[str] = None,
    ) -> list[SecretMetadata]:
        """
        List secrets in Vault.

        Args:
            prefix: Path prefix to list under

        Returns:
            List of secret metadata
        """
        try:
            client = self._get_client()

            path = prefix or ""
            response = client.secrets.kv.v2.list_secrets(
                path=path,
                mount_point=self.config.mount_point,
            )

            keys = response.get("data", {}).get("keys", [])

            results = []
            for key in keys:
                full_path = f"{path}/{key}".lstrip("/") if path else key
                results.append(
                    SecretMetadata(
                        name=full_path.rstrip("/"),
                        provider=self.provider_name,
                    )
                )

            return results

        except Exception as e:
            error_str = str(e).lower()
            if "not found" in error_str or "404" in error_str:
                return []
            if "permission denied" in error_str:
                raise SecretAccessDeniedError(prefix or "/", self.provider_name)
            raise SecretsError(str(e), provider=self.provider_name)


def _parse_vault_time(time_str: Optional[str]) -> Optional[datetime]:
    """Parse Vault timestamp format."""
    if not time_str:
        return None
    try:
        # Vault uses RFC3339 format
        return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
