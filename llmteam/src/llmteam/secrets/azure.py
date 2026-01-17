"""
Azure Key Vault Secrets Provider.

Provides secure access to secrets stored in Azure Key Vault.

Requires:
    pip install azure-keyvault-secrets azure-identity

Environment Variables:
    AZURE_KEY_VAULT_URL - Key Vault URL (e.g., https://myvault.vault.azure.net/)
    AZURE_CLIENT_ID - Service principal client ID
    AZURE_CLIENT_SECRET - Service principal secret
    AZURE_TENANT_ID - Azure AD tenant ID
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
class AzureKeyVaultConfig:
    """Configuration for Azure Key Vault provider."""

    vault_url: str = ""
    # Service principal credentials
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    tenant_id: Optional[str] = None
    # Managed identity
    use_managed_identity: bool = False
    managed_identity_client_id: Optional[str] = None
    # Additional options
    credential_kwargs: dict[str, Any] = field(default_factory=dict)


class AzureKeyVaultProvider(SecretsProvider):
    """
    Azure Key Vault secrets provider.

    Supports multiple authentication methods:
    - Service principal (client_id/client_secret)
    - Managed identity
    - Default Azure credential chain

    Usage:
        # Using default credential chain (recommended)
        config = AzureKeyVaultConfig(
            vault_url="https://myvault.vault.azure.net/",
        )
        provider = AzureKeyVaultProvider(config)

        # Using service principal
        config = AzureKeyVaultConfig(
            vault_url="https://myvault.vault.azure.net/",
            client_id="xxx",
            client_secret="yyy",
            tenant_id="zzz",
        )
        provider = AzureKeyVaultProvider(config)

        # Using managed identity
        config = AzureKeyVaultConfig(
            vault_url="https://myvault.vault.azure.net/",
            use_managed_identity=True,
        )
        provider = AzureKeyVaultProvider(config)

        # Get a secret
        secret = await provider.get_secret("my-api-key")
    """

    def __init__(self, config: Optional[AzureKeyVaultConfig] = None):
        """
        Initialize Azure Key Vault provider.

        Args:
            config: Azure configuration (uses environment vars if not provided)
        """
        import os

        self.config = config or AzureKeyVaultConfig()

        # Override with environment variables if not set
        if not self.config.vault_url:
            self.config.vault_url = os.environ.get("AZURE_KEY_VAULT_URL", "")
        if not self.config.client_id:
            self.config.client_id = os.environ.get("AZURE_CLIENT_ID")
        if not self.config.client_secret:
            self.config.client_secret = os.environ.get("AZURE_CLIENT_SECRET")
        if not self.config.tenant_id:
            self.config.tenant_id = os.environ.get("AZURE_TENANT_ID")

        self._client: Any = None
        self._credential: Any = None

    @property
    def provider_name(self) -> str:
        return "AzureKeyVault"

    def _get_credential(self) -> Any:
        """Get or create Azure credential."""
        if self._credential is not None:
            return self._credential

        try:
            from azure.identity import (
                DefaultAzureCredential,
                ClientSecretCredential,
                ManagedIdentityCredential,
            )
        except ImportError:
            raise SecretsError(
                "azure-identity is required for Azure Key Vault. "
                "Install with: pip install azure-identity",
                provider=self.provider_name,
            )

        if self.config.client_id and self.config.client_secret and self.config.tenant_id:
            # Service principal authentication
            self._credential = ClientSecretCredential(
                tenant_id=self.config.tenant_id,
                client_id=self.config.client_id,
                client_secret=self.config.client_secret,
                **self.config.credential_kwargs,
            )
        elif self.config.use_managed_identity:
            # Managed identity authentication
            kwargs = dict(self.config.credential_kwargs)
            if self.config.managed_identity_client_id:
                kwargs["client_id"] = self.config.managed_identity_client_id
            self._credential = ManagedIdentityCredential(**kwargs)
        else:
            # Default credential chain
            self._credential = DefaultAzureCredential(**self.config.credential_kwargs)

        return self._credential

    def _get_client(self) -> Any:
        """Get or create the Key Vault client."""
        if self._client is not None:
            return self._client

        try:
            from azure.keyvault.secrets import SecretClient
        except ImportError:
            raise SecretsError(
                "azure-keyvault-secrets is required for Azure Key Vault. "
                "Install with: pip install azure-keyvault-secrets",
                provider=self.provider_name,
            )

        if not self.config.vault_url:
            raise SecretsError(
                "vault_url is required. Set AZURE_KEY_VAULT_URL or provide in config.",
                provider=self.provider_name,
            )

        credential = self._get_credential()
        self._client = SecretClient(
            vault_url=self.config.vault_url,
            credential=credential,
        )

        return self._client

    async def get_secret(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> SecretValue:
        """
        Get a secret from Azure Key Vault.

        Args:
            name: Secret name
            version: Optional version ID

        Returns:
            SecretValue with the secret data
        """
        try:
            client = self._get_client()

            # Azure Key Vault uses sync API, wrap in executor
            import asyncio

            loop = asyncio.get_event_loop()
            secret = await loop.run_in_executor(
                None,
                lambda: client.get_secret(name, version=version),
            )

            # Parse properties
            props = secret.properties

            return SecretValue(
                value=secret.value or "",
                metadata=SecretMetadata(
                    name=props.name,
                    version=props.version,
                    created_at=props.created_on,
                    updated_at=props.updated_on,
                    expires_at=props.expires_on,
                    tags=props.tags or {},
                    provider=self.provider_name,
                ),
            )

        except Exception as e:
            self._handle_error(e, name)
            raise

    async def set_secret(
        self,
        name: str,
        value: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SecretMetadata:
        """
        Store a secret in Azure Key Vault.

        Args:
            name: Secret name
            value: Secret value
            metadata: Optional metadata (tags, expires_on, content_type)

        Returns:
            Metadata about the stored secret
        """
        try:
            client = self._get_client()
            metadata = metadata or {}

            import asyncio

            kwargs: dict[str, Any] = {}
            if "tags" in metadata:
                kwargs["tags"] = metadata["tags"]
            if "content_type" in metadata:
                kwargs["content_type"] = metadata["content_type"]
            if "expires_on" in metadata:
                kwargs["expires_on"] = metadata["expires_on"]

            loop = asyncio.get_event_loop()
            secret = await loop.run_in_executor(
                None,
                lambda: client.set_secret(name, value, **kwargs),
            )

            props = secret.properties

            return SecretMetadata(
                name=props.name,
                version=props.version,
                created_at=props.created_on,
                updated_at=props.updated_on,
                expires_at=props.expires_on,
                tags=props.tags or {},
                provider=self.provider_name,
            )

        except Exception as e:
            self._handle_error(e, name)
            raise

    async def delete_secret(self, name: str) -> bool:
        """
        Delete a secret from Azure Key Vault.

        This initiates soft-delete. The secret can be recovered
        during the retention period.
        """
        try:
            client = self._get_client()

            import asyncio

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: client.begin_delete_secret(name).wait(),
            )
            return True

        except Exception as e:
            error_type = type(e).__name__
            if "ResourceNotFoundError" in error_type:
                return False
            self._handle_error(e, name)
            raise

    async def list_secrets(
        self,
        prefix: Optional[str] = None,
    ) -> list[SecretMetadata]:
        """
        List secrets in Azure Key Vault.

        Args:
            prefix: Optional name prefix to filter by (client-side filter)

        Returns:
            List of secret metadata
        """
        try:
            client = self._get_client()

            import asyncio

            loop = asyncio.get_event_loop()
            secrets = await loop.run_in_executor(
                None,
                lambda: list(client.list_properties_of_secrets()),
            )

            results = []
            for props in secrets:
                # Apply prefix filter client-side
                if prefix and not props.name.startswith(prefix):
                    continue

                results.append(
                    SecretMetadata(
                        name=props.name,
                        version=props.version,
                        created_at=props.created_on,
                        updated_at=props.updated_on,
                        expires_at=props.expires_on,
                        tags=props.tags or {},
                        provider=self.provider_name,
                    )
                )

            return results

        except Exception as e:
            self._handle_error(e, prefix or "/")
            raise

    def _handle_error(self, error: Exception, name: str) -> None:
        """Handle Azure errors and convert to standard exceptions."""
        error_type = type(error).__name__
        error_str = str(error).lower()

        if "ResourceNotFoundError" in error_type or "not found" in error_str:
            raise SecretNotFoundError(name, self.provider_name)
        elif (
            "ClientAuthenticationError" in error_type
            or "HttpResponseError" in error_type
            and ("forbidden" in error_str or "unauthorized" in error_str)
        ):
            raise SecretAccessDeniedError(name, self.provider_name)
        else:
            raise SecretsError(str(error), provider=self.provider_name)
