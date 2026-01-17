"""
Secrets Management module.

Provides secure access to secrets from various backends:
- HashiCorp Vault
- AWS Secrets Manager
- Azure Key Vault
- Environment variables (fallback)

Usage:
    from llmteam.secrets import SecretsManager, VaultProvider

    # Using HashiCorp Vault
    vault = VaultProvider(url="https://vault.example.com:8200")
    manager = SecretsManager(provider=vault)

    # Get a secret
    api_key = await manager.get_secret("openai/api-key")

    # Using AWS Secrets Manager
    from llmteam.secrets import AWSSecretsProvider
    aws = AWSSecretsProvider(region_name="us-east-1")
    manager = SecretsManager(provider=aws)
"""

from llmteam.secrets.base import (
    SecretsProvider,
    SecretValue,
    SecretMetadata,
    SecretsError,
    SecretNotFoundError,
    SecretAccessDeniedError,
)

from llmteam.secrets.manager import (
    SecretsManager,
    CachingSecretsManager,
    SecretsCacheConfig,
)

from llmteam.secrets.env import EnvSecretsProvider
from llmteam.secrets.vault import VaultProvider, VaultConfig
from llmteam.secrets.aws import AWSSecretsProvider, AWSSecretsConfig
from llmteam.secrets.azure import AzureKeyVaultProvider, AzureKeyVaultConfig

__all__ = [
    # Base
    "SecretsProvider",
    "SecretValue",
    "SecretMetadata",
    "SecretsError",
    "SecretNotFoundError",
    "SecretAccessDeniedError",
    # Manager
    "SecretsManager",
    "CachingSecretsManager",
    "SecretsCacheConfig",
    # Providers
    "EnvSecretsProvider",
    "VaultProvider",
    "VaultConfig",
    "AWSSecretsProvider",
    "AWSSecretsConfig",
    "AzureKeyVaultProvider",
    "AzureKeyVaultConfig",
]
