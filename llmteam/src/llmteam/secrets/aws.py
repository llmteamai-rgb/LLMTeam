"""
AWS Secrets Manager Provider.

Provides secure access to secrets stored in AWS Secrets Manager.

Requires:
    pip install boto3

Environment Variables:
    AWS_ACCESS_KEY_ID - AWS access key
    AWS_SECRET_ACCESS_KEY - AWS secret key
    AWS_DEFAULT_REGION - AWS region
    AWS_SESSION_TOKEN - Optional session token for temporary credentials
"""

import json
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
class AWSSecretsConfig:
    """Configuration for AWS Secrets Manager provider."""

    region_name: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    endpoint_url: Optional[str] = None  # For LocalStack testing
    profile_name: Optional[str] = None
    # Assume role configuration
    role_arn: Optional[str] = None
    role_session_name: str = "llmteam-secrets"
    # Additional options
    client_config: dict[str, Any] = field(default_factory=dict)


class AWSSecretsProvider(SecretsProvider):
    """
    AWS Secrets Manager secrets provider.

    Usage:
        # Using default credentials (IAM role, environment, etc.)
        provider = AWSSecretsProvider(region_name="us-east-1")

        # Using explicit credentials
        config = AWSSecretsConfig(
            region_name="us-east-1",
            aws_access_key_id="AKIA...",
            aws_secret_access_key="...",
        )
        provider = AWSSecretsProvider(config)

        # With assume role
        config = AWSSecretsConfig(
            region_name="us-east-1",
            role_arn="arn:aws:iam::123456789012:role/SecretsRole",
        )
        provider = AWSSecretsProvider(config)

        # Get a secret
        secret = await provider.get_secret("myapp/api-key")
    """

    def __init__(
        self,
        config: Optional[AWSSecretsConfig] = None,
        region_name: Optional[str] = None,
    ):
        """
        Initialize AWS Secrets Manager provider.

        Args:
            config: AWS configuration
            region_name: AWS region (shortcut for simple config)
        """
        if config is None:
            config = AWSSecretsConfig(region_name=region_name)
        self.config = config
        self._client: Any = None

    @property
    def provider_name(self) -> str:
        return "AWSSecretsManager"

    def _get_client(self) -> Any:
        """Get or create the boto3 client."""
        if self._client is not None:
            return self._client

        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            raise SecretsError(
                "boto3 is required for AWS Secrets Manager. "
                "Install with: pip install boto3",
                provider=self.provider_name,
            )

        # Build session kwargs
        session_kwargs: dict[str, Any] = {}
        if self.config.profile_name:
            session_kwargs["profile_name"] = self.config.profile_name
        if self.config.region_name:
            session_kwargs["region_name"] = self.config.region_name

        session = boto3.Session(**session_kwargs)

        # Build client kwargs
        client_kwargs: dict[str, Any] = {}
        if self.config.aws_access_key_id:
            client_kwargs["aws_access_key_id"] = self.config.aws_access_key_id
        if self.config.aws_secret_access_key:
            client_kwargs["aws_secret_access_key"] = self.config.aws_secret_access_key
        if self.config.aws_session_token:
            client_kwargs["aws_session_token"] = self.config.aws_session_token
        if self.config.endpoint_url:
            client_kwargs["endpoint_url"] = self.config.endpoint_url
        if self.config.client_config:
            client_kwargs["config"] = Config(**self.config.client_config)

        # Handle assume role
        if self.config.role_arn:
            sts = session.client("sts", **client_kwargs)
            response = sts.assume_role(
                RoleArn=self.config.role_arn,
                RoleSessionName=self.config.role_session_name,
            )
            credentials = response["Credentials"]
            client_kwargs["aws_access_key_id"] = credentials["AccessKeyId"]
            client_kwargs["aws_secret_access_key"] = credentials["SecretAccessKey"]
            client_kwargs["aws_session_token"] = credentials["SessionToken"]

        self._client = session.client("secretsmanager", **client_kwargs)
        return self._client

    async def get_secret(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> SecretValue:
        """
        Get a secret from AWS Secrets Manager.

        Args:
            name: Secret name or ARN
            version: Optional version ID or staging label

        Returns:
            SecretValue with the secret data
        """
        try:
            client = self._get_client()

            kwargs: dict[str, Any] = {"SecretId": name}
            if version:
                # Could be version ID or staging label
                if version.startswith("AWSCURRENT") or version.startswith("AWSPREVIOUS"):
                    kwargs["VersionStage"] = version
                else:
                    kwargs["VersionId"] = version

            response = client.get_secret_value(**kwargs)

            # Get value (could be string or binary)
            if "SecretString" in response:
                value = response["SecretString"]
                binary = False
                raw_data = None
            else:
                import base64

                raw_data = response["SecretBinary"]
                value = base64.b64encode(raw_data).decode()
                binary = True

            # Try to parse as JSON for structured secrets
            parsed_value = value
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    # If it's a dict with a "value" key, use that
                    if "value" in parsed:
                        parsed_value = str(parsed["value"])
            except (json.JSONDecodeError, TypeError):
                pass

            return SecretValue(
                value=parsed_value,
                metadata=SecretMetadata(
                    name=response.get("Name", name),
                    version=response.get("VersionId"),
                    created_at=response.get("CreatedDate"),
                    tags=_tags_to_dict(response.get("Tags", [])),
                    provider=self.provider_name,
                ),
                binary=binary,
                raw_data=raw_data,
            )

        except Exception as e:
            self._handle_error(e, name)
            raise  # Re-raise if not handled

    async def set_secret(
        self,
        name: str,
        value: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SecretMetadata:
        """
        Store or update a secret in AWS Secrets Manager.

        Args:
            name: Secret name
            value: Secret value (string or JSON)
            metadata: Optional metadata (tags, description, kms_key_id)

        Returns:
            Metadata about the stored secret
        """
        try:
            client = self._get_client()
            metadata = metadata or {}

            # Check if secret exists
            try:
                client.describe_secret(SecretId=name)
                exists = True
            except client.exceptions.ResourceNotFoundException:
                exists = False

            if exists:
                # Update existing secret
                kwargs: dict[str, Any] = {
                    "SecretId": name,
                    "SecretString": value,
                }
                response = client.put_secret_value(**kwargs)
            else:
                # Create new secret
                kwargs = {
                    "Name": name,
                    "SecretString": value,
                }
                if "description" in metadata:
                    kwargs["Description"] = metadata["description"]
                if "kms_key_id" in metadata:
                    kwargs["KmsKeyId"] = metadata["kms_key_id"]
                if "tags" in metadata:
                    kwargs["Tags"] = _dict_to_tags(metadata["tags"])

                response = client.create_secret(**kwargs)

            return SecretMetadata(
                name=response.get("Name", name),
                version=response.get("VersionId"),
                provider=self.provider_name,
            )

        except Exception as e:
            self._handle_error(e, name)
            raise

    async def delete_secret(self, name: str) -> bool:
        """
        Delete a secret from AWS Secrets Manager.

        By default, schedules deletion with 7-day recovery window.
        """
        try:
            client = self._get_client()

            client.delete_secret(
                SecretId=name,
                RecoveryWindowInDays=7,
            )
            return True

        except Exception as e:
            error_type = type(e).__name__
            if "ResourceNotFoundException" in error_type:
                return False
            self._handle_error(e, name)
            raise

    async def list_secrets(
        self,
        prefix: Optional[str] = None,
    ) -> list[SecretMetadata]:
        """
        List secrets in AWS Secrets Manager.

        Args:
            prefix: Optional name prefix to filter by

        Returns:
            List of secret metadata
        """
        try:
            client = self._get_client()

            results = []
            paginator = client.get_paginator("list_secrets")

            filters = []
            if prefix:
                filters.append({"Key": "name", "Values": [prefix]})

            page_kwargs: dict[str, Any] = {}
            if filters:
                page_kwargs["Filters"] = filters

            for page in paginator.paginate(**page_kwargs):
                for secret in page.get("SecretList", []):
                    results.append(
                        SecretMetadata(
                            name=secret["Name"],
                            created_at=secret.get("CreatedDate"),
                            updated_at=secret.get("LastChangedDate"),
                            tags=_tags_to_dict(secret.get("Tags", [])),
                            provider=self.provider_name,
                        )
                    )

            return results

        except Exception as e:
            self._handle_error(e, prefix or "/")
            raise

    def _handle_error(self, error: Exception, name: str) -> None:
        """Handle AWS errors and convert to standard exceptions."""
        error_type = type(error).__name__
        error_str = str(error).lower()

        if "ResourceNotFoundException" in error_type or "not found" in error_str:
            raise SecretNotFoundError(name, self.provider_name)
        elif (
            "AccessDeniedException" in error_type
            or "UnauthorizedException" in error_type
            or "access denied" in error_str
        ):
            raise SecretAccessDeniedError(name, self.provider_name)
        else:
            raise SecretsError(str(error), provider=self.provider_name)


def _tags_to_dict(tags: list[dict[str, str]]) -> dict[str, str]:
    """Convert AWS tags list to dict."""
    return {tag["Key"]: tag["Value"] for tag in tags}


def _dict_to_tags(d: dict[str, str]) -> list[dict[str, str]]:
    """Convert dict to AWS tags list."""
    return [{"Key": k, "Value": v} for k, v in d.items()]
