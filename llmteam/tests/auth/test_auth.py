"""Tests for auth module."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from llmteam.auth import (
    OIDCConfig,
    OIDCProvider,
    JWTConfig,
    JWTValidator,
    JWTClaims,
    JWTValidationError,
    APIKeyConfig,
    APIKeyValidator,
    APIKeyInfo,
    Permission,
    Role,
    RBACConfig,
)


class TestOIDCConfig:
    """Tests for OIDCConfig."""

    def test_default_values(self):
        """Config has sensible defaults."""
        config = OIDCConfig()

        assert config.scopes == ["openid", "profile", "email"]
        assert config.use_pkce is True
        assert config.token_refresh_buffer_seconds == 60

    def test_discovery_url_generated(self):
        """Discovery URL is generated from issuer."""
        config = OIDCConfig(issuer="https://auth.example.com")

        assert config.discovery_url == "https://auth.example.com/.well-known/openid-configuration"

    def test_custom_values(self):
        """Custom values are respected."""
        config = OIDCConfig(
            issuer="https://auth.example.com",
            client_id="my-client",
            client_secret="secret",
            scopes=["openid", "custom"],
        )

        assert config.issuer == "https://auth.example.com"
        assert config.client_id == "my-client"
        assert config.scopes == ["openid", "custom"]


class TestOIDCProvider:
    """Tests for OIDCProvider."""

    def test_initialization(self):
        """Provider initializes with config."""
        config = OIDCConfig(
            issuer="https://auth.example.com",
            client_id="my-client",
        )
        provider = OIDCProvider(config)

        assert provider.config == config

    def test_get_authorization_url_requires_discovery(self):
        """Authorization URL generation handles discovery."""
        config = OIDCConfig(
            issuer="https://auth.example.com",
            client_id="my-client",
            redirect_uri="https://app.example.com/callback",
        )
        provider = OIDCProvider(config)

        # Mock discovery
        provider._discovery = MagicMock()
        provider._discovery.authorization_endpoint = "https://auth.example.com/authorize"

        url = provider.get_authorization_url(state="test-state")

        assert "https://auth.example.com/authorize" in url
        assert "client_id=my-client" in url
        assert "state=test-state" in url
        assert "response_type=code" in url


class TestJWTConfig:
    """Tests for JWTConfig."""

    def test_default_values(self):
        """Config has sensible defaults."""
        config = JWTConfig()

        assert config.clock_skew_seconds == 60
        assert config.jwks_cache_ttl_seconds == 3600

    def test_custom_values(self):
        """Custom values are respected."""
        config = JWTConfig(
            issuer="https://auth.example.com",
            audience="my-api",
            jwks_uri="https://auth.example.com/.well-known/jwks.json",
        )

        assert config.issuer == "https://auth.example.com"
        assert config.audience == "my-api"


class TestJWTClaims:
    """Tests for JWTClaims."""

    def test_standard_claims(self):
        """Standard claims are accessible."""
        claims = JWTClaims(
            iss="https://auth.example.com",
            sub="user123",
            aud="my-api",
            exp=int((datetime.now() + timedelta(hours=1)).timestamp()),
            iat=int(datetime.now().timestamp()),
        )

        assert claims.iss == "https://auth.example.com"
        assert claims.sub == "user123"

    def test_expiration_time_property(self):
        """Expiration time property returns datetime."""
        future = datetime.now() + timedelta(hours=1)
        claims = JWTClaims(exp=int(future.timestamp()))

        assert isinstance(claims.expiration_time, datetime)
        assert claims.expiration_time.date() == future.date()

    def test_get_custom_claim(self):
        """Get method returns custom claims."""
        claims = JWTClaims(
            iss="issuer",
            sub="subject",
            custom={"role": "admin", "org_id": "org123"},
        )

        assert claims.get("role") == "admin"
        assert claims.get("org_id") == "org123"
        assert claims.get("missing", "default") == "default"


class TestJWTValidator:
    """Tests for JWTValidator."""

    def test_initialization(self):
        """Validator initializes with config."""
        config = JWTConfig(issuer="https://auth.example.com")
        validator = JWTValidator(config)

        assert validator.config == config

    def test_decode_parts_valid_token(self):
        """Decode parts extracts header and payload."""
        validator = JWTValidator(JWTConfig())

        # Create a simple test token (header.payload.signature)
        import base64
        import json

        header = base64.urlsafe_b64encode(
            json.dumps({"alg": "HS256", "typ": "JWT"}).encode()
        ).decode().rstrip("=")

        payload = base64.urlsafe_b64encode(
            json.dumps({"sub": "user123", "iss": "test"}).encode()
        ).decode().rstrip("=")

        token = f"{header}.{payload}.signature"

        decoded_header, decoded_payload = validator._decode_parts(token)

        assert decoded_header["alg"] == "HS256"
        assert decoded_payload["sub"] == "user123"

    def test_decode_parts_invalid_format(self):
        """Decode parts raises on invalid format."""
        validator = JWTValidator(JWTConfig())

        with pytest.raises(JWTValidationError) as exc_info:
            validator._decode_parts("invalid.token")

        assert "expected 3 parts" in str(exc_info.value)

    def test_validate_claims_expired_token(self):
        """Validate claims rejects expired token."""
        config = JWTConfig()
        validator = JWTValidator(config)

        claims = JWTClaims(
            iss="issuer",
            sub="subject",
            exp=int((datetime.now() - timedelta(hours=1)).timestamp()),
        )

        with pytest.raises(JWTValidationError) as exc_info:
            validator._validate_claims(claims)

        assert exc_info.value.claim == "exp"

    def test_validate_claims_wrong_issuer(self):
        """Validate claims rejects wrong issuer."""
        config = JWTConfig(issuer="https://expected.example.com")
        validator = JWTValidator(config)

        claims = JWTClaims(
            iss="https://wrong.example.com",
            sub="subject",
        )

        with pytest.raises(JWTValidationError) as exc_info:
            validator._validate_claims(claims)

        assert exc_info.value.claim == "iss"


class TestAPIKeyConfig:
    """Tests for APIKeyConfig."""

    def test_default_values(self):
        """Config has sensible defaults."""
        config = APIKeyConfig()

        assert config.header_name == "X-API-Key"
        assert config.query_param == "api_key"
        assert config.key_prefix == "llmt_"
        assert config.key_length == 32


class TestAPIKeyInfo:
    """Tests for APIKeyInfo."""

    def test_create_info(self):
        """Create key info with required fields."""
        info = APIKeyInfo(
            key_id="key123",
            name="My API Key",
            tenant_id="tenant1",
        )

        assert info.key_id == "key123"
        assert info.name == "My API Key"
        assert info.tenant_id == "tenant1"

    def test_is_expired_no_expiry(self):
        """Key without expiry is not expired."""
        info = APIKeyInfo(
            key_id="key123",
            name="My API Key",
            tenant_id="tenant1",
            expires_at=None,
        )

        assert info.is_expired is False

    def test_is_expired_future(self):
        """Key with future expiry is not expired."""
        info = APIKeyInfo(
            key_id="key123",
            name="My API Key",
            tenant_id="tenant1",
            expires_at=datetime.now() + timedelta(days=30),
        )

        assert info.is_expired is False

    def test_is_expired_past(self):
        """Key with past expiry is expired."""
        info = APIKeyInfo(
            key_id="key123",
            name="My API Key",
            tenant_id="tenant1",
            expires_at=datetime.now() - timedelta(days=1),
        )

        assert info.is_expired is True


class TestAPIKeyValidator:
    """Tests for APIKeyValidator."""

    def test_generate_key_has_prefix(self):
        """Generated key has configured prefix."""
        validator = APIKeyValidator()

        key = validator.generate_key()

        assert key.startswith("llmt_")
        assert len(key) > 10

    async def test_create_key(self):
        """Create key returns key and info."""
        validator = APIKeyValidator()

        key, info = await validator.create_key(
            name="Test Key",
            tenant_id="tenant1",
            scopes=["read", "write"],
        )

        assert key.startswith("llmt_")
        assert info.name == "Test Key"
        assert info.tenant_id == "tenant1"
        assert info.scopes == ["read", "write"]

    async def test_validate_valid_key(self):
        """Validate returns info for valid key."""
        validator = APIKeyValidator()

        key, _ = await validator.create_key(
            name="Test Key",
            tenant_id="tenant1",
        )

        info = await validator.validate(key)

        assert info is not None
        assert info.name == "Test Key"

    async def test_validate_invalid_key(self):
        """Validate returns None for invalid key."""
        validator = APIKeyValidator()

        info = await validator.validate("llmt_invalid_key_12345")

        assert info is None

    async def test_validate_wrong_prefix(self):
        """Validate returns None for wrong prefix."""
        validator = APIKeyValidator()

        info = await validator.validate("wrong_prefix_key")

        assert info is None

    async def test_revoke_key(self):
        """Revoke removes key from store."""
        validator = APIKeyValidator()

        key, _ = await validator.create_key(
            name="Test Key",
            tenant_id="tenant1",
        )

        result = await validator.revoke(key)

        assert result is True
        assert await validator.validate(key) is None

    def test_extract_key_from_header(self):
        """Extract key from request header."""
        validator = APIKeyValidator()

        key = validator.extract_key_from_request(
            headers={"X-API-Key": "llmt_mykey123"},
        )

        assert key == "llmt_mykey123"

    def test_extract_key_from_query(self):
        """Extract key from query parameter."""
        validator = APIKeyValidator()

        key = validator.extract_key_from_request(
            headers={},
            query_params={"api_key": "llmt_mykey123"},
        )

        assert key == "llmt_mykey123"


class TestPermission:
    """Tests for Permission enum."""

    def test_standard_permissions(self):
        """Standard permissions exist."""
        assert Permission.READ.value == "read"
        assert Permission.WRITE.value == "write"
        assert Permission.EXECUTE.value == "execute"
        assert Permission.ADMIN.value == "admin"


class TestRole:
    """Tests for Role."""

    def test_create_role(self):
        """Create role with permissions."""
        role = Role(
            name="developer",
            permissions=["read", "write", "execute"],
            description="Developer role",
        )

        assert role.name == "developer"
        assert "read" in role.permissions

    def test_has_permission_direct(self):
        """Has permission for direct permission."""
        role = Role(name="dev", permissions=["read", "write"])
        all_roles = {"dev": role}

        assert role.has_permission("read", all_roles) is True
        assert role.has_permission("admin", all_roles) is False

    def test_has_permission_inherited(self):
        """Has permission through inheritance."""
        viewer = Role(name="viewer", permissions=["read"])
        editor = Role(name="editor", permissions=["write"], inherits=["viewer"])
        all_roles = {"viewer": viewer, "editor": editor}

        assert editor.has_permission("write", all_roles) is True
        assert editor.has_permission("read", all_roles) is True  # Inherited


class TestRBACConfig:
    """Tests for RBACConfig."""

    def test_default_roles(self):
        """Default roles are created."""
        config = RBACConfig()

        assert "viewer" in config.roles
        assert "operator" in config.roles
        assert "developer" in config.roles
        assert "admin" in config.roles

    def test_default_step_permissions(self):
        """Default step permissions are set."""
        config = RBACConfig()

        assert "llm_agent" in config.step_permissions
        assert "http_action" in config.step_permissions
        assert "transform" in config.step_permissions
