"""
Authentication Module for LLMTeam.

Provides authentication mechanisms including:
- OIDC (OpenID Connect)
- API Key authentication
- JWT token validation

Usage:
    from llmteam.auth import OIDCProvider, JWTValidator, APIKeyValidator

    # OIDC authentication
    oidc = OIDCProvider(
        issuer="https://auth.example.com",
        client_id="my-client",
        client_secret="secret",
    )
    token = await oidc.authenticate()

    # JWT validation
    validator = JWTValidator(issuer="https://auth.example.com")
    claims = await validator.validate(token)
"""

from llmteam.auth.oidc import (
    OIDCConfig,
    OIDCProvider,
    OIDCError,
    OIDCAuthenticationError,
    OIDCTokenError,
)

from llmteam.auth.jwt import (
    JWTConfig,
    JWTValidator,
    JWTClaims,
    JWTValidationError,
)

from llmteam.auth.apikey import (
    APIKeyConfig,
    APIKeyValidator,
    APIKeyInfo,
)

from llmteam.auth.middleware import (
    AuthenticationMiddleware,
    AuthorizationMiddleware,
    Permission,
    Role,
    RBACConfig,
)

__all__ = [
    # OIDC
    "OIDCConfig",
    "OIDCProvider",
    "OIDCError",
    "OIDCAuthenticationError",
    "OIDCTokenError",
    # JWT
    "JWTConfig",
    "JWTValidator",
    "JWTClaims",
    "JWTValidationError",
    # API Key
    "APIKeyConfig",
    "APIKeyValidator",
    "APIKeyInfo",
    # Middleware
    "AuthenticationMiddleware",
    "AuthorizationMiddleware",
    "Permission",
    "Role",
    "RBACConfig",
]
