"""
OIDC (OpenID Connect) Authentication Provider.

Provides OIDC authentication for machine-to-machine and user authentication.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional
import asyncio
import base64
import hashlib
import json
import os
import secrets


@dataclass
class OIDCConfig:
    """Configuration for OIDC provider."""

    # Required
    issuer: str = ""
    client_id: str = ""
    client_secret: str = ""

    # Optional
    redirect_uri: str = ""
    scopes: list[str] = field(default_factory=lambda: ["openid", "profile", "email"])

    # Discovery
    discovery_url: str = ""  # Default: {issuer}/.well-known/openid-configuration

    # Token settings
    token_refresh_buffer_seconds: int = 60  # Refresh token this many seconds before expiry

    # PKCE (Proof Key for Code Exchange)
    use_pkce: bool = True

    def __post_init__(self) -> None:
        # Load from environment if not set
        if not self.issuer:
            self.issuer = os.environ.get("OIDC_ISSUER", "")
        if not self.client_id:
            self.client_id = os.environ.get("OIDC_CLIENT_ID", "")
        if not self.client_secret:
            self.client_secret = os.environ.get("OIDC_CLIENT_SECRET", "")
        if not self.redirect_uri:
            self.redirect_uri = os.environ.get("OIDC_REDIRECT_URI", "")

        # Set default discovery URL
        if not self.discovery_url and self.issuer:
            self.discovery_url = f"{self.issuer.rstrip('/')}/.well-known/openid-configuration"


class OIDCError(Exception):
    """Base exception for OIDC errors."""
    pass


class OIDCAuthenticationError(OIDCError):
    """Authentication failed."""
    pass


class OIDCTokenError(OIDCError):
    """Token operation failed."""
    pass


@dataclass
class OIDCToken:
    """OIDC token response."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 3600
    refresh_token: Optional[str] = None
    id_token: Optional[str] = None
    scope: str = ""

    # Computed
    expires_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if self.expires_at == datetime.now():
            self.expires_at = datetime.now() + timedelta(seconds=self.expires_in)

    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        return datetime.now() >= self.expires_at

    def needs_refresh(self, buffer_seconds: int = 60) -> bool:
        """Check if token needs refresh."""
        return datetime.now() >= (self.expires_at - timedelta(seconds=buffer_seconds))


@dataclass
class OIDCDiscovery:
    """OIDC discovery document."""

    issuer: str = ""
    authorization_endpoint: str = ""
    token_endpoint: str = ""
    userinfo_endpoint: str = ""
    jwks_uri: str = ""
    end_session_endpoint: str = ""
    revocation_endpoint: str = ""

    # Supported features
    response_types_supported: list[str] = field(default_factory=list)
    grant_types_supported: list[str] = field(default_factory=list)
    scopes_supported: list[str] = field(default_factory=list)
    token_endpoint_auth_methods_supported: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "OIDCDiscovery":
        """Create from discovery response."""
        return cls(
            issuer=data.get("issuer", ""),
            authorization_endpoint=data.get("authorization_endpoint", ""),
            token_endpoint=data.get("token_endpoint", ""),
            userinfo_endpoint=data.get("userinfo_endpoint", ""),
            jwks_uri=data.get("jwks_uri", ""),
            end_session_endpoint=data.get("end_session_endpoint", ""),
            revocation_endpoint=data.get("revocation_endpoint", ""),
            response_types_supported=data.get("response_types_supported", []),
            grant_types_supported=data.get("grant_types_supported", []),
            scopes_supported=data.get("scopes_supported", []),
            token_endpoint_auth_methods_supported=data.get(
                "token_endpoint_auth_methods_supported", []
            ),
        )


class OIDCProvider:
    """
    OIDC authentication provider.

    Supports:
    - Client credentials flow (machine-to-machine)
    - Authorization code flow (user authentication)
    - Token refresh
    - Token revocation

    Usage:
        provider = OIDCProvider(OIDCConfig(
            issuer="https://auth.example.com",
            client_id="my-client",
            client_secret="secret",
        ))

        # Client credentials (M2M)
        token = await provider.authenticate_client_credentials()

        # Authorization code flow
        auth_url = provider.get_authorization_url(state="my-state")
        # ... user authenticates ...
        token = await provider.exchange_code(code="auth-code")
    """

    def __init__(self, config: OIDCConfig) -> None:
        """
        Initialize OIDC provider.

        Args:
            config: OIDC configuration
        """
        self.config = config
        self._discovery: Optional[OIDCDiscovery] = None
        self._current_token: Optional[OIDCToken] = None
        self._lock = asyncio.Lock()

        # PKCE state
        self._code_verifier: Optional[str] = None

    async def discover(self) -> OIDCDiscovery:
        """
        Fetch OIDC discovery document.

        Returns:
            OIDCDiscovery with endpoint URLs
        """
        if self._discovery:
            return self._discovery

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(self.config.discovery_url) as response:
                    if response.status != 200:
                        raise OIDCError(
                            f"Discovery failed: {response.status} {await response.text()}"
                        )
                    data = await response.json()
                    self._discovery = OIDCDiscovery.from_dict(data)
                    return self._discovery

        except ImportError:
            raise OIDCError("aiohttp required for OIDC. Install with: pip install aiohttp")

    async def authenticate_client_credentials(
        self,
        scopes: Optional[list[str]] = None,
    ) -> OIDCToken:
        """
        Authenticate using client credentials flow.

        This is for machine-to-machine authentication.

        Args:
            scopes: Optional override for scopes

        Returns:
            OIDCToken with access token
        """
        discovery = await self.discover()

        token_data = {
            "grant_type": "client_credentials",
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "scope": " ".join(scopes or self.config.scopes),
        }

        return await self._request_token(discovery.token_endpoint, token_data)

    def get_authorization_url(
        self,
        state: str,
        scopes: Optional[list[str]] = None,
        nonce: Optional[str] = None,
    ) -> str:
        """
        Generate authorization URL for user authentication.

        Args:
            state: State parameter for CSRF protection
            scopes: Optional override for scopes
            nonce: Optional nonce for ID token validation

        Returns:
            Authorization URL to redirect user to
        """
        if not self._discovery:
            # Synchronous discovery fallback
            import urllib.request
            import json as json_module

            with urllib.request.urlopen(self.config.discovery_url) as response:
                data = json_module.loads(response.read().decode())
                self._discovery = OIDCDiscovery.from_dict(data)

        params = {
            "response_type": "code",
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "scope": " ".join(scopes or self.config.scopes),
            "state": state,
        }

        if nonce:
            params["nonce"] = nonce

        # PKCE
        if self.config.use_pkce:
            self._code_verifier = secrets.token_urlsafe(32)
            code_challenge = base64.urlsafe_b64encode(
                hashlib.sha256(self._code_verifier.encode()).digest()
            ).decode().rstrip("=")
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"

        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self._discovery.authorization_endpoint}?{query}"

    async def exchange_code(
        self,
        code: str,
        state: Optional[str] = None,
    ) -> OIDCToken:
        """
        Exchange authorization code for tokens.

        Args:
            code: Authorization code from callback
            state: State parameter for validation

        Returns:
            OIDCToken with access and ID tokens
        """
        discovery = await self.discover()

        token_data = {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "redirect_uri": self.config.redirect_uri,
        }

        # PKCE
        if self.config.use_pkce and self._code_verifier:
            token_data["code_verifier"] = self._code_verifier
            self._code_verifier = None

        return await self._request_token(discovery.token_endpoint, token_data)

    async def refresh_token(self, refresh_token: str) -> OIDCToken:
        """
        Refresh an access token.

        Args:
            refresh_token: Refresh token

        Returns:
            New OIDCToken
        """
        discovery = await self.discover()

        token_data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
        }

        return await self._request_token(discovery.token_endpoint, token_data)

    async def revoke_token(self, token: str, token_type: str = "access_token") -> bool:
        """
        Revoke a token.

        Args:
            token: Token to revoke
            token_type: Type of token ("access_token" or "refresh_token")

        Returns:
            True if revoked successfully
        """
        discovery = await self.discover()

        if not discovery.revocation_endpoint:
            raise OIDCError("Revocation endpoint not available")

        try:
            import aiohttp

            data = {
                "token": token,
                "token_type_hint": token_type,
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    discovery.revocation_endpoint,
                    data=data,
                ) as response:
                    return response.status == 200

        except ImportError:
            raise OIDCError("aiohttp required for OIDC")

    async def get_token(self) -> OIDCToken:
        """
        Get a valid token, refreshing if necessary.

        Uses client credentials flow.

        Returns:
            Valid OIDCToken
        """
        async with self._lock:
            if self._current_token is None or self._current_token.is_expired:
                self._current_token = await self.authenticate_client_credentials()
            elif self._current_token.needs_refresh(
                self.config.token_refresh_buffer_seconds
            ):
                if self._current_token.refresh_token:
                    try:
                        self._current_token = await self.refresh_token(
                            self._current_token.refresh_token
                        )
                    except OIDCTokenError:
                        # Refresh failed, get new token
                        self._current_token = await self.authenticate_client_credentials()
                else:
                    self._current_token = await self.authenticate_client_credentials()

            return self._current_token

    async def _request_token(self, endpoint: str, data: dict) -> OIDCToken:
        """Make token request."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                ) as response:
                    response_data = await response.json()

                    if response.status != 200:
                        error = response_data.get("error", "unknown_error")
                        description = response_data.get("error_description", "")
                        raise OIDCTokenError(f"{error}: {description}")

                    return OIDCToken(
                        access_token=response_data["access_token"],
                        token_type=response_data.get("token_type", "Bearer"),
                        expires_in=response_data.get("expires_in", 3600),
                        refresh_token=response_data.get("refresh_token"),
                        id_token=response_data.get("id_token"),
                        scope=response_data.get("scope", ""),
                    )

        except ImportError:
            raise OIDCError("aiohttp required for OIDC")
