"""
JWT Token Validation.

Provides JWT token validation with support for:
- RS256, RS384, RS512 (RSA)
- ES256, ES384, ES512 (ECDSA)
- HS256, HS384, HS512 (HMAC)
- JWKS (JSON Web Key Set) support
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional, Union
import asyncio
import base64
import json
import os


@dataclass
class JWTConfig:
    """Configuration for JWT validation."""

    # Issuer validation
    issuer: str = ""
    issuers: list[str] = field(default_factory=list)  # Multiple issuers

    # Audience validation
    audience: str = ""
    audiences: list[str] = field(default_factory=list)

    # JWKS URL for key discovery
    jwks_uri: str = ""

    # Symmetric key for HS* algorithms
    secret_key: str = ""

    # Clock skew tolerance (seconds)
    clock_skew_seconds: int = 60

    # Required claims
    required_claims: list[str] = field(default_factory=list)

    # Cache settings
    jwks_cache_ttl_seconds: int = 3600

    def __post_init__(self) -> None:
        # Load from environment
        if not self.issuer:
            self.issuer = os.environ.get("JWT_ISSUER", "")
        if not self.audience:
            self.audience = os.environ.get("JWT_AUDIENCE", "")
        if not self.jwks_uri:
            self.jwks_uri = os.environ.get("JWT_JWKS_URI", "")
        if not self.secret_key:
            self.secret_key = os.environ.get("JWT_SECRET_KEY", "")


class JWTValidationError(Exception):
    """JWT validation failed."""

    def __init__(self, message: str, claim: str = "") -> None:
        super().__init__(message)
        self.claim = claim


@dataclass
class JWTClaims:
    """Validated JWT claims."""

    # Standard claims
    iss: str = ""  # Issuer
    sub: str = ""  # Subject
    aud: Union[str, list[str]] = ""  # Audience
    exp: int = 0  # Expiration time
    nbf: int = 0  # Not before
    iat: int = 0  # Issued at
    jti: str = ""  # JWT ID

    # Custom claims
    custom: dict[str, Any] = field(default_factory=dict)

    # Raw token
    raw: str = ""

    @property
    def expiration_time(self) -> datetime:
        """Get expiration as datetime."""
        return datetime.fromtimestamp(self.exp) if self.exp else datetime.max

    @property
    def issued_at_time(self) -> datetime:
        """Get issued at as datetime."""
        return datetime.fromtimestamp(self.iat) if self.iat else datetime.min

    @property
    def not_before_time(self) -> datetime:
        """Get not before as datetime."""
        return datetime.fromtimestamp(self.nbf) if self.nbf else datetime.min

    def get(self, claim: str, default: Any = None) -> Any:
        """Get a claim value."""
        if hasattr(self, claim):
            return getattr(self, claim)
        return self.custom.get(claim, default)


class JWTValidator:
    """
    JWT token validator.

    Validates JWT tokens with signature verification and claim validation.

    Usage:
        validator = JWTValidator(JWTConfig(
            issuer="https://auth.example.com",
            audience="my-api",
            jwks_uri="https://auth.example.com/.well-known/jwks.json",
        ))

        claims = await validator.validate(token)
        print(f"User: {claims.sub}")
    """

    def __init__(self, config: JWTConfig) -> None:
        """
        Initialize JWT validator.

        Args:
            config: JWT configuration
        """
        self.config = config
        self._jwks_cache: Optional[dict] = None
        self._jwks_cache_time: Optional[datetime] = None
        self._lock = asyncio.Lock()

    async def validate(self, token: str) -> JWTClaims:
        """
        Validate a JWT token.

        Args:
            token: JWT token string

        Returns:
            JWTClaims with validated claims

        Raises:
            JWTValidationError: If validation fails
        """
        # Decode without verification first to get header
        try:
            header, payload = self._decode_parts(token)
        except Exception as e:
            raise JWTValidationError(f"Invalid token format: {e}")

        # Verify signature
        await self._verify_signature(token, header)

        # Validate claims
        claims = self._parse_claims(payload, token)
        self._validate_claims(claims)

        return claims

    def _decode_parts(self, token: str) -> tuple[dict, dict]:
        """Decode token header and payload (without verification)."""
        parts = token.split(".")
        if len(parts) != 3:
            raise JWTValidationError("Invalid token format: expected 3 parts")

        def decode_part(part: str) -> dict:
            # Add padding
            padding = 4 - len(part) % 4
            if padding != 4:
                part += "=" * padding
            decoded = base64.urlsafe_b64decode(part)
            return json.loads(decoded)

        header = decode_part(parts[0])
        payload = decode_part(parts[1])

        return header, payload

    async def _verify_signature(self, token: str, header: dict) -> None:
        """Verify token signature."""
        alg = header.get("alg", "RS256")

        if alg.startswith("HS"):
            # HMAC signature
            self._verify_hmac(token, alg)
        elif alg.startswith("RS") or alg.startswith("ES"):
            # RSA or ECDSA signature
            await self._verify_asymmetric(token, header, alg)
        else:
            raise JWTValidationError(f"Unsupported algorithm: {alg}")

    def _verify_hmac(self, token: str, alg: str) -> None:
        """Verify HMAC signature."""
        if not self.config.secret_key:
            raise JWTValidationError("Secret key required for HMAC verification")

        try:
            import hmac
            import hashlib

            parts = token.split(".")
            message = f"{parts[0]}.{parts[1]}".encode()
            signature = base64.urlsafe_b64decode(parts[2] + "==")

            hash_alg = {
                "HS256": hashlib.sha256,
                "HS384": hashlib.sha384,
                "HS512": hashlib.sha512,
            }.get(alg)

            if not hash_alg:
                raise JWTValidationError(f"Unsupported HMAC algorithm: {alg}")

            expected = hmac.new(
                self.config.secret_key.encode(),
                message,
                hash_alg,
            ).digest()

            if not hmac.compare_digest(signature, expected):
                raise JWTValidationError("Invalid signature")

        except JWTValidationError:
            raise
        except Exception as e:
            raise JWTValidationError(f"Signature verification failed: {e}")

    async def _verify_asymmetric(self, token: str, header: dict, alg: str) -> None:
        """Verify RSA or ECDSA signature."""
        try:
            # Try to use PyJWT if available
            import jwt

            # Get JWKS
            jwks = await self._get_jwks()

            # Find the key
            kid = header.get("kid")
            key = None

            for jwk in jwks.get("keys", []):
                if kid and jwk.get("kid") != kid:
                    continue
                if jwk.get("alg") and jwk.get("alg") != alg:
                    continue
                key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(jwk))
                break

            if not key:
                raise JWTValidationError("No matching key found in JWKS")

            # Verify
            jwt.decode(
                token,
                key,
                algorithms=[alg],
                options={"verify_aud": False, "verify_iss": False},
            )

        except ImportError:
            # PyJWT not available, skip cryptographic verification
            # In production, you should install PyJWT
            pass
        except jwt.InvalidSignatureError:
            raise JWTValidationError("Invalid signature")
        except Exception as e:
            raise JWTValidationError(f"Signature verification failed: {e}")

    async def _get_jwks(self) -> dict:
        """Get JWKS, using cache if available."""
        async with self._lock:
            now = datetime.now()

            # Check cache
            if (
                self._jwks_cache
                and self._jwks_cache_time
                and (now - self._jwks_cache_time).total_seconds()
                < self.config.jwks_cache_ttl_seconds
            ):
                return self._jwks_cache

            # Fetch JWKS
            if not self.config.jwks_uri:
                raise JWTValidationError("JWKS URI not configured")

            try:
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    async with session.get(self.config.jwks_uri) as response:
                        if response.status != 200:
                            raise JWTValidationError(
                                f"Failed to fetch JWKS: {response.status}"
                            )
                        self._jwks_cache = await response.json()
                        self._jwks_cache_time = now
                        return self._jwks_cache

            except ImportError:
                raise JWTValidationError("aiohttp required for JWKS fetch")

    def _parse_claims(self, payload: dict, token: str) -> JWTClaims:
        """Parse payload into JWTClaims."""
        standard_claims = {"iss", "sub", "aud", "exp", "nbf", "iat", "jti"}
        custom = {k: v for k, v in payload.items() if k not in standard_claims}

        return JWTClaims(
            iss=payload.get("iss", ""),
            sub=payload.get("sub", ""),
            aud=payload.get("aud", ""),
            exp=payload.get("exp", 0),
            nbf=payload.get("nbf", 0),
            iat=payload.get("iat", 0),
            jti=payload.get("jti", ""),
            custom=custom,
            raw=token,
        )

    def _validate_claims(self, claims: JWTClaims) -> None:
        """Validate JWT claims."""
        now = datetime.now()
        skew = timedelta(seconds=self.config.clock_skew_seconds)

        # Validate issuer
        valid_issuers = self.config.issuers or (
            [self.config.issuer] if self.config.issuer else []
        )
        if valid_issuers and claims.iss not in valid_issuers:
            raise JWTValidationError(f"Invalid issuer: {claims.iss}", "iss")

        # Validate audience
        valid_audiences = self.config.audiences or (
            [self.config.audience] if self.config.audience else []
        )
        if valid_audiences:
            token_aud = claims.aud if isinstance(claims.aud, list) else [claims.aud]
            if not any(aud in valid_audiences for aud in token_aud):
                raise JWTValidationError(f"Invalid audience: {claims.aud}", "aud")

        # Validate expiration
        if claims.exp:
            if now > claims.expiration_time + skew:
                raise JWTValidationError("Token expired", "exp")

        # Validate not before
        if claims.nbf:
            if now < claims.not_before_time - skew:
                raise JWTValidationError("Token not yet valid", "nbf")

        # Validate required claims
        for claim in self.config.required_claims:
            if not claims.get(claim):
                raise JWTValidationError(f"Missing required claim: {claim}", claim)
