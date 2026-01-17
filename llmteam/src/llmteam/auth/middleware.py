"""
Authentication and Authorization Middleware.

Provides middleware for authenticating and authorizing step execution.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum

from llmteam.runtime import StepContext


class Permission(Enum):
    """Standard permissions for steps."""

    # Read permissions
    READ = "read"
    READ_SECRETS = "read:secrets"
    READ_STORE = "read:store"

    # Write permissions
    WRITE = "write"
    WRITE_SECRETS = "write:secrets"
    WRITE_STORE = "write:store"

    # Execute permissions
    EXECUTE = "execute"
    EXECUTE_LLM = "execute:llm"
    EXECUTE_HTTP = "execute:http"
    EXECUTE_HUMAN = "execute:human"

    # Admin permissions
    ADMIN = "admin"
    ADMIN_TENANT = "admin:tenant"
    ADMIN_AUDIT = "admin:audit"


@dataclass
class Role:
    """Role definition for RBAC."""

    name: str
    permissions: list[str] = field(default_factory=list)
    description: str = ""

    # Inheritance
    inherits: list[str] = field(default_factory=list)

    def has_permission(self, permission: str, all_roles: dict[str, "Role"]) -> bool:
        """Check if role has permission (including inherited)."""
        if permission in self.permissions:
            return True

        # Check inherited roles
        for inherited_name in self.inherits:
            inherited_role = all_roles.get(inherited_name)
            if inherited_role and inherited_role.has_permission(permission, all_roles):
                return True

        return False


@dataclass
class RBACConfig:
    """Configuration for role-based access control."""

    # Built-in roles
    roles: dict[str, Role] = field(default_factory=dict)

    # Step type to required permissions mapping
    step_permissions: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Set up default roles if not provided
        if not self.roles:
            self.roles = {
                "viewer": Role(
                    name="viewer",
                    permissions=[
                        Permission.READ.value,
                        Permission.READ_STORE.value,
                    ],
                    description="Read-only access",
                ),
                "operator": Role(
                    name="operator",
                    permissions=[
                        Permission.EXECUTE.value,
                        Permission.EXECUTE_LLM.value,
                        Permission.EXECUTE_HTTP.value,
                    ],
                    inherits=["viewer"],
                    description="Can execute workflows",
                ),
                "developer": Role(
                    name="developer",
                    permissions=[
                        Permission.WRITE.value,
                        Permission.WRITE_STORE.value,
                        Permission.EXECUTE_HUMAN.value,
                    ],
                    inherits=["operator"],
                    description="Full workflow access",
                ),
                "admin": Role(
                    name="admin",
                    permissions=[
                        Permission.ADMIN.value,
                        Permission.ADMIN_TENANT.value,
                        Permission.ADMIN_AUDIT.value,
                        Permission.READ_SECRETS.value,
                        Permission.WRITE_SECRETS.value,
                    ],
                    inherits=["developer"],
                    description="Full administrative access",
                ),
            }

        # Default step permissions
        if not self.step_permissions:
            self.step_permissions = {
                "llm_agent": [Permission.EXECUTE_LLM.value],
                "http_action": [Permission.EXECUTE_HTTP.value],
                "human_task": [Permission.EXECUTE_HUMAN.value],
                "transform": [Permission.EXECUTE.value],
                "condition": [Permission.EXECUTE.value],
                "parallel_split": [Permission.EXECUTE.value],
                "parallel_join": [Permission.EXECUTE.value],
            }


class AuthenticationMiddleware:
    """
    Middleware for authenticating requests.

    Validates tokens/API keys and attaches user info to context.
    """

    name = "authentication"
    priority = 0  # Run first
    enabled = True

    def __init__(
        self,
        jwt_validator: Optional[Any] = None,
        api_key_validator: Optional[Any] = None,
        on_auth_failure: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Initialize authentication middleware.

        Args:
            jwt_validator: JWTValidator instance
            api_key_validator: APIKeyValidator instance
            on_auth_failure: Callback for authentication failures
        """
        self.jwt_validator = jwt_validator
        self.api_key_validator = api_key_validator
        self.on_auth_failure = on_auth_failure

    def should_run(
        self,
        step_type: str,
        step_id: str,
        middleware_ctx: Any,
    ) -> bool:
        """Always run authentication."""
        return self.enabled

    async def __call__(
        self,
        ctx: StepContext,
        config: dict,
        input_data: dict,
        next_handler: Callable,
        middleware_ctx: Any,
    ) -> Any:
        """Authenticate the request."""
        # Check for existing authentication
        if ctx.metadata.get("authenticated"):
            return await next_handler(ctx, config, input_data)

        # Try JWT
        token = ctx.metadata.get("authorization_token")
        if token and self.jwt_validator:
            try:
                claims = await self.jwt_validator.validate(token)
                ctx.metadata["authenticated"] = True
                ctx.metadata["auth_method"] = "jwt"
                ctx.metadata["user_id"] = claims.sub
                ctx.metadata["jwt_claims"] = claims
                middleware_ctx.middleware_data["auth"] = {
                    "method": "jwt",
                    "user_id": claims.sub,
                }
                return await next_handler(ctx, config, input_data)
            except Exception as e:
                if self.on_auth_failure:
                    self.on_auth_failure(str(e))
                raise PermissionError(f"JWT authentication failed: {e}")

        # Try API key
        api_key = ctx.metadata.get("api_key")
        if api_key and self.api_key_validator:
            key_info = await self.api_key_validator.validate(api_key)
            if key_info:
                ctx.metadata["authenticated"] = True
                ctx.metadata["auth_method"] = "api_key"
                ctx.metadata["api_key_info"] = key_info
                ctx.metadata["permissions"] = key_info.permissions
                middleware_ctx.middleware_data["auth"] = {
                    "method": "api_key",
                    "key_id": key_info.key_id,
                }
                return await next_handler(ctx, config, input_data)
            else:
                if self.on_auth_failure:
                    self.on_auth_failure("Invalid API key")
                raise PermissionError("Invalid API key")

        # No authentication provided - allow if no validators configured
        if not self.jwt_validator and not self.api_key_validator:
            ctx.metadata["authenticated"] = False
            ctx.metadata["auth_method"] = "none"
            return await next_handler(ctx, config, input_data)

        raise PermissionError("Authentication required")


class AuthorizationMiddleware:
    """
    Middleware for authorizing step execution.

    Checks permissions based on step type and user roles.
    """

    name = "authorization"
    priority = 1  # Run after authentication
    enabled = True

    def __init__(
        self,
        config: Optional[RBACConfig] = None,
        permission_checker: Optional[Callable[[StepContext, str, list[str]], bool]] = None,
    ) -> None:
        """
        Initialize authorization middleware.

        Args:
            config: RBAC configuration
            permission_checker: Custom permission checker
        """
        self.config = config or RBACConfig()
        self.permission_checker = permission_checker

    def should_run(
        self,
        step_type: str,
        step_id: str,
        middleware_ctx: Any,
    ) -> bool:
        """Always run authorization."""
        return self.enabled

    async def __call__(
        self,
        ctx: StepContext,
        config: dict,
        input_data: dict,
        next_handler: Callable,
        middleware_ctx: Any,
    ) -> Any:
        """Authorize the request."""
        step_type = middleware_ctx.step_type

        # Get required permissions for step type
        required_permissions = self.config.step_permissions.get(step_type, [])

        if not required_permissions:
            # No permissions required
            return await next_handler(ctx, config, input_data)

        # Custom permission checker
        if self.permission_checker:
            if not self.permission_checker(ctx, step_type, required_permissions):
                raise PermissionError(
                    f"Access denied for step type '{step_type}'. "
                    f"Required permissions: {required_permissions}"
                )
            return await next_handler(ctx, config, input_data)

        # Check user permissions
        user_permissions = set(ctx.metadata.get("permissions", []))
        user_roles = ctx.metadata.get("roles", [])

        # Expand roles to permissions
        for role_name in user_roles:
            role = self.config.roles.get(role_name)
            if role:
                for perm in self._get_role_permissions(role):
                    user_permissions.add(perm)

        # Check if user has required permissions
        missing = set(required_permissions) - user_permissions

        # Check for admin override
        if Permission.ADMIN.value in user_permissions:
            missing = set()

        if missing:
            raise PermissionError(
                f"Access denied for step type '{step_type}'. "
                f"Missing permissions: {missing}"
            )

        middleware_ctx.middleware_data["authorization"] = {
            "permissions_checked": required_permissions,
            "user_permissions": list(user_permissions),
        }

        return await next_handler(ctx, config, input_data)

    def _get_role_permissions(self, role: Role) -> set[str]:
        """Get all permissions for a role, including inherited."""
        permissions = set(role.permissions)

        for inherited_name in role.inherits:
            inherited_role = self.config.roles.get(inherited_name)
            if inherited_role:
                permissions.update(self._get_role_permissions(inherited_role))

        return permissions
