"""
Runtime Context - unified access point for enterprise resources.

RuntimeContext is passed to each step through injection.
Contains all dependencies resolved by ID/ref.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from llmteam.observability import get_logger
from llmteam.runtime.protocols import Store, Client, LLMProvider, SecretsProvider
from llmteam.runtime.registries import StoreRegistry, ClientRegistry, LLMRegistry
from llmteam.runtime.exceptions import ResourceNotFoundError, RuntimeContextError

if TYPE_CHECKING:
    from llmteam.ratelimit import RateLimitedExecutor
    from llmteam.audit import AuditTrail
    from llmteam.transport import SecureBus
    from llmteam.team import LLMTeam
    from llmteam.registry import TeamRegistry


logger = get_logger(__name__)


# Context variable for current runtime
current_runtime: ContextVar[Optional["RuntimeContext"]] = ContextVar(
    "current_runtime",
    default=None,
)


def get_current_runtime() -> "RuntimeContext":
    """Get current RuntimeContext or raise error."""
    ctx = current_runtime.get()
    if ctx is None:
        raise RuntimeContextError("No RuntimeContext active. Use RuntimeContextManager.")
    return ctx


@dataclass
class RuntimeContext:
    """
    Unified access point for enterprise resources.

    Passed to each step through injection.
    Contains all dependencies resolved by ID/ref.
    """

    # === Identity ===
    tenant_id: str
    instance_id: str  # Unique workflow instance ID
    run_id: str  # Current run ID
    segment_id: str  # Segment (pipeline) ID

    # === Resource Registries ===
    stores: StoreRegistry = field(default_factory=StoreRegistry)
    clients: ClientRegistry = field(default_factory=ClientRegistry)
    llms: LLMRegistry = field(default_factory=LLMRegistry)
    secrets: Optional[SecretsProvider] = None

    # === Policies (from v1.7.0-v1.9.0) ===
    rate_limiter: Optional["RateLimitedExecutor"] = None
    audit_trail: Optional["AuditTrail"] = None

    # === Transport (v2.3.0) ===
    bus: Optional["SecureBus"] = None

    # === Teams (v3.0.0) ===
    teams: Optional["TeamRegistry"] = None

    # === Event Hooks ===
    on_step_start: Optional[Callable[[Any], None]] = None
    on_step_complete: Optional[Callable[[Any], None]] = None
    on_step_error: Optional[Callable[[Any], None]] = None
    on_event: Optional[Callable[[Any], None]] = None

    # === Timestamps ===
    created_at: datetime = field(default_factory=datetime.now)

    # === Custom data ===
    data: Dict[str, Any] = field(default_factory=dict)

    # === Helpers ===

    def resolve_store(self, store_ref: str) -> Store:
        """Resolve store by reference."""
        try:
            return self.stores.get(store_ref)
        except ResourceNotFoundError:
            logger.error(
                f"Store resolution failed: '{store_ref}' not found. "
                f"Context: tenant={self.tenant_id}, run_id={self.run_id}"
            )
            raise

    def resolve_client(self, client_ref: str) -> Client:
        """Resolve client by reference."""
        try:
            return self.clients.get(client_ref)
        except ResourceNotFoundError:
            logger.error(
                f"Client resolution failed: '{client_ref}' not found. "
                f"Context: tenant={self.tenant_id}, run_id={self.run_id}"
            )
            raise

    def resolve_llm(self, llm_ref: str) -> LLMProvider:
        """Resolve LLM provider by reference."""
        try:
            return self.llms.get(llm_ref)
        except ResourceNotFoundError:
            logger.error(
                f"LLM resolution failed: '{llm_ref}' not found. "
                f"Context: tenant={self.tenant_id}, run_id={self.run_id}"
            )
            raise

    async def resolve_secret(self, secret_ref: str) -> str:
        """Resolve secret by reference."""
        if not self.secrets:
            logger.error(
                f"Secret resolution failed: '{secret_ref}'. "
                f"No SecretsProvider configured. "
                f"Context: tenant={self.tenant_id}, run_id={self.run_id}"
            )
            raise ResourceNotFoundError("SecretsProvider not configured")
        
        try:
            return await self.secrets.get_secret(secret_ref)
        except Exception as e:
            logger.error(
                f"Failed to retrieve secret '{secret_ref}': {str(e)}. "
                f"Context: tenant={self.tenant_id}, run_id={self.run_id}"
            )
            raise

    def resolve_team(self, team_ref: str) -> "LLMTeam":
        """Resolve team by reference."""
        if not self.teams:
            logger.error(
                f"Team resolution failed: '{team_ref}'. "
                f"No TeamRegistry configured. "
                f"Context: tenant={self.tenant_id}, run_id={self.run_id}"
            )
            raise ResourceNotFoundError("TeamRegistry not configured")

        team = self.teams.get_optional(team_ref)
        if team is None:
            logger.error(
                f"Team resolution failed: '{team_ref}' not found. "
                f"Context: tenant={self.tenant_id}, run_id={self.run_id}"
            )
            raise ResourceNotFoundError(f"Team '{team_ref}' not found")

        return team

    def get_team(self, team_ref: str) -> Optional["LLMTeam"]:
        """Get team by reference (returns None if not found)."""
        if not self.teams:
            return None
        return self.teams.get_optional(team_ref)

    def register_team(self, team: "LLMTeam") -> None:
        """Register a team with the runtime."""
        if self.teams is None:
            from llmteam.registry import TeamRegistry
            self.teams = TeamRegistry()
        self.teams.register_team(team)
        team.runtime = self

    def child_context(self, step_id: str) -> "StepContext":
        """Create child context for a step."""
        return StepContext(
            runtime=self,
            step_id=step_id,
        )

    def copy(self, **overrides: Any) -> "RuntimeContext":
        """
        Create a copy with optional overrides.
        
        Using copy() is preferred over manual instantiation for branching contexts,
        as it preserves existing services and hooks unless explicitly overridden.
        """
        return RuntimeContext(
            tenant_id=overrides.get("tenant_id", self.tenant_id),
            instance_id=overrides.get("instance_id", self.instance_id),
            run_id=overrides.get("run_id", self.run_id),
            segment_id=overrides.get("segment_id", self.segment_id),
            stores=overrides.get("stores", self.stores),
            clients=overrides.get("clients", self.clients),
            llms=overrides.get("llms", self.llms),
            secrets=overrides.get("secrets", self.secrets),
            rate_limiter=overrides.get("rate_limiter", self.rate_limiter),
            audit_trail=overrides.get("audit_trail", self.audit_trail),
            bus=overrides.get("bus", self.bus),
            teams=overrides.get("teams", self.teams),
            on_step_start=overrides.get("on_step_start", self.on_step_start),
            on_step_complete=overrides.get("on_step_complete", self.on_step_complete),
            on_step_error=overrides.get("on_step_error", self.on_step_error),
            on_event=overrides.get("on_event", self.on_event),
            data=overrides.get("data", dict(self.data)),
        )


@dataclass
class StepContext:
    """Context for a specific step."""

    runtime: RuntimeContext
    step_id: str

    # Step-local state
    _state: Dict[str, Any] = field(default_factory=dict)

    @property
    def tenant_id(self) -> str:
        return self.runtime.tenant_id

    @property
    def instance_id(self) -> str:
        return self.runtime.instance_id

    @property
    def run_id(self) -> str:
        return self.runtime.run_id

    @property
    def segment_id(self) -> str:
        return self.runtime.segment_id

    def get_store(self, store_ref: str) -> Store:
        """Get store by reference."""
        return self.runtime.resolve_store(store_ref)

    def get_client(self, client_ref: str) -> Client:
        """Get client by reference."""
        return self.runtime.resolve_client(client_ref)

    def get_llm(self, llm_ref: str) -> LLMProvider:
        """Get LLM provider by reference."""
        return self.runtime.resolve_llm(llm_ref)

    async def get_secret(self, secret_ref: str) -> str:
        """Get secret by reference."""
        return await self.runtime.resolve_secret(secret_ref)

    def get_bus(self) -> Optional["SecureBus"]:
        """Get SecureBus for event publishing."""
        return self.runtime.bus

    def get_team(self, team_ref: str) -> Optional["LLMTeam"]:
        """Get team by reference."""
        return self.runtime.get_team(team_ref)

    # === Step-local state ===

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get step-local state value."""
        return self._state.get(key, default)

    def set_state(self, key: str, value: Any) -> None:
        """Set step-local state value."""
        self._state[key] = value

    def clear_state(self) -> None:
        """Clear step-local state."""
        self._state.clear()


class RuntimeContextManager:
    """Context manager for RuntimeContext."""

    def __init__(self, context: RuntimeContext):
        self.context = context
        self._token: Any = None

    def __enter__(self) -> RuntimeContext:
        self._token = current_runtime.set(self.context)
        return self.context

    def __exit__(self, *args: Any) -> None:
        if self._token is not None:
            current_runtime.reset(self._token)
            self._token = None

    async def __aenter__(self) -> RuntimeContext:
        return self.__enter__()

    async def __aexit__(self, *args: Any) -> None:
        self.__exit__(*args)


class RuntimeContextFactory:
    """
    Factory for creating RuntimeContext instances.

    Maintains shared registries for stores, clients, and LLMs.
    Use create_runtime() to create isolated contexts for workflow runs.

    Example:
        factory = RuntimeContextFactory()
        factory.register_store("redis", redis_store)
        factory.register_llm("gpt4", openai_provider)

        # Create runtime for a specific run
        runtime = factory.create_runtime(
            tenant_id="tenant1",
            instance_id="workflow-123",
        )
    """

    def __init__(
        self,
        stores: Optional[StoreRegistry] = None,
        clients: Optional[ClientRegistry] = None,
        llms: Optional[LLMRegistry] = None,
        secrets: Optional[SecretsProvider] = None,
        bus: Optional["SecureBus"] = None,
        teams: Optional["TeamRegistry"] = None,
    ):
        self.stores = stores or StoreRegistry()
        self.clients = clients or ClientRegistry()
        self.llms = llms or LLMRegistry()
        self.secrets = secrets
        self.bus = bus
        self.teams = teams
        logger.debug("RuntimeContextFactory initialized")

    def set_secrets_provider(self, secrets: SecretsProvider) -> None:
        """Set the secrets provider."""
        self.secrets = secrets
        logger.debug("SecretsProvider configured")

    def set_bus(self, bus: SecureBus) -> None:
        """Set the secure bus."""
        self.bus = bus
        logger.debug("SecureBus configured")

    def register_store(self, name: str, store: Store) -> None:
        """Register a store."""
        self.stores.register(name, store)

    def register_client(self, name: str, client: Client) -> None:
        """Register a client."""
        self.clients.register(name, client)

    def register_llm(self, name: str, llm: LLMProvider) -> None:
        """Register an LLM provider."""
        self.llms.register(name, llm)

    def register_team(self, team: "LLMTeam") -> None:
        """Register a team."""
        if self.teams is None:
            from llmteam.registry import TeamRegistry
            self.teams = TeamRegistry()
        self.teams.register_team(team)

    def create_runtime(
        self,
        tenant_id: str,
        instance_id: str,
        run_id: Optional[str] = None,
        segment_id: str = "",
        **kwargs: Any,
    ) -> RuntimeContext:
        """
        Create a new RuntimeContext.

        Args:
            tenant_id: Tenant identifier
            instance_id: Workflow instance identifier
            run_id: Run identifier (auto-generated if not provided)
            segment_id: Segment identifier
            **kwargs: Additional context arguments

        Returns:
            New RuntimeContext with shared registries
        """
        import uuid
        
        actual_run_id = run_id or str(uuid.uuid4())
        
        logger.info(
            f"Creating runtime context: tenant={tenant_id}, "
            f"run_id={actual_run_id}, segment={segment_id}"
        )

        return RuntimeContext(
            tenant_id=tenant_id,
            instance_id=instance_id,
            run_id=actual_run_id,
            segment_id=segment_id,
            stores=self.stores,
            clients=self.clients,
            llms=self.llms,
            secrets=self.secrets,
            bus=self.bus,
            teams=self.teams,
            **kwargs,
        )
