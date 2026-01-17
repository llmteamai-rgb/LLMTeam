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

from llmteam.runtime.protocols import Store, Client, LLMProvider, SecretsProvider
from llmteam.runtime.registries import StoreRegistry, ClientRegistry, LLMRegistry
from llmteam.runtime.exceptions import ResourceNotFoundError, RuntimeContextError

if TYPE_CHECKING:
    from llmteam.ratelimit import RateLimitedExecutor
    from llmteam.audit import AuditTrail


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
        return self.stores.get(store_ref)

    def resolve_client(self, client_ref: str) -> Client:
        """Resolve client by reference."""
        return self.clients.get(client_ref)

    def resolve_llm(self, llm_ref: str) -> LLMProvider:
        """Resolve LLM provider by reference."""
        return self.llms.get(llm_ref)

    async def resolve_secret(self, secret_ref: str) -> str:
        """Resolve secret by reference."""
        if not self.secrets:
            raise ResourceNotFoundError("SecretsProvider not configured")
        return await self.secrets.get_secret(secret_ref)

    def child_context(self, step_id: str) -> "StepContext":
        """Create child context for a step."""
        return StepContext(
            runtime=self,
            step_id=step_id,
        )

    def copy(self, **overrides: Any) -> "RuntimeContext":
        """Create a copy with optional overrides."""
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

    async def __aenter__(self) -> RuntimeContext:
        return self.__enter__()

    async def __aexit__(self, *args: Any) -> None:
        self.__exit__(*args)
