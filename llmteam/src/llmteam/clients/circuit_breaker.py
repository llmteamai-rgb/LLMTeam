"""
Circuit Breaker Pattern.

Prevents cascading failures by temporarily blocking requests
to failing services.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional
import asyncio


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit is open."""

    def __init__(self, message: str = "Circuit breaker is open") -> None:
        super().__init__(message)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    # Failure threshold before opening circuit
    failure_threshold: int = 5

    # Success threshold to close circuit from half-open
    success_threshold: int = 2

    # Time to wait before moving from open to half-open
    recovery_timeout: timedelta = field(default_factory=lambda: timedelta(seconds=30))

    # Time window for counting failures
    failure_window: timedelta = field(default_factory=lambda: timedelta(minutes=1))

    # Exception types that count as failures
    failure_exceptions: tuple = (Exception,)

    # Callbacks
    on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None
    on_failure: Optional[Callable[[Exception], None]] = None
    on_success: Optional[Callable[[], None]] = None


class CircuitBreaker:
    """
    Circuit breaker implementation.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Requests are blocked immediately
    - HALF_OPEN: Testing if service recovered

    Transitions:
    - CLOSED -> OPEN: When failure_threshold is reached
    - OPEN -> HALF_OPEN: After recovery_timeout
    - HALF_OPEN -> CLOSED: After success_threshold successes
    - HALF_OPEN -> OPEN: On any failure

    Usage:
        breaker = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=timedelta(seconds=30),
        ))

        async def call_service():
            async with breaker:
                return await http_client.get("/api")

        try:
            result = await call_service()
        except CircuitBreakerOpen:
            # Use fallback
            pass
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None) -> None:
        """
        Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration
        """
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._opened_at: Optional[datetime] = None
        self._failures: list[datetime] = []
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing)."""
        return self._state == CircuitState.HALF_OPEN

    async def __aenter__(self) -> "CircuitBreaker":
        """Enter async context."""
        await self._check_state()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit async context."""
        if exc_type is None:
            await self._record_success()
        elif isinstance(exc_val, self.config.failure_exceptions):
            await self._record_failure(exc_val)
        return False  # Don't suppress exceptions

    async def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpen: If circuit is open
        """
        async with self:
            return await func(*args, **kwargs)

    async def _check_state(self) -> None:
        """Check and potentially update circuit state."""
        async with self._lock:
            now = datetime.now()

            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._opened_at and (now - self._opened_at) >= self.config.recovery_timeout:
                    self._transition_to(CircuitState.HALF_OPEN)
                else:
                    raise CircuitBreakerOpen(
                        f"Circuit is open. Will retry after "
                        f"{self.config.recovery_timeout - (now - self._opened_at) if self._opened_at else self.config.recovery_timeout}"
                    )

            elif self._state == CircuitState.HALF_OPEN:
                # Allow request through for testing
                pass

            elif self._state == CircuitState.CLOSED:
                # Clean up old failures outside the window
                cutoff = now - self.config.failure_window
                self._failures = [f for f in self._failures if f > cutoff]

    async def _record_success(self) -> None:
        """Record successful operation."""
        async with self._lock:
            if self.config.on_success:
                self.config.on_success()

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1

                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    async def _record_failure(self, exception: Exception) -> None:
        """Record failed operation."""
        async with self._lock:
            now = datetime.now()
            self._last_failure_time = now
            self._failures.append(now)

            if self.config.on_failure:
                self.config.on_failure(exception)

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately opens circuit
                self._transition_to(CircuitState.OPEN)

            elif self._state == CircuitState.CLOSED:
                # Clean up old failures
                cutoff = now - self.config.failure_window
                self._failures = [f for f in self._failures if f > cutoff]

                # Check if threshold reached
                if len(self._failures) >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state."""
        old_state = self._state
        self._state = new_state

        if new_state == CircuitState.OPEN:
            self._opened_at = datetime.now()
            self._success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._failures.clear()
            self._opened_at = None

        if self.config.on_state_change:
            self.config.on_state_change(old_state, new_state)

    async def reset(self) -> None:
        """Manually reset circuit to closed state."""
        async with self._lock:
            self._transition_to(CircuitState.CLOSED)

    async def trip(self) -> None:
        """Manually trip circuit to open state."""
        async with self._lock:
            self._transition_to(CircuitState.OPEN)

    def get_metrics(self) -> dict[str, Any]:
        """Get circuit breaker metrics."""
        return {
            "state": self._state.value,
            "failure_count": len(self._failures),
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time.isoformat() if self._last_failure_time else None,
            "opened_at": self._opened_at.isoformat() if self._opened_at else None,
        }
