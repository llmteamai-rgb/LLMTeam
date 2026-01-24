"""
Per-agent retry policies and circuit breaker integration.

RFC-012: Retry Policies & Circuit Breaker Integration (per-agent level).

Provides:
- RetryPolicy: Per-agent retry configuration
- CircuitBreakerPolicy: Per-agent circuit breaker configuration
- AgentRetryExecutor: Wraps agent execution with retry + circuit breaker
"""

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import asyncio

from llmteam.clients.retry import (
    RetryStrategy,
    ExponentialBackoff,
    LinearBackoff,
    ConstantBackoff,
)
from llmteam.clients.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitState,
)


@dataclass
class RetryPolicy:
    """
    Per-agent retry policy configuration.

    Defines how many times and with what backoff strategy
    an agent's execution should be retried on failure.

    Example:
        policy = RetryPolicy(
            max_retries=3,
            backoff="exponential",
            base_delay=1.0,
            max_delay=30.0,
            retryable_exceptions=(TimeoutError, ConnectionError),
        )
    """

    # Maximum retry attempts (0 = no retries, execute once)
    max_retries: int = 3

    # Backoff strategy: "exponential", "linear", "constant"
    backoff: str = "exponential"

    # Delay settings (seconds)
    base_delay: float = 1.0
    max_delay: float = 30.0
    multiplier: float = 2.0  # For exponential backoff

    # Jitter (randomization) factor (0.0 - 1.0)
    jitter: float = 0.1

    # Exception types that trigger retry (empty = all exceptions)
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)

    # Callback on each retry: (attempt, exception, delay) -> None
    on_retry: Optional[Callable[[int, Exception, float], None]] = None

    def build_strategy(self) -> RetryStrategy:
        """Build RetryStrategy instance from config."""
        if self.backoff == "linear":
            return LinearBackoff(
                base_delay=self.base_delay,
                increment=self.base_delay,
                max_delay=self.max_delay,
            )
        elif self.backoff == "constant":
            return ConstantBackoff(delay=self.base_delay)
        else:  # exponential (default)
            return ExponentialBackoff(
                base_delay=self.base_delay,
                multiplier=self.multiplier,
                max_delay=self.max_delay,
            )


@dataclass
class CircuitBreakerPolicy:
    """
    Per-agent circuit breaker policy configuration.

    Prevents cascading failures by temporarily disabling
    an agent after consecutive failures.

    Example:
        policy = CircuitBreakerPolicy(
            failure_threshold=5,
            recovery_timeout_seconds=30,
            success_threshold=2,
        )
    """

    # Number of failures before opening circuit
    failure_threshold: int = 5

    # Seconds to wait before testing recovery (OPEN -> HALF_OPEN)
    recovery_timeout_seconds: float = 30.0

    # Number of successes in HALF_OPEN to close circuit
    success_threshold: int = 2

    # Time window (seconds) for counting failures
    failure_window_seconds: float = 60.0

    # Exception types that count as failures (empty = all)
    failure_exceptions: Tuple[Type[Exception], ...] = (Exception,)

    # Callback on state change: (old_state, new_state) -> None
    on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None

    def build_circuit_breaker(self, name: str = "") -> CircuitBreaker:
        """Build CircuitBreaker instance from policy."""
        config = CircuitBreakerConfig(
            failure_threshold=self.failure_threshold,
            success_threshold=self.success_threshold,
            recovery_timeout=timedelta(seconds=self.recovery_timeout_seconds),
            failure_window=timedelta(seconds=self.failure_window_seconds),
            failure_exceptions=self.failure_exceptions,
            on_state_change=self.on_state_change,
        )
        return CircuitBreaker(config=config)


@dataclass
class RetryMetrics:
    """Metrics collected during agent retry execution."""

    total_attempts: int = 0
    successful_attempt: int = 0  # Which attempt succeeded (0 = first try)
    retries_performed: int = 0
    total_retry_delay_ms: float = 0.0
    last_error: Optional[str] = None
    circuit_breaker_state: Optional[str] = None
    circuit_breaker_blocked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "total_attempts": self.total_attempts,
            "successful_attempt": self.successful_attempt,
            "retries_performed": self.retries_performed,
            "total_retry_delay_ms": round(self.total_retry_delay_ms, 2),
            "last_error": self.last_error,
            "circuit_breaker_state": self.circuit_breaker_state,
            "circuit_breaker_blocked": self.circuit_breaker_blocked,
        }


class AgentRetryExecutor:
    """
    Executes agent operations with retry and circuit breaker protection.

    Integrates per-agent RetryPolicy and CircuitBreakerPolicy
    into the agent execution pipeline.

    Usage:
        executor = AgentRetryExecutor(
            agent_id="my_agent",
            retry_policy=RetryPolicy(max_retries=3),
            circuit_breaker_policy=CircuitBreakerPolicy(failure_threshold=5),
        )

        result, metrics = await executor.execute(agent._execute, input_data, context)
    """

    def __init__(
        self,
        agent_id: str,
        retry_policy: Optional[RetryPolicy] = None,
        circuit_breaker_policy: Optional[CircuitBreakerPolicy] = None,
    ):
        self._agent_id = agent_id
        self._retry_policy = retry_policy
        self._circuit_breaker_policy = circuit_breaker_policy

        # Build strategy
        self._strategy: Optional[RetryStrategy] = None
        if retry_policy:
            self._strategy = retry_policy.build_strategy()

        # Build circuit breaker
        self._circuit_breaker: Optional[CircuitBreaker] = None
        if circuit_breaker_policy:
            self._circuit_breaker = circuit_breaker_policy.build_circuit_breaker(
                name=f"agent_{agent_id}"
            )

    @property
    def circuit_breaker(self) -> Optional[CircuitBreaker]:
        """Access circuit breaker instance (for metrics/inspection)."""
        return self._circuit_breaker

    @property
    def has_retry(self) -> bool:
        """Check if retry policy is configured."""
        return self._retry_policy is not None and self._retry_policy.max_retries > 0

    @property
    def has_circuit_breaker(self) -> bool:
        """Check if circuit breaker is configured."""
        return self._circuit_breaker is not None

    async def execute(
        self,
        func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Any, RetryMetrics]:
        """
        Execute function with retry and circuit breaker protection.

        Args:
            func: Async function to execute (agent._execute)
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Tuple of (result, RetryMetrics)

        Raises:
            CircuitBreakerOpen: If circuit breaker is open
            Exception: Last exception if all retries exhausted
        """
        metrics = RetryMetrics()

        # Check circuit breaker state
        if self._circuit_breaker:
            metrics.circuit_breaker_state = self._circuit_breaker.state.value
            if self._circuit_breaker.is_open:
                # Check if recovery timeout has passed
                try:
                    await self._circuit_breaker._check_state()
                except CircuitBreakerOpen:
                    metrics.circuit_breaker_blocked = True
                    raise

        # Determine max attempts
        max_retries = self._retry_policy.max_retries if self._retry_policy else 0
        retryable_exceptions = (
            self._retry_policy.retryable_exceptions
            if self._retry_policy
            else (Exception,)
        )

        last_exception: Optional[Exception] = None
        total_delay = 0.0

        for attempt in range(max_retries + 1):
            metrics.total_attempts = attempt + 1

            try:
                # Execute with circuit breaker if available
                if self._circuit_breaker:
                    async with self._circuit_breaker:
                        result = await func(*args, **kwargs)
                else:
                    result = await func(*args, **kwargs)

                # Success
                metrics.successful_attempt = attempt
                metrics.retries_performed = attempt
                if self._circuit_breaker:
                    metrics.circuit_breaker_state = self._circuit_breaker.state.value

                return result, metrics

            except CircuitBreakerOpen:
                # Circuit breaker opened during execution
                metrics.circuit_breaker_blocked = True
                metrics.circuit_breaker_state = CircuitState.OPEN.value
                raise

            except retryable_exceptions as e:
                last_exception = e
                metrics.last_error = str(e)

                # Update circuit breaker state in metrics
                if self._circuit_breaker:
                    metrics.circuit_breaker_state = self._circuit_breaker.state.value

                # Check if we should retry
                if attempt < max_retries:
                    # Calculate delay
                    delay = 0.0
                    if self._strategy:
                        delay = self._strategy.get_delay(attempt)
                        delay = self._strategy.apply_jitter(
                            delay,
                            self._retry_policy.jitter if self._retry_policy else 0.0,
                        )

                    # Notify callback
                    if self._retry_policy and self._retry_policy.on_retry:
                        self._retry_policy.on_retry(attempt + 1, e, delay)

                    # Wait
                    if delay > 0:
                        await asyncio.sleep(delay)
                        total_delay += delay * 1000  # Convert to ms

                    metrics.total_retry_delay_ms = total_delay
                else:
                    # All retries exhausted
                    metrics.retries_performed = attempt
                    metrics.total_retry_delay_ms = total_delay

        # All retries exhausted - raise last exception
        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected: no exception after failed retries")

    def get_circuit_breaker_metrics(self) -> Optional[Dict[str, Any]]:
        """Get circuit breaker metrics if available."""
        if self._circuit_breaker:
            return self._circuit_breaker.get_metrics()
        return None

    async def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker to closed state."""
        if self._circuit_breaker:
            await self._circuit_breaker.reset()
