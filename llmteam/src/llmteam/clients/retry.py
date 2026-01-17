"""
Retry Strategies.

Provides configurable retry logic with various backoff strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Set
import asyncio
import random


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    # Maximum retry attempts (0 = no retries)
    max_retries: int = 3

    # Retryable HTTP status codes
    retry_status_codes: Set[int] = field(
        default_factory=lambda: {408, 429, 500, 502, 503, 504}
    )

    # Retryable exception types
    retry_exceptions: tuple = (ConnectionError, TimeoutError)

    # Whether to retry on timeout
    retry_on_timeout: bool = True

    # Jitter (randomization) factor (0.0 - 1.0)
    jitter: float = 0.1

    # Callback for retry events
    on_retry: Optional[Callable[[int, Exception, float], None]] = None


class RetryStrategy(ABC):
    """Base class for retry backoff strategies."""

    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """
        Get delay in seconds for the given attempt.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds before next retry
        """
        pass

    def apply_jitter(self, delay: float, jitter: float) -> float:
        """Apply jitter to delay."""
        if jitter <= 0:
            return delay
        jitter_range = delay * jitter
        return delay + random.uniform(-jitter_range, jitter_range)


class ExponentialBackoff(RetryStrategy):
    """
    Exponential backoff strategy.

    Delay increases exponentially: base * (multiplier ^ attempt)

    Example with base=1, multiplier=2:
        Attempt 0: 1s
        Attempt 1: 2s
        Attempt 2: 4s
        Attempt 3: 8s
    """

    def __init__(
        self,
        base_delay: float = 1.0,
        multiplier: float = 2.0,
        max_delay: float = 60.0,
    ) -> None:
        """
        Initialize exponential backoff.

        Args:
            base_delay: Initial delay in seconds
            multiplier: Exponential multiplier
            max_delay: Maximum delay in seconds
        """
        self.base_delay = base_delay
        self.multiplier = multiplier
        self.max_delay = max_delay

    def get_delay(self, attempt: int) -> float:
        """Get exponential delay."""
        delay = self.base_delay * (self.multiplier ** attempt)
        return min(delay, self.max_delay)


class LinearBackoff(RetryStrategy):
    """
    Linear backoff strategy.

    Delay increases linearly: base + (increment * attempt)

    Example with base=1, increment=2:
        Attempt 0: 1s
        Attempt 1: 3s
        Attempt 2: 5s
        Attempt 3: 7s
    """

    def __init__(
        self,
        base_delay: float = 1.0,
        increment: float = 1.0,
        max_delay: float = 60.0,
    ) -> None:
        """
        Initialize linear backoff.

        Args:
            base_delay: Initial delay in seconds
            increment: Delay increment per attempt
            max_delay: Maximum delay in seconds
        """
        self.base_delay = base_delay
        self.increment = increment
        self.max_delay = max_delay

    def get_delay(self, attempt: int) -> float:
        """Get linear delay."""
        delay = self.base_delay + (self.increment * attempt)
        return min(delay, self.max_delay)


class ConstantBackoff(RetryStrategy):
    """
    Constant backoff strategy.

    Same delay for all attempts.
    """

    def __init__(self, delay: float = 1.0) -> None:
        """
        Initialize constant backoff.

        Args:
            delay: Constant delay in seconds
        """
        self.delay = delay

    def get_delay(self, attempt: int) -> float:
        """Get constant delay."""
        return self.delay


class RetryExecutor:
    """
    Executes operations with retry logic.

    Usage:
        executor = RetryExecutor(
            config=RetryConfig(max_retries=3),
            strategy=ExponentialBackoff(),
        )

        result = await executor.execute(my_async_function, arg1, arg2)
    """

    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        strategy: Optional[RetryStrategy] = None,
    ) -> None:
        """
        Initialize retry executor.

        Args:
            config: Retry configuration
            strategy: Backoff strategy
        """
        self.config = config or RetryConfig()
        self.strategy = strategy or ExponentialBackoff()

    async def execute(
        self,
        func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Last exception if all retries exhausted
        """
        last_exception: Optional[Exception] = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await func(*args, **kwargs)

            except self.config.retry_exceptions as e:
                last_exception = e

                if attempt < self.config.max_retries:
                    delay = self.strategy.get_delay(attempt)
                    delay = self.strategy.apply_jitter(delay, self.config.jitter)

                    if self.config.on_retry:
                        self.config.on_retry(attempt + 1, e, delay)

                    await asyncio.sleep(delay)

        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected: no exception after failed retries")

    def should_retry_status(self, status_code: int) -> bool:
        """Check if status code is retryable."""
        return status_code in self.config.retry_status_codes
