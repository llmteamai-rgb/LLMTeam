"""
Rate limiting configuration.

This module defines configuration structures for:
- RateLimitConfig: Rate limiting parameters
- CircuitBreakerConfig: Circuit breaker parameters
"""

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Any, Callable, Optional


class RateLimitStrategy(Enum):
    """
    Strategy for handling rate limit exceeded.
    
    WAIT: Block and wait until rate limit allows
    REJECT: Immediately reject the request
    QUEUE: Add to queue for later processing
    FALLBACK: Use fallback handler if available
    """
    WAIT = "wait"
    REJECT = "reject"
    QUEUE = "queue"
    FALLBACK = "fallback"


class CircuitState(Enum):
    """
    Circuit breaker states.
    
    CLOSED: Normal operation, requests pass through
    OPEN: Requests are blocked (too many failures)
    HALF_OPEN: Testing if service has recovered
    """
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class RateLimitConfig:
    """
    Configuration for rate limiting.
    
    Uses token bucket algorithm with configurable rates at different
    time scales (second, minute, hour).
    
    Attributes:
        requests_per_second: Max requests per second
        requests_per_minute: Max requests per minute
        requests_per_hour: Max requests per hour
        burst_size: Max concurrent requests (token bucket capacity)
        strategy: What to do when limit exceeded
        max_wait_seconds: Max time to wait (for WAIT strategy)
        queue_size: Max queue size (for QUEUE strategy)
        retry_count: Number of retries on failure
        retry_base_delay: Base delay for retry (seconds)
        retry_max_delay: Max delay for retry (seconds)
        retry_exponential: Whether to use exponential backoff
        fallback_handler: Function to call on FALLBACK strategy
        fallback_value: Default value to return on FALLBACK
        
    Example:
        config = RateLimitConfig(
            requests_per_minute=100,
            burst_size=10,
            strategy=RateLimitStrategy.QUEUE,
            retry_count=3,
            retry_exponential=True,
        )
    """
    
    # Rate limits
    requests_per_second: float = 10.0
    requests_per_minute: float = 100.0
    requests_per_hour: float = 1000.0
    burst_size: int = 10
    
    # Strategy
    strategy: RateLimitStrategy = RateLimitStrategy.WAIT
    max_wait_seconds: float = 30.0
    queue_size: int = 100
    
    # Retry
    retry_count: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    retry_exponential: bool = True
    
    # Fallback
    fallback_handler: Optional[Callable] = None
    fallback_value: Any = None
    
    def get_min_interval(self) -> float:
        """Get minimum interval between requests (seconds)."""
        return 1.0 / self.requests_per_second
    
    def get_retry_delay(self, attempt: int) -> float:
        """
        Get retry delay for a specific attempt.
        
        Args:
            attempt: Attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        if self.retry_exponential:
            delay = self.retry_base_delay * (2 ** attempt)
        else:
            delay = self.retry_base_delay
        
        return min(delay, self.retry_max_delay)


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for circuit breaker.
    
    The circuit breaker prevents cascading failures by stopping
    requests to a failing service.
    
    Attributes:
        failure_threshold: Number of failures to open circuit
        failure_rate_threshold: Failure rate (0-1) to open circuit
        sample_size: Number of requests to sample for failure rate
        open_timeout: How long to keep circuit open before testing
        half_open_max_requests: Max requests in half-open state
        success_threshold: Successes needed to close circuit from half-open
        
    Example:
        config = CircuitBreakerConfig(
            failure_threshold=5,
            open_timeout=timedelta(seconds=30),
            half_open_max_requests=3,
        )
    """
    
    # Thresholds
    failure_threshold: int = 5
    failure_rate_threshold: float = 0.5
    sample_size: int = 10
    
    # Timeouts
    open_timeout: timedelta = field(default_factory=lambda: timedelta(seconds=30))
    
    # Half-open behavior
    half_open_max_requests: int = 3
    success_threshold: int = 2
    
    def should_open(self, failures: int, total: int) -> bool:
        """
        Check if circuit should open.
        
        Args:
            failures: Number of recent failures
            total: Total recent requests
            
        Returns:
            True if circuit should open
        """
        # Check absolute threshold
        if failures >= self.failure_threshold:
            return True
        
        # Check rate threshold
        if total >= self.sample_size:
            failure_rate = failures / total
            if failure_rate >= self.failure_rate_threshold:
                return True
        
        return False


# Exceptions

class RateLimitError(Exception):
    """Base exception for rate limiting errors."""
    pass


class RateLimitExceeded(RateLimitError):
    """Raised when rate limit is exceeded and strategy is REJECT."""
    
    def __init__(self, name: str, retry_after: float = None):
        self.name = name
        self.retry_after = retry_after
        message = f"Rate limit exceeded for '{name}'"
        if retry_after:
            message += f". Retry after {retry_after:.1f} seconds"
        super().__init__(message)


class CircuitOpenError(RateLimitError):
    """Raised when circuit breaker is open."""
    
    def __init__(self, name: str, retry_after: float = None):
        self.name = name
        self.retry_after = retry_after
        message = f"Circuit breaker open for '{name}'"
        if retry_after:
            message += f". Retry after {retry_after:.1f} seconds"
        super().__init__(message)


class QueueFullError(RateLimitError):
    """Raised when rate limit queue is full."""
    
    def __init__(self, name: str, queue_size: int):
        self.name = name
        self.queue_size = queue_size
        super().__init__(f"Rate limit queue full for '{name}' (max: {queue_size})")
