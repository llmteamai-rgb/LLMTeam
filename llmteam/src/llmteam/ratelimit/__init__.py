"""
Rate limiting for llmteam.

This module provides protection for external API calls with:
- Token bucket rate limiting
- Circuit breaker pattern
- Automatic retry with exponential backoff
- Fallback support

Quick Start:
    from llmteam.ratelimit import (
        RateLimitedExecutor,
        RateLimitConfig,
        CircuitBreakerConfig,
        RateLimitStrategy,
    )
    from datetime import timedelta
    
    # Create executor
    executor = RateLimitedExecutor()
    
    # Register endpoint
    executor.register(
        "external_api",
        RateLimitConfig(
            requests_per_minute=100,
            burst_size=10,
            strategy=RateLimitStrategy.QUEUE,
            retry_count=3,
        ),
        CircuitBreakerConfig(
            failure_threshold=5,
            open_timeout=timedelta(seconds=30),
        ),
    )
    
    # Execute with protection
    result = await executor.execute(
        "external_api",
        call_api,
        param1="value",
    )
    
    if result.success:
        print(result.value)
    else:
        print(f"Failed: {result.error}")
"""

from llmteam.ratelimit.config import (
    RateLimitConfig,
    CircuitBreakerConfig,
    RateLimitStrategy,
    CircuitState,
    RateLimitError,
    RateLimitExceeded,
    CircuitOpenError,
    QueueFullError,
)

from llmteam.ratelimit.limiter import (
    RateLimiter,
    RateLimiterStats,
)

from llmteam.ratelimit.circuit import (
    CircuitBreaker,
    CircuitBreakerStats,
)

from llmteam.ratelimit.executor import (
    RateLimitedExecutor,
    ExecutionResult,
    EndpointStats,
)

__all__ = [
    # Config
    "RateLimitConfig",
    "CircuitBreakerConfig",
    "RateLimitStrategy",
    "CircuitState",
    
    # Exceptions
    "RateLimitError",
    "RateLimitExceeded",
    "CircuitOpenError",
    "QueueFullError",
    
    # Rate Limiter
    "RateLimiter",
    "RateLimiterStats",
    
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerStats",
    
    # Executor
    "RateLimitedExecutor",
    "ExecutionResult",
    "EndpointStats",
]
