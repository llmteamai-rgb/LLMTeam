"""
HTTP and API Clients Module.

Provides robust HTTP clients with:
- Retry with exponential backoff
- Circuit breaker pattern
- Request/response logging
- Authentication handling
- Rate limiting

Usage:
    from llmteam.clients import HTTPClient, HTTPClientConfig

    client = HTTPClient(HTTPClientConfig(
        base_url="https://api.example.com",
        timeout_seconds=30,
        max_retries=3,
    ))

    response = await client.get("/users")
"""

from llmteam.clients.http import (
    HTTPClient,
    HTTPClientConfig,
    HTTPResponse,
    HTTPError,
    HTTPTimeoutError,
    HTTPConnectionError,
    HTTPRetryExhaustedError,
)

from llmteam.clients.retry import (
    RetryConfig,
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

__all__ = [
    # HTTP Client
    "HTTPClient",
    "HTTPClientConfig",
    "HTTPResponse",
    "HTTPError",
    "HTTPTimeoutError",
    "HTTPConnectionError",
    "HTTPRetryExhaustedError",
    # Retry
    "RetryConfig",
    "RetryStrategy",
    "ExponentialBackoff",
    "LinearBackoff",
    "ConstantBackoff",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpen",
    "CircuitState",
]
