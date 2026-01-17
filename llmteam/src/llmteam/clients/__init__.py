"""
HTTP and API Clients Module.

Provides robust HTTP, GraphQL, and gRPC clients with:
- Retry with exponential backoff
- Circuit breaker pattern
- Request/response logging
- Authentication handling
- Rate limiting

Usage:
    # HTTP Client
    from llmteam.clients import HTTPClient, HTTPClientConfig

    client = HTTPClient(HTTPClientConfig(
        base_url="https://api.example.com",
        timeout_seconds=30,
        max_retries=3,
    ))
    response = await client.get("/users")

    # GraphQL Client
    from llmteam.clients import GraphQLClient

    client = GraphQLClient(endpoint="https://api.example.com/graphql")
    result = await client.execute('query { users { name } }')

    # gRPC Client
    from llmteam.clients import GRPCClient

    async with GRPCClient(target="localhost:50051") as client:
        response = await client.unary_call("greeter.Greeter", "SayHello", {"name": "World"})
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

from llmteam.clients.graphql import (
    GraphQLClient,
    GraphQLClientConfig,
    GraphQLResponse,
    GraphQLError,
    GraphQLClientError,
    GraphQLSubscription,
)

from llmteam.clients.grpc import (
    GRPCClient,
    GRPCClientConfig,
    GRPCError,
    GRPCClientError,
    CallOptions,
    ChannelState,
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
    # GraphQL
    "GraphQLClient",
    "GraphQLClientConfig",
    "GraphQLResponse",
    "GraphQLError",
    "GraphQLClientError",
    "GraphQLSubscription",
    # gRPC
    "GRPCClient",
    "GRPCClientConfig",
    "GRPCError",
    "GRPCClientError",
    "CallOptions",
    "ChannelState",
]
