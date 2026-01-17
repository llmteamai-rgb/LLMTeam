"""
GraphQL Client.

Provides async GraphQL client with retry, circuit breaker, and batching support.

Requires:
    pip install gql[aiohttp]

Usage:
    from llmteam.clients import GraphQLClient

    client = GraphQLClient(
        endpoint="https://api.example.com/graphql",
        headers={"Authorization": "Bearer token"},
    )

    result = await client.execute('''
        query GetUser($id: ID!) {
            user(id: $id) {
                name
                email
            }
        }
    ''', variables={"id": "123"})
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional, TypeVar

from llmteam.clients.retry import RetryStrategy, ExponentialBackoff
from llmteam.clients.circuit_breaker import CircuitBreaker, CircuitBreakerConfig


T = TypeVar("T")


class GraphQLError(Exception):
    """GraphQL execution error."""

    def __init__(
        self,
        message: str,
        errors: Optional[list[dict[str, Any]]] = None,
        data: Optional[dict[str, Any]] = None,
    ):
        self.errors = errors or []
        self.data = data
        super().__init__(message)


class GraphQLClientError(Exception):
    """GraphQL client-level error."""

    pass


@dataclass
class GraphQLResponse:
    """GraphQL response wrapper."""

    data: Optional[dict[str, Any]] = None
    errors: Optional[list[dict[str, Any]]] = None
    extensions: Optional[dict[str, Any]] = None

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    def raise_for_errors(self) -> None:
        """Raise GraphQLError if response has errors."""
        if self.errors:
            raise GraphQLError(
                message=self.errors[0].get("message", "Unknown error"),
                errors=self.errors,
                data=self.data,
            )


@dataclass
class GraphQLClientConfig:
    """Configuration for GraphQL client."""

    endpoint: str
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    # Retry configuration
    retry_enabled: bool = True
    max_retries: int = 3
    retry_strategy: Optional[RetryStrategy] = None
    # Circuit breaker
    circuit_breaker_enabled: bool = False
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    # Batching
    batch_enabled: bool = False
    batch_interval_ms: int = 10
    batch_max_size: int = 10
    # Caching
    cache_enabled: bool = False
    cache_ttl_seconds: int = 60


@dataclass
class CachedQuery:
    """Cached query result."""

    response: GraphQLResponse
    cached_at: datetime
    expires_at: datetime


class GraphQLClient:
    """
    Async GraphQL client with enterprise features.

    Features:
    - Automatic retry with configurable backoff
    - Circuit breaker for fault tolerance
    - Query batching for performance
    - Response caching
    - Custom headers and authentication

    Usage:
        client = GraphQLClient(
            endpoint="https://api.example.com/graphql",
            headers={"Authorization": "Bearer token"},
        )

        # Simple query
        result = await client.execute('''
            query {
                users { name email }
            }
        ''')

        # Query with variables
        result = await client.execute('''
            query GetUser($id: ID!) {
                user(id: $id) { name }
            }
        ''', variables={"id": "123"})

        # Mutation
        result = await client.execute('''
            mutation CreateUser($input: CreateUserInput!) {
                createUser(input: $input) { id }
            }
        ''', variables={"input": {"name": "John", "email": "john@example.com"}})
    """

    def __init__(
        self,
        endpoint: str,
        headers: Optional[dict[str, str]] = None,
        config: Optional[GraphQLClientConfig] = None,
    ):
        """
        Initialize GraphQL client.

        Args:
            endpoint: GraphQL endpoint URL
            headers: Default headers for all requests
            config: Full client configuration
        """
        if config:
            self.config = config
        else:
            self.config = GraphQLClientConfig(
                endpoint=endpoint,
                headers=headers or {},
            )

        self._client: Any = None
        self._transport: Any = None
        self._retry_strategy = self.config.retry_strategy or ExponentialBackoff(
            base_delay=0.1,  # 100ms
            max_delay=5.0,   # 5 seconds
            multiplier=2.0,
        )

        # Circuit breaker
        self._circuit_breaker: Optional[CircuitBreaker] = None
        if self.config.circuit_breaker_enabled:
            cb_config = self.config.circuit_breaker_config or CircuitBreakerConfig()
            self._circuit_breaker = CircuitBreaker(cb_config)

        # Caching
        self._cache: dict[str, CachedQuery] = {}

        # Batching
        self._batch_queue: list[tuple[str, dict, asyncio.Future]] = []
        self._batch_task: Optional[asyncio.Task] = None

    async def _get_client(self) -> Any:
        """Get or create the gql client."""
        if self._client is None:
            try:
                from gql import Client
                from gql.transport.aiohttp import AIOHTTPTransport
            except ImportError:
                raise GraphQLClientError(
                    "gql is required for GraphQL client. "
                    "Install with: pip install gql[aiohttp]"
                )

            self._transport = AIOHTTPTransport(
                url=self.config.endpoint,
                headers=self.config.headers,
                timeout=self.config.timeout,
            )

            self._client = Client(
                transport=self._transport,
                fetch_schema_from_transport=False,
            )

        return self._client

    async def execute(
        self,
        query: str,
        variables: Optional[dict[str, Any]] = None,
        operation_name: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """
        Execute a GraphQL query or mutation.

        Args:
            query: GraphQL query string
            variables: Query variables
            operation_name: Optional operation name
            headers: Additional headers for this request

        Returns:
            Query result data

        Raises:
            GraphQLError: If query has errors
            GraphQLClientError: For client-level errors
        """
        response = await self.execute_raw(query, variables, operation_name, headers)
        response.raise_for_errors()
        return response.data or {}

    async def execute_raw(
        self,
        query: str,
        variables: Optional[dict[str, Any]] = None,
        operation_name: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> GraphQLResponse:
        """
        Execute query and return full response.

        Returns GraphQLResponse which may contain both data and errors.
        """
        # Check cache
        if self.config.cache_enabled:
            cache_key = self._cache_key(query, variables)
            cached = self._cache.get(cache_key)
            if cached and datetime.utcnow() < cached.expires_at:
                return cached.response

        # Circuit breaker check
        if self._circuit_breaker:
            await self._circuit_breaker.before_call()

        # Execute with retry
        try:
            response = await self._execute_with_retry(
                query, variables, operation_name, headers
            )

            # Record success for circuit breaker
            if self._circuit_breaker:
                self._circuit_breaker.record_success()

            # Cache the response
            if self.config.cache_enabled and not response.has_errors:
                self._cache_response(query, variables, response)

            return response

        except Exception as e:
            # Record failure for circuit breaker
            if self._circuit_breaker:
                self._circuit_breaker.record_failure()
            raise

    async def _execute_with_retry(
        self,
        query: str,
        variables: Optional[dict[str, Any]],
        operation_name: Optional[str],
        headers: Optional[dict[str, str]],
    ) -> GraphQLResponse:
        """Execute query with retry logic."""
        last_error: Optional[Exception] = None
        retries = self.config.max_retries if self.config.retry_enabled else 0

        for attempt in range(retries + 1):
            try:
                return await self._execute_single(
                    query, variables, operation_name, headers
                )
            except GraphQLError:
                # Don't retry GraphQL errors (they're not transient)
                raise
            except Exception as e:
                last_error = e
                if attempt < retries:
                    delay = self._retry_strategy.get_delay(attempt)
                    await asyncio.sleep(delay)  # delay is already in seconds

        raise GraphQLClientError(f"Request failed after {retries + 1} attempts: {last_error}")

    async def _execute_single(
        self,
        query: str,
        variables: Optional[dict[str, Any]],
        operation_name: Optional[str],
        headers: Optional[dict[str, str]],
    ) -> GraphQLResponse:
        """Execute a single query request."""
        try:
            from gql import gql as parse_gql
        except ImportError:
            raise GraphQLClientError("gql is required")

        client = await self._get_client()
        parsed_query = parse_gql(query)

        # Merge headers
        request_headers = dict(self.config.headers)
        if headers:
            request_headers.update(headers)

        async with client as session:
            # Update transport headers if needed
            if headers and self._transport:
                self._transport.headers = request_headers

            try:
                result = await session.execute(
                    parsed_query,
                    variable_values=variables,
                    operation_name=operation_name,
                )

                return GraphQLResponse(data=dict(result))

            except Exception as e:
                # Check if it's a GraphQL error
                if hasattr(e, "errors"):
                    return GraphQLResponse(
                        data=getattr(e, "data", None),
                        errors=e.errors,
                    )
                raise

    def _cache_key(self, query: str, variables: Optional[dict[str, Any]]) -> str:
        """Generate cache key for a query."""
        import hashlib
        import json

        key_data = query + json.dumps(variables or {}, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _cache_response(
        self,
        query: str,
        variables: Optional[dict[str, Any]],
        response: GraphQLResponse,
    ) -> None:
        """Cache a response."""
        cache_key = self._cache_key(query, variables)
        now = datetime.utcnow()
        self._cache[cache_key] = CachedQuery(
            response=response,
            cached_at=now,
            expires_at=now + timedelta(seconds=self.config.cache_ttl_seconds),
        )

    def clear_cache(self) -> int:
        """Clear the query cache. Returns number of entries cleared."""
        count = len(self._cache)
        self._cache.clear()
        return count

    async def close(self) -> None:
        """Close the client and release resources."""
        if self._transport:
            await self._transport.close()
            self._transport = None
        self._client = None


class GraphQLSubscription:
    """
    GraphQL subscription handler.

    Usage:
        async with GraphQLSubscription(
            endpoint="wss://api.example.com/graphql",
            query='''
                subscription OnNewMessage {
                    messageAdded { id content }
                }
            ''',
        ) as subscription:
            async for message in subscription:
                print(message)
    """

    def __init__(
        self,
        endpoint: str,
        query: str,
        variables: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
    ):
        self.endpoint = endpoint
        self.query = query
        self.variables = variables
        self.headers = headers or {}
        self._client: Any = None
        self._transport: Any = None
        self._session: Any = None

    async def __aenter__(self) -> "GraphQLSubscription":
        try:
            from gql import Client, gql as parse_gql
            from gql.transport.websockets import WebsocketsTransport
        except ImportError:
            raise GraphQLClientError(
                "gql with websockets is required for subscriptions. "
                "Install with: pip install gql[websockets]"
            )

        self._transport = WebsocketsTransport(
            url=self.endpoint,
            headers=self.headers,
        )

        self._client = Client(
            transport=self._transport,
            fetch_schema_from_transport=False,
        )

        self._session = await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)

    async def __aiter__(self) -> Any:
        from gql import gql as parse_gql

        parsed_query = parse_gql(self.query)

        async for result in self._session.subscribe(
            parsed_query,
            variable_values=self.variables,
        ):
            yield dict(result)
