"""
HTTP Client with Retry and Circuit Breaker.

Provides a robust HTTP client for REST API calls.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, Union
import asyncio
import json as json_module

from llmteam.clients.retry import RetryConfig, RetryStrategy, ExponentialBackoff
from llmteam.clients.circuit_breaker import CircuitBreaker, CircuitBreakerConfig


class HTTPError(Exception):
    """Base HTTP error."""

    def __init__(
        self,
        message: str,
        status_code: int = 0,
        response_body: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class HTTPTimeoutError(HTTPError):
    """Request timed out."""
    pass


class HTTPConnectionError(HTTPError):
    """Connection failed."""
    pass


class HTTPRetryExhaustedError(HTTPError):
    """All retries exhausted."""
    pass


@dataclass
class HTTPResponse:
    """HTTP response wrapper."""

    status_code: int
    headers: dict[str, str]
    body: bytes
    elapsed_ms: int

    # Request info
    method: str = ""
    url: str = ""

    @property
    def text(self) -> str:
        """Get response as text."""
        return self.body.decode("utf-8", errors="replace")

    @property
    def json(self) -> Any:
        """Parse response as JSON."""
        return json_module.loads(self.body)

    @property
    def ok(self) -> bool:
        """Check if response is successful (2xx)."""
        return 200 <= self.status_code < 300


@dataclass
class HTTPClientConfig:
    """Configuration for HTTP client."""

    # Base URL for all requests
    base_url: str = ""

    # Timeouts
    timeout_seconds: float = 30.0
    connect_timeout_seconds: float = 10.0

    # Default headers
    headers: dict[str, str] = field(default_factory=dict)

    # Authentication
    auth_token: str = ""
    auth_type: str = "Bearer"  # "Bearer", "Basic", "ApiKey"
    api_key_header: str = "X-API-Key"

    # Retry configuration
    retry_config: Optional[RetryConfig] = None
    retry_strategy: Optional[RetryStrategy] = None

    # Circuit breaker
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None

    # Request/response logging
    log_requests: bool = False
    log_responses: bool = False
    log_func: Optional[Callable[[str], None]] = None

    # SSL verification
    verify_ssl: bool = True

    # User agent
    user_agent: str = "llmteam-http-client/1.0"


class HTTPClient:
    """
    HTTP client with retry and circuit breaker support.

    Features:
    - Automatic retries with configurable backoff
    - Circuit breaker for fault tolerance
    - Request/response logging
    - Authentication handling
    - Timeout management

    Usage:
        client = HTTPClient(HTTPClientConfig(
            base_url="https://api.example.com",
            timeout_seconds=30,
            auth_token="my-token",
        ))

        # Simple GET
        response = await client.get("/users")

        # POST with JSON body
        response = await client.post("/users", json={"name": "John"})

        # With custom headers
        response = await client.get("/protected", headers={"X-Custom": "value"})
    """

    def __init__(self, config: Optional[HTTPClientConfig] = None) -> None:
        """
        Initialize HTTP client.

        Args:
            config: Client configuration
        """
        self.config = config or HTTPClientConfig()

        # Initialize retry
        self._retry_config = self.config.retry_config or RetryConfig()
        self._retry_strategy = self.config.retry_strategy or ExponentialBackoff()

        # Initialize circuit breaker
        self._circuit_breaker: Optional[CircuitBreaker] = None
        if self.config.circuit_breaker_config:
            self._circuit_breaker = CircuitBreaker(self.config.circuit_breaker_config)

        # Session management
        self._session: Optional[Any] = None

    async def _get_session(self) -> Any:
        """Get or create aiohttp session."""
        if self._session is None:
            try:
                import aiohttp

                timeout = aiohttp.ClientTimeout(
                    total=self.config.timeout_seconds,
                    connect=self.config.connect_timeout_seconds,
                )
                connector = aiohttp.TCPConnector(ssl=self.config.verify_ssl)
                self._session = aiohttp.ClientSession(
                    timeout=timeout,
                    connector=connector,
                )
            except ImportError:
                raise ImportError(
                    "aiohttp is required for HTTP client. "
                    "Install with: pip install aiohttp"
                )
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "HTTPClient":
        """Enter async context."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context."""
        await self.close()

    def _build_url(self, path: str) -> str:
        """Build full URL from path."""
        if path.startswith(("http://", "https://")):
            return path
        base = self.config.base_url.rstrip("/")
        path = path.lstrip("/")
        return f"{base}/{path}"

    def _build_headers(self, extra_headers: Optional[dict] = None) -> dict[str, str]:
        """Build request headers."""
        headers = {
            "User-Agent": self.config.user_agent,
            **self.config.headers,
        }

        # Add authentication
        if self.config.auth_token:
            if self.config.auth_type == "Bearer":
                headers["Authorization"] = f"Bearer {self.config.auth_token}"
            elif self.config.auth_type == "Basic":
                headers["Authorization"] = f"Basic {self.config.auth_token}"
            elif self.config.auth_type == "ApiKey":
                headers[self.config.api_key_header] = self.config.auth_token

        # Add extra headers
        if extra_headers:
            headers.update(extra_headers)

        return headers

    def _log(self, message: str) -> None:
        """Log message."""
        if self.config.log_func:
            self.config.log_func(message)
        else:
            print(message)

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        json: Optional[Any] = None,
        data: Optional[Union[dict, bytes]] = None,
        timeout: Optional[float] = None,
    ) -> HTTPResponse:
        """
        Make HTTP request.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            path: URL path (or full URL)
            params: Query parameters
            headers: Additional headers
            json: JSON body
            data: Form data or raw bytes
            timeout: Override timeout

        Returns:
            HTTPResponse

        Raises:
            HTTPError: On HTTP errors
            HTTPTimeoutError: On timeout
            HTTPConnectionError: On connection errors
        """
        url = self._build_url(path)
        request_headers = self._build_headers(headers)

        if self.config.log_requests:
            self._log(f"HTTP {method} {url}")

        # Execute with retry and circuit breaker
        return await self._execute_with_resilience(
            method=method,
            url=url,
            params=params,
            headers=request_headers,
            json=json,
            data=data,
            timeout=timeout,
        )

    async def _execute_with_resilience(
        self,
        method: str,
        url: str,
        params: Optional[dict],
        headers: dict,
        json: Optional[Any],
        data: Optional[Union[dict, bytes]],
        timeout: Optional[float],
    ) -> HTTPResponse:
        """Execute request with retry and circuit breaker."""
        last_error: Optional[Exception] = None

        for attempt in range(self._retry_config.max_retries + 1):
            try:
                # Check circuit breaker
                if self._circuit_breaker:
                    await self._circuit_breaker._check_state()

                # Execute request
                response = await self._execute_request(
                    method, url, params, headers, json, data, timeout
                )

                # Record success for circuit breaker
                if self._circuit_breaker:
                    await self._circuit_breaker._record_success()

                # Check if we should retry based on status code
                if not response.ok and response.status_code in self._retry_config.retry_status_codes:
                    if attempt < self._retry_config.max_retries:
                        delay = self._retry_strategy.get_delay(attempt)
                        delay = self._retry_strategy.apply_jitter(delay, self._retry_config.jitter)

                        if self._retry_config.on_retry:
                            error = HTTPError(
                                f"HTTP {response.status_code}",
                                response.status_code,
                            )
                            self._retry_config.on_retry(attempt + 1, error, delay)

                        await asyncio.sleep(delay)
                        continue

                return response

            except self._retry_config.retry_exceptions as e:
                last_error = e

                # Record failure for circuit breaker
                if self._circuit_breaker:
                    await self._circuit_breaker._record_failure(e)

                if attempt < self._retry_config.max_retries:
                    delay = self._retry_strategy.get_delay(attempt)
                    delay = self._retry_strategy.apply_jitter(delay, self._retry_config.jitter)

                    if self._retry_config.on_retry:
                        self._retry_config.on_retry(attempt + 1, e, delay)

                    await asyncio.sleep(delay)

        # All retries exhausted
        if last_error:
            raise HTTPRetryExhaustedError(
                f"All {self._retry_config.max_retries + 1} attempts failed: {last_error}"
            )
        raise HTTPRetryExhaustedError("Request failed after all retries")

    async def _execute_request(
        self,
        method: str,
        url: str,
        params: Optional[dict],
        headers: dict,
        json: Optional[Any],
        data: Optional[Union[dict, bytes]],
        timeout: Optional[float],
    ) -> HTTPResponse:
        """Execute single HTTP request."""
        import aiohttp

        session = await self._get_session()
        start_time = datetime.now()

        try:
            # Prepare request kwargs
            kwargs: dict[str, Any] = {
                "params": params,
                "headers": headers,
            }

            if json is not None:
                kwargs["json"] = json
            elif data is not None:
                kwargs["data"] = data

            if timeout:
                kwargs["timeout"] = aiohttp.ClientTimeout(total=timeout)

            async with session.request(method, url, **kwargs) as response:
                body = await response.read()
                elapsed = datetime.now() - start_time

                http_response = HTTPResponse(
                    status_code=response.status,
                    headers=dict(response.headers),
                    body=body,
                    elapsed_ms=int(elapsed.total_seconds() * 1000),
                    method=method,
                    url=url,
                )

                if self.config.log_responses:
                    self._log(
                        f"HTTP {method} {url} -> {response.status} "
                        f"({http_response.elapsed_ms}ms)"
                    )

                return http_response

        except asyncio.TimeoutError:
            raise HTTPTimeoutError(f"Request to {url} timed out")
        except aiohttp.ClientConnectorError as e:
            raise HTTPConnectionError(f"Connection to {url} failed: {e}")

    # Convenience methods

    async def get(
        self,
        path: str,
        *,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> HTTPResponse:
        """Make GET request."""
        return await self.request("GET", path, params=params, headers=headers, timeout=timeout)

    async def post(
        self,
        path: str,
        *,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        json: Optional[Any] = None,
        data: Optional[Union[dict, bytes]] = None,
        timeout: Optional[float] = None,
    ) -> HTTPResponse:
        """Make POST request."""
        return await self.request(
            "POST", path, params=params, headers=headers, json=json, data=data, timeout=timeout
        )

    async def put(
        self,
        path: str,
        *,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        json: Optional[Any] = None,
        data: Optional[Union[dict, bytes]] = None,
        timeout: Optional[float] = None,
    ) -> HTTPResponse:
        """Make PUT request."""
        return await self.request(
            "PUT", path, params=params, headers=headers, json=json, data=data, timeout=timeout
        )

    async def patch(
        self,
        path: str,
        *,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        json: Optional[Any] = None,
        data: Optional[Union[dict, bytes]] = None,
        timeout: Optional[float] = None,
    ) -> HTTPResponse:
        """Make PATCH request."""
        return await self.request(
            "PATCH", path, params=params, headers=headers, json=json, data=data, timeout=timeout
        )

    async def delete(
        self,
        path: str,
        *,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> HTTPResponse:
        """Make DELETE request."""
        return await self.request("DELETE", path, params=params, headers=headers, timeout=timeout)

    async def head(
        self,
        path: str,
        *,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> HTTPResponse:
        """Make HEAD request."""
        return await self.request("HEAD", path, params=params, headers=headers, timeout=timeout)
