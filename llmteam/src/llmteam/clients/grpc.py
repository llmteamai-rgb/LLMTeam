"""
gRPC Client.

Provides async gRPC client with channel management, retry, and health checks.

Requires:
    pip install grpcio grpcio-tools

Usage:
    from llmteam.clients import GRPCClient

    async with GRPCClient(target="localhost:50051") as client:
        # Make unary call
        response = await client.unary_call(
            service="greeter.Greeter",
            method="SayHello",
            request={"name": "World"},
        )

        # Stream responses
        async for response in client.server_streaming_call(
            service="chat.Chat",
            method="StreamMessages",
            request={"room_id": "123"},
        ):
            print(response)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Callable, Optional, TypeVar
from enum import Enum

from llmteam.clients.retry import RetryStrategy, ExponentialBackoff
from llmteam.clients.circuit_breaker import CircuitBreaker, CircuitBreakerConfig


T = TypeVar("T")


class GRPCError(Exception):
    """gRPC call error."""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[str] = None,
    ):
        self.code = code
        self.details = details
        super().__init__(message)


class GRPCClientError(Exception):
    """gRPC client-level error."""

    pass


class ChannelState(Enum):
    """gRPC channel state."""

    IDLE = "idle"
    CONNECTING = "connecting"
    READY = "ready"
    TRANSIENT_FAILURE = "transient_failure"
    SHUTDOWN = "shutdown"


@dataclass
class GRPCClientConfig:
    """Configuration for gRPC client."""

    target: str
    secure: bool = False
    # Credentials
    root_certificates: Optional[bytes] = None
    private_key: Optional[bytes] = None
    certificate_chain: Optional[bytes] = None
    # Channel options
    options: dict[str, Any] = field(default_factory=dict)
    # Timeouts
    default_timeout: float = 30.0
    connect_timeout: float = 10.0
    # Retry configuration
    retry_enabled: bool = True
    max_retries: int = 3
    retry_strategy: Optional[RetryStrategy] = None
    # Circuit breaker
    circuit_breaker_enabled: bool = False
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    # Health check
    health_check_enabled: bool = False
    health_check_service: str = ""
    # Interceptors
    interceptors: list[Any] = field(default_factory=list)
    # Metadata
    default_metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class CallOptions:
    """Options for a single gRPC call."""

    timeout: Optional[float] = None
    metadata: Optional[dict[str, str]] = None
    wait_for_ready: bool = False
    compression: Optional[str] = None


class GRPCClient:
    """
    Async gRPC client with enterprise features.

    Features:
    - Automatic channel management
    - Secure (TLS) and insecure connections
    - Retry with configurable backoff
    - Circuit breaker for fault tolerance
    - Health checks
    - Metadata (headers) support
    - Interceptors

    Usage:
        # Insecure connection
        client = GRPCClient(target="localhost:50051")

        # Secure connection
        client = GRPCClient(
            target="api.example.com:443",
            secure=True,
        )

        # With custom config
        config = GRPCClientConfig(
            target="localhost:50051",
            retry_enabled=True,
            max_retries=3,
            default_metadata={"authorization": "Bearer token"},
        )
        client = GRPCClient(config=config)

        async with client:
            result = await client.unary_call(
                service="greeter.Greeter",
                method="SayHello",
                request={"name": "World"},
            )
    """

    def __init__(
        self,
        target: Optional[str] = None,
        secure: bool = False,
        config: Optional[GRPCClientConfig] = None,
    ):
        """
        Initialize gRPC client.

        Args:
            target: Server address (host:port)
            secure: Use TLS
            config: Full client configuration
        """
        if config:
            self.config = config
        else:
            self.config = GRPCClientConfig(
                target=target or "localhost:50051",
                secure=secure,
            )

        self._channel: Any = None
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

    async def __aenter__(self) -> "GRPCClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def connect(self) -> None:
        """Establish connection to the gRPC server."""
        if self._channel is not None:
            return

        try:
            import grpc.aio
        except ImportError:
            raise GRPCClientError(
                "grpcio is required for gRPC client. "
                "Install with: pip install grpcio"
            )

        # Build channel options
        options = list(self.config.options.items())

        if self.config.secure:
            # Secure channel
            if self.config.root_certificates or self.config.private_key:
                credentials = grpc.ssl_channel_credentials(
                    root_certificates=self.config.root_certificates,
                    private_key=self.config.private_key,
                    certificate_chain=self.config.certificate_chain,
                )
            else:
                credentials = grpc.ssl_channel_credentials()

            self._channel = grpc.aio.secure_channel(
                self.config.target,
                credentials,
                options=options if options else None,
                interceptors=self.config.interceptors if self.config.interceptors else None,
            )
        else:
            # Insecure channel
            self._channel = grpc.aio.insecure_channel(
                self.config.target,
                options=options if options else None,
                interceptors=self.config.interceptors if self.config.interceptors else None,
            )

        # Wait for channel to be ready
        try:
            await asyncio.wait_for(
                self._channel.channel_ready(),
                timeout=self.config.connect_timeout,
            )
        except asyncio.TimeoutError:
            raise GRPCClientError(
                f"Failed to connect to {self.config.target} within {self.config.connect_timeout}s"
            )

    async def close(self) -> None:
        """Close the channel."""
        if self._channel:
            await self._channel.close()
            self._channel = None

    def _get_channel(self) -> Any:
        """Get the channel, raising if not connected."""
        if self._channel is None:
            raise GRPCClientError("Client not connected. Use 'async with' or call connect()")
        return self._channel

    async def unary_call(
        self,
        service: str,
        method: str,
        request: dict[str, Any],
        request_type: Optional[Any] = None,
        response_type: Optional[Any] = None,
        options: Optional[CallOptions] = None,
    ) -> dict[str, Any]:
        """
        Make a unary (request-response) gRPC call.

        Args:
            service: Full service name (e.g., "package.ServiceName")
            method: Method name
            request: Request data as dict (will be converted to protobuf)
            request_type: Protobuf request message class (optional)
            response_type: Protobuf response message class (optional)
            options: Call options

        Returns:
            Response data as dict

        Note:
            If request_type/response_type are not provided, uses generic
            JSON-like dict passing which works with proto3 JSON mapping.
        """
        options = options or CallOptions()

        # Circuit breaker check
        if self._circuit_breaker:
            await self._circuit_breaker.before_call()

        try:
            result = await self._unary_call_with_retry(
                service, method, request, request_type, response_type, options
            )

            if self._circuit_breaker:
                self._circuit_breaker.record_success()

            return result

        except Exception as e:
            if self._circuit_breaker:
                self._circuit_breaker.record_failure()
            raise

    async def _unary_call_with_retry(
        self,
        service: str,
        method: str,
        request: dict[str, Any],
        request_type: Optional[Any],
        response_type: Optional[Any],
        options: CallOptions,
    ) -> dict[str, Any]:
        """Execute unary call with retry."""
        last_error: Optional[Exception] = None
        retries = self.config.max_retries if self.config.retry_enabled else 0

        for attempt in range(retries + 1):
            try:
                return await self._unary_call_single(
                    service, method, request, request_type, response_type, options
                )
            except GRPCError as e:
                # Check if error is retryable
                if e.code in ("UNAVAILABLE", "RESOURCE_EXHAUSTED", "ABORTED"):
                    last_error = e
                    if attempt < retries:
                        delay = self._retry_strategy.get_delay(attempt)
                        await asyncio.sleep(delay)
                else:
                    raise
            except Exception as e:
                last_error = e
                if attempt < retries:
                    delay = self._retry_strategy.get_delay(attempt)
                    await asyncio.sleep(delay)

        raise GRPCClientError(f"Call failed after {retries + 1} attempts: {last_error}")

    async def _unary_call_single(
        self,
        service: str,
        method: str,
        request: dict[str, Any],
        request_type: Optional[Any],
        response_type: Optional[Any],
        options: CallOptions,
    ) -> dict[str, Any]:
        """Execute a single unary call."""
        import grpc

        channel = self._get_channel()

        # Build method path
        method_path = f"/{service}/{method}"

        # Build metadata
        metadata = list(self.config.default_metadata.items())
        if options.metadata:
            metadata.extend(options.metadata.items())

        timeout = options.timeout or self.config.default_timeout

        # If we have protobuf types, use them
        if request_type and response_type:
            # Serialize request
            req_message = request_type(**request)

            # Make call
            call = channel.unary_unary(
                method_path,
                request_serializer=request_type.SerializeToString,
                response_deserializer=response_type.FromString,
            )

            try:
                response = await call(
                    req_message,
                    timeout=timeout,
                    metadata=metadata if metadata else None,
                    wait_for_ready=options.wait_for_ready,
                )
                # Convert to dict
                from google.protobuf.json_format import MessageToDict

                return MessageToDict(response)
            except grpc.aio.AioRpcError as e:
                raise GRPCError(
                    message=str(e.details()),
                    code=e.code().name,
                    details=str(e.debug_error_string()),
                )

        else:
            # Use generic bytes serialization with JSON
            import json

            def serialize(data: dict) -> bytes:
                return json.dumps(data).encode()

            def deserialize(data: bytes) -> dict:
                return json.loads(data.decode())

            call = channel.unary_unary(
                method_path,
                request_serializer=serialize,
                response_deserializer=deserialize,
            )

            try:
                response = await call(
                    request,
                    timeout=timeout,
                    metadata=metadata if metadata else None,
                    wait_for_ready=options.wait_for_ready,
                )
                return response
            except grpc.aio.AioRpcError as e:
                raise GRPCError(
                    message=str(e.details()),
                    code=e.code().name,
                    details=str(e.debug_error_string()),
                )

    async def server_streaming_call(
        self,
        service: str,
        method: str,
        request: dict[str, Any],
        request_type: Optional[Any] = None,
        response_type: Optional[Any] = None,
        options: Optional[CallOptions] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Make a server streaming gRPC call.

        Args:
            service: Full service name
            method: Method name
            request: Request data
            request_type: Protobuf request message class
            response_type: Protobuf response message class
            options: Call options

        Yields:
            Response messages as dicts
        """
        import grpc
        import json

        options = options or CallOptions()
        channel = self._get_channel()
        method_path = f"/{service}/{method}"

        metadata = list(self.config.default_metadata.items())
        if options.metadata:
            metadata.extend(options.metadata.items())

        timeout = options.timeout or self.config.default_timeout

        if request_type and response_type:
            req_message = request_type(**request)

            call = channel.unary_stream(
                method_path,
                request_serializer=request_type.SerializeToString,
                response_deserializer=response_type.FromString,
            )

            try:
                async for response in call(
                    req_message,
                    timeout=timeout,
                    metadata=metadata if metadata else None,
                ):
                    from google.protobuf.json_format import MessageToDict

                    yield MessageToDict(response)
            except grpc.aio.AioRpcError as e:
                raise GRPCError(
                    message=str(e.details()),
                    code=e.code().name,
                )
        else:
            def serialize(data: dict) -> bytes:
                return json.dumps(data).encode()

            def deserialize(data: bytes) -> dict:
                return json.loads(data.decode())

            call = channel.unary_stream(
                method_path,
                request_serializer=serialize,
                response_deserializer=deserialize,
            )

            try:
                async for response in call(
                    request,
                    timeout=timeout,
                    metadata=metadata if metadata else None,
                ):
                    yield response
            except grpc.aio.AioRpcError as e:
                raise GRPCError(
                    message=str(e.details()),
                    code=e.code().name,
                )

    async def health_check(self, service: str = "") -> bool:
        """
        Perform gRPC health check.

        Uses the standard gRPC health checking protocol.

        Args:
            service: Service name to check (empty for overall health)

        Returns:
            True if healthy
        """
        try:
            result = await self.unary_call(
                service="grpc.health.v1.Health",
                method="Check",
                request={"service": service},
            )
            return result.get("status") == "SERVING"
        except GRPCError:
            return False

    @property
    def channel_state(self) -> ChannelState:
        """Get current channel state."""
        if self._channel is None:
            return ChannelState.IDLE

        import grpc

        state = self._channel.get_state(try_to_connect=False)
        state_map = {
            grpc.ChannelConnectivity.IDLE: ChannelState.IDLE,
            grpc.ChannelConnectivity.CONNECTING: ChannelState.CONNECTING,
            grpc.ChannelConnectivity.READY: ChannelState.READY,
            grpc.ChannelConnectivity.TRANSIENT_FAILURE: ChannelState.TRANSIENT_FAILURE,
            grpc.ChannelConnectivity.SHUTDOWN: ChannelState.SHUTDOWN,
        }
        return state_map.get(state, ChannelState.IDLE)
