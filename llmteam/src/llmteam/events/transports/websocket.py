"""
WebSocket Transport for Worktrail Events.

Provides bidirectional event streaming over WebSocket.

Usage:
    transport = WebSocketTransport(url="ws://localhost:8000/events")
    await transport.connect()

    # Send events
    await transport.send(event)

    # Receive events
    async for event in transport.receive():
        print(event)

    await transport.close()
"""

import asyncio
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional, Callable, Awaitable
from datetime import datetime


class ConnectionState(Enum):
    """WebSocket connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass
class WebSocketConfig:
    """WebSocket transport configuration."""
    url: str
    reconnect: bool = True
    reconnect_interval: float = 1.0
    max_reconnect_attempts: int = 10
    ping_interval: float = 30.0
    ping_timeout: float = 10.0
    message_queue_size: int = 1000


class WebSocketTransport:
    """
    WebSocket transport for event streaming.

    Supports:
    - Automatic reconnection
    - Message queueing during disconnection
    - Ping/pong keepalive
    - Event serialization/deserialization
    """

    def __init__(
        self,
        url: str,
        config: Optional[WebSocketConfig] = None,
        on_connect: Optional[Callable[[], Awaitable[None]]] = None,
        on_disconnect: Optional[Callable[[], Awaitable[None]]] = None,
        on_error: Optional[Callable[[Exception], Awaitable[None]]] = None,
    ):
        """
        Initialize WebSocket transport.

        Args:
            url: WebSocket URL to connect to.
            config: Transport configuration.
            on_connect: Callback when connected.
            on_disconnect: Callback when disconnected.
            on_error: Callback on error.
        """
        self._config = config or WebSocketConfig(url=url)
        if config is None:
            self._config.url = url

        self._state = ConnectionState.DISCONNECTED
        self._websocket: Any = None
        self._reconnect_attempts = 0

        # Callbacks
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect
        self._on_error = on_error

        # Message queue for offline buffering
        self._send_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=self._config.message_queue_size
        )

        # Tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._send_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None

    @property
    def state(self) -> ConnectionState:
        """Current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state == ConnectionState.CONNECTED

    async def connect(self) -> None:
        """
        Connect to WebSocket server.

        Raises:
            ImportError: If websockets package not installed.
            ConnectionError: If connection fails.
        """
        if self._state in (ConnectionState.CONNECTED, ConnectionState.CONNECTING):
            return

        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets package not installed. "
                "Install with: pip install llmteam-ai[websockets]"
            )

        self._state = ConnectionState.CONNECTING

        try:
            self._websocket = await websockets.connect(
                self._config.url,
                ping_interval=self._config.ping_interval,
                ping_timeout=self._config.ping_timeout,
            )
            self._state = ConnectionState.CONNECTED
            self._reconnect_attempts = 0

            # Start background tasks
            self._send_task = asyncio.create_task(self._send_loop())

            if self._on_connect:
                await self._on_connect()

        except Exception as e:
            self._state = ConnectionState.DISCONNECTED
            if self._on_error:
                await self._on_error(e)
            raise ConnectionError(f"Failed to connect: {e}") from e

    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self._state == ConnectionState.CLOSED:
            return

        self._state = ConnectionState.CLOSING

        # Cancel background tasks
        for task in [self._send_task, self._receive_task, self._ping_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close websocket
        if self._websocket:
            await self._websocket.close()
            self._websocket = None

        self._state = ConnectionState.CLOSED

        if self._on_disconnect:
            await self._on_disconnect()

    async def send(self, event: Any) -> None:
        """
        Send an event.

        Events are queued if disconnected and sent when reconnected.

        Args:
            event: Event to send (WorktrailEvent or dict).
        """
        # Serialize event
        if hasattr(event, "to_dict"):
            data = event.to_dict()
        elif hasattr(event, "__dict__"):
            data = self._serialize_event(event)
        else:
            data = event

        # Add to queue
        try:
            self._send_queue.put_nowait(data)
        except asyncio.QueueFull:
            # Drop oldest message
            try:
                self._send_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._send_queue.put_nowait(data)

    async def receive(self) -> AsyncIterator[dict[str, Any]]:
        """
        Receive events from WebSocket.

        Yields:
            Deserialized event dictionaries.
        """
        if not self._websocket:
            raise ConnectionError("Not connected")

        try:
            async for message in self._websocket:
                try:
                    data = json.loads(message)
                    yield data
                except json.JSONDecodeError:
                    yield {"raw": message}

        except Exception as e:
            if self._config.reconnect:
                await self._reconnect()
            else:
                raise

    async def _send_loop(self) -> None:
        """Background task to send queued messages."""
        while self._state == ConnectionState.CONNECTED:
            try:
                data = await asyncio.wait_for(
                    self._send_queue.get(),
                    timeout=1.0,
                )

                if self._websocket:
                    await self._websocket.send(json.dumps(data))

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if self._on_error:
                    await self._on_error(e)

    async def _reconnect(self) -> None:
        """Attempt to reconnect."""
        if not self._config.reconnect:
            return

        if self._reconnect_attempts >= self._config.max_reconnect_attempts:
            self._state = ConnectionState.DISCONNECTED
            return

        self._state = ConnectionState.RECONNECTING
        self._reconnect_attempts += 1

        await asyncio.sleep(
            self._config.reconnect_interval * self._reconnect_attempts
        )

        try:
            await self.connect()
        except Exception:
            await self._reconnect()

    def _serialize_event(self, event: Any) -> dict[str, Any]:
        """Serialize an event object to dict."""
        data: dict[str, Any] = {}

        for key, value in event.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, Enum):
                data[key] = value.value
            elif hasattr(value, "__dict__"):
                data[key] = self._serialize_event(value)
            else:
                data[key] = value

        return data

    async def __aenter__(self) -> "WebSocketTransport":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
