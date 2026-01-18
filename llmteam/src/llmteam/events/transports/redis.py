"""
Redis Transport for Worktrail Events.

Provides Pub/Sub event streaming using Redis.

Usage:
    from llmteam.events.transports import RedisTransport

    transport = RedisTransport(url="redis://localhost:6379")
    await transport.connect()

    # Publish events
    await transport.publish("workflow-events", event)

    # Subscribe to events
    async for event in transport.subscribe("workflow-events"):
        print(event)

    await transport.close()

Install:
    pip install llmteam-ai[redis]
"""

import asyncio
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional, Callable, Awaitable
from datetime import datetime


class RedisConnectionState(Enum):
    """Redis connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"


@dataclass
class RedisConfig:
    """Redis transport configuration."""
    url: str = "redis://localhost:6379"
    db: int = 0
    password: Optional[str] = None
    channel_prefix: str = "llmteam:"
    reconnect: bool = True
    reconnect_interval: float = 1.0
    max_reconnect_attempts: int = 10
    message_queue_size: int = 1000
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0


class RedisTransport:
    """
    Redis transport for event streaming.

    Supports:
    - Pub/Sub messaging
    - Automatic reconnection
    - Message queueing during disconnection
    - Event serialization/deserialization
    - Channel patterns (glob-style)
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        config: Optional[RedisConfig] = None,
        on_connect: Optional[Callable[[], Awaitable[None]]] = None,
        on_disconnect: Optional[Callable[[], Awaitable[None]]] = None,
        on_error: Optional[Callable[[Exception], Awaitable[None]]] = None,
    ):
        """
        Initialize Redis transport.

        Args:
            url: Redis URL (redis://host:port/db)
            config: Transport configuration
            on_connect: Callback when connected
            on_disconnect: Callback when disconnected
            on_error: Callback on error
        """
        self._config = config or RedisConfig(url=url)
        if config is None:
            self._config.url = url

        self._state = RedisConnectionState.DISCONNECTED
        self._redis: Any = None
        self._pubsub: Any = None
        self._reconnect_attempts = 0

        # Callbacks
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect
        self._on_error = on_error

        # Message queue for offline buffering
        self._send_queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue(
            maxsize=self._config.message_queue_size
        )

        # Subscription handlers
        self._subscriptions: dict[str, list[Callable[[dict], Awaitable[None]]]] = {}

        # Tasks
        self._publish_task: Optional[asyncio.Task] = None
        self._subscribe_task: Optional[asyncio.Task] = None

    @property
    def state(self) -> RedisConnectionState:
        """Current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state == RedisConnectionState.CONNECTED

    async def connect(self) -> None:
        """
        Connect to Redis server.

        Raises:
            ImportError: If redis package not installed
            ConnectionError: If connection fails
        """
        if self._state in (RedisConnectionState.CONNECTED, RedisConnectionState.CONNECTING):
            return

        try:
            import redis.asyncio as redis
        except ImportError:
            raise ImportError(
                "redis package not installed. "
                "Install with: pip install llmteam-ai[redis]"
            )

        self._state = RedisConnectionState.CONNECTING

        try:
            self._redis = redis.from_url(
                self._config.url,
                db=self._config.db,
                password=self._config.password,
                socket_timeout=self._config.socket_timeout,
                socket_connect_timeout=self._config.socket_connect_timeout,
                decode_responses=True,
            )

            # Test connection
            await self._redis.ping()

            self._state = RedisConnectionState.CONNECTED
            self._reconnect_attempts = 0

            # Start background publish task
            self._publish_task = asyncio.create_task(self._publish_loop())

            if self._on_connect:
                await self._on_connect()

        except Exception as e:
            self._state = RedisConnectionState.DISCONNECTED
            if self._on_error:
                await self._on_error(e)
            raise ConnectionError(f"Failed to connect to Redis: {e}") from e

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._state == RedisConnectionState.CLOSED:
            return

        self._state = RedisConnectionState.CLOSED

        # Cancel background tasks
        for task in [self._publish_task, self._subscribe_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close pubsub
        if self._pubsub:
            await self._pubsub.close()
            self._pubsub = None

        # Close redis connection
        if self._redis:
            await self._redis.close()
            self._redis = None

        if self._on_disconnect:
            await self._on_disconnect()

    async def publish(self, channel: str, event: Any) -> int:
        """
        Publish an event to a channel.

        Events are queued if disconnected and sent when reconnected.

        Args:
            channel: Channel name (prefix will be added)
            event: Event to publish (WorktrailEvent or dict)

        Returns:
            Number of subscribers that received the message
        """
        # Serialize event
        if hasattr(event, "to_dict"):
            data = event.to_dict()
        elif hasattr(event, "__dict__"):
            data = self._serialize_event(event)
        else:
            data = event

        full_channel = f"{self._config.channel_prefix}{channel}"

        if self._state == RedisConnectionState.CONNECTED and self._redis:
            try:
                return await self._redis.publish(full_channel, json.dumps(data))
            except Exception as e:
                if self._on_error:
                    await self._on_error(e)
                # Queue for later
                await self._queue_message(channel, data)
                return 0
        else:
            # Queue for later
            await self._queue_message(channel, data)
            return 0

    async def subscribe(
        self,
        channel: str,
        pattern: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Subscribe to a channel and yield events.

        Args:
            channel: Channel name or pattern
            pattern: Whether to use pattern matching

        Yields:
            Deserialized event dictionaries
        """
        if not self._redis:
            raise ConnectionError("Not connected")

        full_channel = f"{self._config.channel_prefix}{channel}"

        # Create pubsub if needed
        if not self._pubsub:
            self._pubsub = self._redis.pubsub()

        # Subscribe
        if pattern:
            await self._pubsub.psubscribe(full_channel)
        else:
            await self._pubsub.subscribe(full_channel)

        try:
            async for message in self._pubsub.listen():
                if message["type"] in ("message", "pmessage"):
                    try:
                        data = json.loads(message["data"])
                        data["_channel"] = message.get("channel", channel)
                        yield data
                    except json.JSONDecodeError:
                        yield {"raw": message["data"], "_channel": channel}

        except asyncio.CancelledError:
            raise
        except Exception as e:
            if self._config.reconnect:
                await self._reconnect()
            else:
                raise

    async def subscribe_handler(
        self,
        channel: str,
        handler: Callable[[dict], Awaitable[None]],
        pattern: bool = False,
    ) -> None:
        """
        Subscribe with a handler function.

        Args:
            channel: Channel name or pattern
            handler: Async function to call for each event
            pattern: Whether to use pattern matching
        """
        if channel not in self._subscriptions:
            self._subscriptions[channel] = []
        self._subscriptions[channel].append(handler)

        # Start subscription task
        asyncio.create_task(self._handle_subscription(channel, pattern))

    async def unsubscribe(self, channel: str) -> None:
        """Unsubscribe from a channel."""
        full_channel = f"{self._config.channel_prefix}{channel}"

        if self._pubsub:
            await self._pubsub.unsubscribe(full_channel)

        if channel in self._subscriptions:
            del self._subscriptions[channel]

    async def _handle_subscription(self, channel: str, pattern: bool) -> None:
        """Handle subscription with registered handlers."""
        try:
            async for event in self.subscribe(channel, pattern):
                handlers = self._subscriptions.get(channel, [])
                for handler in handlers:
                    try:
                        await handler(event)
                    except Exception as e:
                        if self._on_error:
                            await self._on_error(e)
        except asyncio.CancelledError:
            pass

    async def _queue_message(self, channel: str, data: dict) -> None:
        """Queue a message for later delivery."""
        try:
            self._send_queue.put_nowait((channel, data))
        except asyncio.QueueFull:
            # Drop oldest message
            try:
                self._send_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._send_queue.put_nowait((channel, data))

    async def _publish_loop(self) -> None:
        """Background task to publish queued messages."""
        while self._state == RedisConnectionState.CONNECTED:
            try:
                channel, data = await asyncio.wait_for(
                    self._send_queue.get(),
                    timeout=1.0,
                )

                if self._redis:
                    full_channel = f"{self._config.channel_prefix}{channel}"
                    await self._redis.publish(full_channel, json.dumps(data))

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._on_error:
                    await self._on_error(e)

    async def _reconnect(self) -> None:
        """Attempt to reconnect."""
        if not self._config.reconnect:
            return

        if self._reconnect_attempts >= self._config.max_reconnect_attempts:
            self._state = RedisConnectionState.DISCONNECTED
            return

        self._state = RedisConnectionState.RECONNECTING
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

    async def __aenter__(self) -> "RedisTransport":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
