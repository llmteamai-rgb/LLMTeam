"""
Redis Event Transport.

Enables horizontal scaling by distributing events
across multiple llmteam instances using Redis Pub/Sub.
"""

from typing import Any, AsyncIterator, Callable, Optional, Union
import json
import asyncio

from llmteam.events.models import WorktrailEvent
from llmteam.observability import get_logger

from dataclasses import dataclass
from enum import Enum

logger = get_logger(__name__)


class RedisConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class RedisConfig:
    url: str = "redis://localhost:6379"
    channel_prefix: str = "llmteam:events:"
    max_retries: int = 3
    socket_timeout: float = 5.0
    ssl: bool = False



class RedisTransport:
    """
    Redis Pub/Sub transport for events.
    
    Enables horizontal scaling by distributing events
    across multiple llmteam instances.
    
    Usage:
        transport = RedisTransport(redis_url="redis://localhost:6379")
        await transport.connect()
        
        # Publish events
        await transport.publish(event)
        
        # Subscribe to events
        async for event in transport.subscribe("run_*"):
            print(event)
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        channel_prefix: str = "llmteam:events:",
        max_retries: int = 3,
    ):
        self.redis_url = redis_url
        self.channel_prefix = channel_prefix
        self.max_retries = max_retries
        self._redis = None
        self._pubsub = None
    
    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            from redis import asyncio as aioredis
        except ImportError:
            raise ImportError(
                "redis package required. Install with: pip install redis"
            )
        
        self._redis = await aioredis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        logger.info(f"Connected to Redis: {self.redis_url}")
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._pubsub:
            await self._pubsub.close()
        if self._redis:
            await self._redis.close()
        logger.info("Disconnected from Redis")
    
    async def publish(
        self,
        event: WorktrailEvent,
        channel: Optional[str] = None,
    ) -> int:
        """
        Publish event to Redis channel.
        
        Args:
            event: Event to publish
            channel: Channel name (default: derived from event)
            
        Returns:
            Number of subscribers that received the message
        """
        if not self._redis:
            raise RuntimeError("Not connected to Redis")
        
        if channel is None:
            channel = f"{self.channel_prefix}{event.run_id}"
        
        message = json.dumps(event.to_dict())
        return await self._redis.publish(channel, message)
    
    async def subscribe(
        self,
        pattern: str,
    ) -> AsyncIterator[WorktrailEvent]:
        """
        Subscribe to events matching pattern.
        
        Args:
            pattern: Channel pattern (supports * wildcards)
            
        Yields:
            WorktrailEvent objects
        """
        if not self._redis:
            raise RuntimeError("Not connected to Redis")
        
        self._pubsub = self._redis.pubsub()
        full_pattern = f"{self.channel_prefix}{pattern}"
        
        await self._pubsub.psubscribe(full_pattern)
        logger.info(f"Subscribed to: {full_pattern}")
        
        try:
            async for message in self._pubsub.listen():
                if message["type"] == "pmessage":
                    try:
                        data = json.loads(message["data"])
                        yield WorktrailEvent.from_dict(data)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Invalid event message: {e}")
        finally:
            await self._pubsub.punsubscribe(full_pattern)
    
    async def subscribe_run(
        self,
        run_id: str,
        callback: Callable[[WorktrailEvent], Any],
    ) -> None:
        """
        Subscribe to events for specific run.
        
        Args:
            run_id: Run ID to subscribe to
            callback: Callback for each event
        """
        async for event in self.subscribe(run_id):
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, *args):
        await self.disconnect()
