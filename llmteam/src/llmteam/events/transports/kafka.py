"""
Kafka Event Transport.

Enterprise-grade event streaming with guaranteed delivery using Apache Kafka.
"""

from typing import Any, AsyncIterator, Optional
import json
import asyncio

from llmteam.events.models import WorktrailEvent
from llmteam.observability import get_logger

from dataclasses import dataclass
from enum import Enum

logger = get_logger(__name__)


class KafkaConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class KafkaConfig:
    bootstrap_servers: str = "localhost:9092"
    topic: str = "llmteam-events"
    group_id: str = "llmteam-consumers"
    client_id: Optional[str] = None
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_plain_username: Optional[str] = None
    sasl_plain_password: Optional[str] = None



class KafkaTransport:
    """
    Kafka transport for enterprise event streaming.
    
    Features:
    - Guaranteed delivery
    - Event ordering
    - Replay capability
    - Horizontal scaling
    
    Usage:
        transport = KafkaTransport(
            bootstrap_servers="kafka:9092",
            topic="llmteam-events",
        )
        await transport.connect()
        
        # Produce events
        await transport.publish(event)
        
        # Consume events
        async for event in transport.consume():
            process(event)
    """
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "llmteam-events",
        group_id: str = "llmteam-consumers",
        client_id: Optional[str] = None,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.client_id = client_id or f"llmteam-{id(self)}"
        self._producer = None
        self._consumer = None
    
    async def connect(self) -> None:
        """Connect producer and consumer."""
        try:
            from aiokafka import AIOKafkaProducer
        except ImportError:
            raise ImportError(
                "aiokafka package required. Install with: pip install aiokafka"
            )
        
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            client_id=self.client_id,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        await self._producer.start()
        
        logger.info(f"Kafka producer connected: {self.bootstrap_servers}")
    
    async def connect_consumer(self) -> None:
        """Connect consumer (separate from producer for flexibility)."""
        try:
            from aiokafka import AIOKafkaConsumer
        except ImportError:
             raise ImportError(
                "aiokafka package required. Install with: pip install aiokafka"
            )
        
        self._consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            client_id=f"{self.client_id}-consumer",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest",
        )
        await self._consumer.start()
        
        logger.info(f"Kafka consumer connected: {self.group_id}")
    
    async def disconnect(self) -> None:
        """Disconnect producer and consumer."""
        if self._producer:
            await self._producer.stop()
        if self._consumer:
            await self._consumer.stop()
        
        logger.info("Kafka transport disconnected")
    
    async def publish(
        self,
        event: WorktrailEvent,
        key: Optional[str] = None,
    ) -> None:
        """
        Publish event to Kafka.
        
        Args:
            event: Event to publish
            key: Partition key (default: run_id for ordering)
        """
        if not self._producer:
            raise RuntimeError("Producer not connected")
        
        partition_key = key or event.run_id
        
        await self._producer.send_and_wait(
            self.topic,
            value=event.to_dict(),
            key=partition_key.encode("utf-8") if partition_key else None,
        )
    
    async def consume(
        self,
        timeout_ms: int = 1000,
    ) -> AsyncIterator[WorktrailEvent]:
        """
        Consume events from Kafka.
        
        Yields:
            WorktrailEvent objects
        """
        if not self._consumer:
            await self.connect_consumer()
        
        try:
            async for message in self._consumer:
                try:
                    yield WorktrailEvent.from_dict(message.value)
                except (KeyError, TypeError) as e:
                    logger.warning(f"Invalid Kafka message: {e}")
        except Exception as e:
            logger.error(f"Kafka consumer error: {e}")
            raise
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, *args):
        await self.disconnect()
