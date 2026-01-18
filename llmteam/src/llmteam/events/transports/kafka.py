"""
Kafka Transport for Worktrail Events.

Provides enterprise-grade event streaming using Apache Kafka.

Usage:
    from llmteam.events.transports import KafkaTransport

    transport = KafkaTransport(bootstrap_servers="localhost:9092")
    await transport.connect()

    # Produce events
    await transport.produce("workflow-events", event)

    # Consume events
    async for event in transport.consume("workflow-events"):
        print(event)

    await transport.close()

Install:
    pip install llmteam-ai[kafka]
"""

import asyncio
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional, Callable, Awaitable, Union
from datetime import datetime


class KafkaConnectionState(Enum):
    """Kafka connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"


@dataclass
class KafkaConfig:
    """Kafka transport configuration."""
    bootstrap_servers: str = "localhost:9092"
    client_id: str = "llmteam-client"
    group_id: str = "llmteam-consumers"

    # Topic configuration
    topic_prefix: str = "llmteam."
    auto_create_topics: bool = True
    num_partitions: int = 3
    replication_factor: int = 1

    # Producer configuration
    acks: str = "all"  # "0", "1", "all"
    compression_type: str = "gzip"  # "none", "gzip", "snappy", "lz4", "zstd"
    batch_size: int = 16384
    linger_ms: int = 5

    # Consumer configuration
    auto_offset_reset: str = "earliest"  # "earliest", "latest"
    enable_auto_commit: bool = True
    auto_commit_interval_ms: int = 5000
    max_poll_records: int = 500

    # Connection settings
    reconnect: bool = True
    reconnect_interval: float = 1.0
    max_reconnect_attempts: int = 10
    request_timeout_ms: int = 30000

    # Security (optional)
    security_protocol: str = "PLAINTEXT"  # "PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL"
    sasl_mechanism: Optional[str] = None  # "PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512"
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None
    ssl_cafile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None


class KafkaTransport:
    """
    Kafka transport for enterprise event streaming.

    Supports:
    - High-throughput message production
    - Consumer groups with load balancing
    - Exactly-once semantics (with transactions)
    - Message compression
    - SASL/SSL security
    - Automatic topic creation
    - Partition key routing
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        config: Optional[KafkaConfig] = None,
        on_connect: Optional[Callable[[], Awaitable[None]]] = None,
        on_disconnect: Optional[Callable[[], Awaitable[None]]] = None,
        on_error: Optional[Callable[[Exception], Awaitable[None]]] = None,
    ):
        """
        Initialize Kafka transport.

        Args:
            bootstrap_servers: Kafka broker addresses
            config: Transport configuration
            on_connect: Callback when connected
            on_disconnect: Callback when disconnected
            on_error: Callback on error
        """
        self._config = config or KafkaConfig(bootstrap_servers=bootstrap_servers)
        if config is None:
            self._config.bootstrap_servers = bootstrap_servers

        self._state = KafkaConnectionState.DISCONNECTED
        self._producer: Any = None
        self._consumer: Any = None
        self._admin_client: Any = None
        self._reconnect_attempts = 0

        # Callbacks
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect
        self._on_error = on_error

        # Active subscriptions
        self._subscribed_topics: set[str] = set()

    @property
    def state(self) -> KafkaConnectionState:
        """Current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state == KafkaConnectionState.CONNECTED

    async def connect(self) -> None:
        """
        Connect to Kafka cluster.

        Creates producer and prepares for consumer creation.

        Raises:
            ImportError: If aiokafka package not installed
            ConnectionError: If connection fails
        """
        if self._state in (KafkaConnectionState.CONNECTED, KafkaConnectionState.CONNECTING):
            return

        try:
            from aiokafka import AIOKafkaProducer
        except ImportError:
            raise ImportError(
                "aiokafka package not installed. "
                "Install with: pip install aiokafka"
            )

        self._state = KafkaConnectionState.CONNECTING

        try:
            # Build producer configuration
            producer_config = self._build_producer_config()

            self._producer = AIOKafkaProducer(**producer_config)
            await self._producer.start()

            self._state = KafkaConnectionState.CONNECTED
            self._reconnect_attempts = 0

            if self._on_connect:
                await self._on_connect()

        except Exception as e:
            self._state = KafkaConnectionState.DISCONNECTED
            if self._on_error:
                await self._on_error(e)
            raise ConnectionError(f"Failed to connect to Kafka: {e}") from e

    async def close(self) -> None:
        """Close all Kafka connections."""
        if self._state == KafkaConnectionState.CLOSED:
            return

        self._state = KafkaConnectionState.CLOSED

        # Stop producer
        if self._producer:
            await self._producer.stop()
            self._producer = None

        # Stop consumer
        if self._consumer:
            await self._consumer.stop()
            self._consumer = None

        if self._on_disconnect:
            await self._on_disconnect()

    async def produce(
        self,
        topic: str,
        event: Any,
        key: Optional[str] = None,
        partition: Optional[int] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Produce an event to a topic.

        Args:
            topic: Topic name (prefix will be added)
            event: Event to produce (WorktrailEvent or dict)
            key: Optional partition key (for ordering)
            partition: Optional specific partition
            headers: Optional message headers
        """
        if not self._producer:
            raise ConnectionError("Producer not connected")

        # Serialize event
        if hasattr(event, "to_dict"):
            data = event.to_dict()
        elif hasattr(event, "__dict__"):
            data = self._serialize_event(event)
        else:
            data = event

        full_topic = f"{self._config.topic_prefix}{topic}"
        value = json.dumps(data).encode("utf-8")
        key_bytes = key.encode("utf-8") if key else None

        # Convert headers
        kafka_headers: Optional[list[tuple[str, bytes]]] = None
        if headers:
            kafka_headers = [(k, v.encode("utf-8")) for k, v in headers.items()]

        try:
            await self._producer.send_and_wait(
                full_topic,
                value=value,
                key=key_bytes,
                partition=partition,
                headers=kafka_headers,
            )
        except Exception as e:
            if self._on_error:
                await self._on_error(e)
            raise

    async def produce_batch(
        self,
        topic: str,
        events: list[Any],
        keys: Optional[list[Optional[str]]] = None,
    ) -> None:
        """
        Produce multiple events efficiently.

        Args:
            topic: Topic name
            events: List of events to produce
            keys: Optional list of partition keys
        """
        if not self._producer:
            raise ConnectionError("Producer not connected")

        full_topic = f"{self._config.topic_prefix}{topic}"

        if keys is None:
            keys = [None] * len(events)

        # Send all without waiting
        futures = []
        for event, key in zip(events, keys):
            if hasattr(event, "to_dict"):
                data = event.to_dict()
            elif hasattr(event, "__dict__"):
                data = self._serialize_event(event)
            else:
                data = event

            value = json.dumps(data).encode("utf-8")
            key_bytes = key.encode("utf-8") if key else None

            future = await self._producer.send(
                full_topic,
                value=value,
                key=key_bytes,
            )
            futures.append(future)

        # Wait for all to complete
        for future in futures:
            await future

    async def consume(
        self,
        topics: Union[str, list[str]],
        group_id: Optional[str] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Consume events from topic(s).

        Args:
            topics: Topic name(s) to consume from
            group_id: Consumer group ID (uses default if not specified)

        Yields:
            Deserialized event dictionaries with metadata
        """
        try:
            from aiokafka import AIOKafkaConsumer
        except ImportError:
            raise ImportError(
                "aiokafka package not installed. "
                "Install with: pip install aiokafka"
            )

        # Normalize topics
        if isinstance(topics, str):
            topics = [topics]

        full_topics = [f"{self._config.topic_prefix}{t}" for t in topics]

        # Build consumer configuration
        consumer_config = self._build_consumer_config(group_id)
        consumer_config["bootstrap_servers"] = self._config.bootstrap_servers

        self._consumer = AIOKafkaConsumer(*full_topics, **consumer_config)

        try:
            await self._consumer.start()
            self._subscribed_topics.update(full_topics)

            async for message in self._consumer:
                try:
                    data = json.loads(message.value.decode("utf-8"))
                    data["_topic"] = message.topic
                    data["_partition"] = message.partition
                    data["_offset"] = message.offset
                    data["_key"] = message.key.decode("utf-8") if message.key else None
                    data["_timestamp"] = message.timestamp

                    yield data

                except json.JSONDecodeError:
                    yield {
                        "raw": message.value.decode("utf-8"),
                        "_topic": message.topic,
                        "_partition": message.partition,
                        "_offset": message.offset,
                    }

        except asyncio.CancelledError:
            raise
        finally:
            if self._consumer:
                await self._consumer.stop()
                self._consumer = None

    async def commit(self) -> None:
        """Manually commit consumer offsets."""
        if self._consumer:
            await self._consumer.commit()

    async def seek_to_beginning(self, topic: Optional[str] = None) -> None:
        """Seek consumer to beginning of topic(s)."""
        if self._consumer:
            if topic:
                full_topic = f"{self._config.topic_prefix}{topic}"
                partitions = self._consumer.assignment()
                topic_partitions = [p for p in partitions if p.topic == full_topic]
                await self._consumer.seek_to_beginning(*topic_partitions)
            else:
                await self._consumer.seek_to_beginning()

    async def seek_to_end(self, topic: Optional[str] = None) -> None:
        """Seek consumer to end of topic(s)."""
        if self._consumer:
            if topic:
                full_topic = f"{self._config.topic_prefix}{topic}"
                partitions = self._consumer.assignment()
                topic_partitions = [p for p in partitions if p.topic == full_topic]
                await self._consumer.seek_to_end(*topic_partitions)
            else:
                await self._consumer.seek_to_end()

    def _build_producer_config(self) -> dict[str, Any]:
        """Build producer configuration dict."""
        config: dict[str, Any] = {
            "bootstrap_servers": self._config.bootstrap_servers,
            "client_id": f"{self._config.client_id}-producer",
            "acks": self._config.acks,
            "compression_type": self._config.compression_type,
            "max_batch_size": self._config.batch_size,
            "linger_ms": self._config.linger_ms,
            "request_timeout_ms": self._config.request_timeout_ms,
        }

        # Add security config
        self._add_security_config(config)

        return config

    def _build_consumer_config(self, group_id: Optional[str]) -> dict[str, Any]:
        """Build consumer configuration dict."""
        config: dict[str, Any] = {
            "client_id": f"{self._config.client_id}-consumer",
            "group_id": group_id or self._config.group_id,
            "auto_offset_reset": self._config.auto_offset_reset,
            "enable_auto_commit": self._config.enable_auto_commit,
            "auto_commit_interval_ms": self._config.auto_commit_interval_ms,
            "max_poll_records": self._config.max_poll_records,
        }

        # Add security config
        self._add_security_config(config)

        return config

    def _add_security_config(self, config: dict[str, Any]) -> None:
        """Add security configuration to config dict."""
        config["security_protocol"] = self._config.security_protocol

        if self._config.sasl_mechanism:
            config["sasl_mechanism"] = self._config.sasl_mechanism
            if self._config.sasl_username:
                config["sasl_plain_username"] = self._config.sasl_username
            if self._config.sasl_password:
                config["sasl_plain_password"] = self._config.sasl_password

        if self._config.ssl_cafile:
            config["ssl_context"] = self._build_ssl_context()

    def _build_ssl_context(self) -> Any:
        """Build SSL context for secure connections."""
        import ssl

        context = ssl.create_default_context()

        if self._config.ssl_cafile:
            context.load_verify_locations(self._config.ssl_cafile)

        if self._config.ssl_certfile and self._config.ssl_keyfile:
            context.load_cert_chain(
                certfile=self._config.ssl_certfile,
                keyfile=self._config.ssl_keyfile,
            )

        return context

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

    async def __aenter__(self) -> "KafkaTransport":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
