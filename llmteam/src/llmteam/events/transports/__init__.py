"""
Event Transports module.

Provides transports for streaming Worktrail events:
- WebSocket: Bidirectional real-time streaming
- SSE: Server-Sent Events for unidirectional streaming
- Redis: Pub/Sub messaging (requires redis extra)
- Kafka: Enterprise event streaming (requires kafka extra)

Usage:
    from llmteam.events.transports import WebSocketTransport, SSETransport

    # WebSocket transport
    transport = WebSocketTransport(url="ws://localhost:8000/events")
    await transport.connect()
    await transport.send(event)

    # SSE transport (server-side)
    transport = SSETransport()
    async for chunk in transport.stream(events):
        yield chunk

    # Redis transport (requires: pip install llmteam-ai[redis])
    from llmteam.events.transports import RedisTransport
    transport = RedisTransport(url="redis://localhost:6379")
    await transport.publish("events", event)

    # Kafka transport (requires: pip install aiokafka)
    from llmteam.events.transports import KafkaTransport
    transport = KafkaTransport(bootstrap_servers="localhost:9092")
    await transport.produce("events", event)
"""

from llmteam.events.transports.websocket import (
    WebSocketTransport,
    WebSocketConfig,
    ConnectionState,
)

from llmteam.events.transports.sse import (
    SSETransport,
    SSEConfig,
    format_sse_event,
)

from llmteam.events.transports.redis import (
    RedisTransport,
    RedisConfig,
    RedisConnectionState,
)

from llmteam.events.transports.kafka import (
    KafkaTransport,
    KafkaConfig,
    KafkaConnectionState,
)

__all__ = [
    # WebSocket
    "WebSocketTransport",
    "WebSocketConfig",
    "ConnectionState",
    # SSE
    "SSETransport",
    "SSEConfig",
    "format_sse_event",
    # Redis
    "RedisTransport",
    "RedisConfig",
    "RedisConnectionState",
    # Kafka
    "KafkaTransport",
    "KafkaConfig",
    "KafkaConnectionState",
]
