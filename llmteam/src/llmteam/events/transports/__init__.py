"""
Event Transports module.

Provides WebSocket and SSE transports for streaming Worktrail events.

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

__all__ = [
    # WebSocket
    "WebSocketTransport",
    "WebSocketConfig",
    "ConnectionState",
    # SSE
    "SSETransport",
    "SSEConfig",
    "format_sse_event",
]
