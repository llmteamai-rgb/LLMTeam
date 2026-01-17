"""
Server-Sent Events (SSE) Transport for Worktrail Events.

Provides one-way event streaming from server to client.

Usage:
    # Server-side (FastAPI example)
    from fastapi.responses import StreamingResponse

    transport = SSETransport()

    @app.get("/events")
    async def events():
        return StreamingResponse(
            transport.stream(event_source),
            media_type="text/event-stream",
        )

    # Client-side (JavaScript)
    const source = new EventSource('/events');
    source.onmessage = (event) => console.log(JSON.parse(event.data));
"""

import json
import asyncio
from enum import Enum
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional
from datetime import datetime


@dataclass
class SSEConfig:
    """SSE transport configuration."""
    retry_ms: int = 3000
    keep_alive_interval: float = 15.0
    event_type: str = "message"


def format_sse_event(
    data: Any,
    event: Optional[str] = None,
    id: Optional[str] = None,
    retry: Optional[int] = None,
) -> str:
    """
    Format data as SSE event.

    Args:
        data: Event data (will be JSON serialized if not string).
        event: Event type/name.
        id: Event ID for client reconnection.
        retry: Reconnection time in milliseconds.

    Returns:
        Formatted SSE event string.
    """
    lines = []

    if id is not None:
        lines.append(f"id: {id}")

    if event is not None:
        lines.append(f"event: {event}")

    if retry is not None:
        lines.append(f"retry: {retry}")

    # Serialize data
    if isinstance(data, str):
        data_str = data
    else:
        data_str = json.dumps(data, default=_json_serializer)

    # Handle multi-line data
    for line in data_str.split("\n"):
        lines.append(f"data: {line}")

    lines.append("")  # Empty line to end event
    lines.append("")

    return "\n".join(lines)


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for SSE events."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return str(obj)


class SSETransport:
    """
    Server-Sent Events transport for event streaming.

    Supports:
    - Event type filtering
    - Keep-alive messages
    - Event ID for reconnection
    - Retry configuration
    """

    def __init__(self, config: Optional[SSEConfig] = None):
        """
        Initialize SSE transport.

        Args:
            config: Transport configuration.
        """
        self._config = config or SSEConfig()
        self._event_counter = 0

    async def stream(
        self,
        events: AsyncIterator[Any],
        include_retry: bool = True,
    ) -> AsyncIterator[str]:
        """
        Stream events as SSE.

        Args:
            events: Async iterator of events to stream.
            include_retry: Include retry directive in first event.

        Yields:
            SSE-formatted event strings.
        """
        # Send initial retry configuration
        if include_retry:
            yield f"retry: {self._config.retry_ms}\n\n"

        # Start keep-alive task
        keep_alive_task = asyncio.create_task(self._keep_alive_generator())

        try:
            async for event in events:
                self._event_counter += 1

                # Serialize event
                event_data = self._serialize_event(event)

                # Determine event type
                event_type = self._get_event_type(event)

                yield format_sse_event(
                    data=event_data,
                    event=event_type,
                    id=str(self._event_counter),
                )

        finally:
            keep_alive_task.cancel()
            try:
                await keep_alive_task
            except asyncio.CancelledError:
                pass

    async def _keep_alive_generator(self) -> None:
        """Generate keep-alive comments."""
        while True:
            await asyncio.sleep(self._config.keep_alive_interval)
            # Keep-alive is handled by the stream consumer

    def format_event(
        self,
        event: Any,
        event_type: Optional[str] = None,
    ) -> str:
        """
        Format a single event as SSE.

        Args:
            event: Event to format.
            event_type: Override event type.

        Returns:
            SSE-formatted string.
        """
        self._event_counter += 1
        event_data = self._serialize_event(event)

        return format_sse_event(
            data=event_data,
            event=event_type or self._get_event_type(event),
            id=str(self._event_counter),
        )

    def keep_alive(self) -> str:
        """
        Generate a keep-alive comment.

        Returns:
            SSE comment for keep-alive.
        """
        return f": keep-alive {datetime.now().isoformat()}\n\n"

    def _serialize_event(self, event: Any) -> dict[str, Any]:
        """Serialize event to dictionary."""
        if isinstance(event, dict):
            return event

        if hasattr(event, "to_dict"):
            return event.to_dict()

        result: dict[str, Any] = {}
        for key, value in event.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, Enum):
                result[key] = value.value
            elif hasattr(value, "to_dict"):
                result[key] = value.to_dict()
            elif hasattr(value, "__dict__"):
                result[key] = self._serialize_event(value)
            else:
                result[key] = value

        return result

    def _get_event_type(self, event: Any) -> str:
        """Get event type from event object."""
        # Check for event_type attribute
        if hasattr(event, "event_type"):
            event_type = event.event_type
            if isinstance(event_type, Enum):
                return event_type.value
            return str(event_type)

        # Check for type attribute
        if hasattr(event, "type"):
            return str(event.type)

        # Use class name
        if hasattr(event, "__class__"):
            return event.__class__.__name__

        return self._config.event_type


class SSEClient:
    """
    SSE client for receiving events.

    Simple client for testing or non-browser use cases.

    Usage:
        async with SSEClient("http://localhost:8000/events") as client:
            async for event in client:
                print(event)
    """

    def __init__(self, url: str, headers: Optional[dict[str, str]] = None):
        """
        Initialize SSE client.

        Args:
            url: SSE endpoint URL.
            headers: Optional HTTP headers.
        """
        self.url = url
        self.headers = headers or {}
        self._session: Any = None
        self._response: Any = None

    async def connect(self) -> None:
        """Connect to SSE endpoint."""
        try:
            import aiohttp
        except ImportError:
            raise ImportError(
                "aiohttp package required for SSEClient. "
                "Install with: pip install aiohttp"
            )

        self._session = aiohttp.ClientSession()
        self._response = await self._session.get(
            self.url,
            headers={
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
                **self.headers,
            },
        )

    async def close(self) -> None:
        """Close the connection."""
        if self._response:
            self._response.close()
        if self._session:
            await self._session.close()

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        """
        Iterate over received events.

        Yields:
            Parsed event dictionaries with keys: id, event, data.
        """
        if not self._response:
            await self.connect()

        current_event: dict[str, Any] = {}

        async for line in self._response.content:
            line = line.decode("utf-8").rstrip("\n\r")

            if not line:
                # Empty line = event complete
                if current_event.get("data"):
                    try:
                        current_event["data"] = json.loads(current_event["data"])
                    except json.JSONDecodeError:
                        pass
                    yield current_event
                current_event = {}
                continue

            if line.startswith(":"):
                # Comment (keep-alive)
                continue

            if ":" in line:
                field, _, value = line.partition(":")
                value = value.lstrip(" ")

                if field == "data":
                    # Append to data (handle multi-line)
                    existing = current_event.get("data", "")
                    current_event["data"] = existing + value if existing else value
                else:
                    current_event[field] = value

    async def __aenter__(self) -> "SSEClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        """Iterate over events."""
        return self.events()
