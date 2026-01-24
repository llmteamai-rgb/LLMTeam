"""
Worktrail Events module for LLMTeam v2.0.0.

Standardized events for displaying progress in canvas UI.
"""

from llmteam.events.models import (
    EventType,
    EventSeverity,
    ErrorInfo,
    WorktrailEvent,
)
from llmteam.events.emitter import EventEmitter
from llmteam.events.store import (
    EventStore,
    MemoryEventStore,
)
from llmteam.events.stream import EventStream
from llmteam.events.streaming import StreamEventType, StreamEvent

__all__ = [
    # Models
    "EventType",
    "EventSeverity",
    "ErrorInfo",
    "WorktrailEvent",
    # Emitter
    "EventEmitter",
    # Store
    "EventStore",
    "MemoryEventStore",
    # Stream
    "EventStream",
    # RFC-011: Streaming
    "StreamEventType",
    "StreamEvent",
]
