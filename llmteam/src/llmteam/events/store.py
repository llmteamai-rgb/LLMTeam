"""
Event Store for persisting Worktrail events.
"""

from collections import deque
from typing import Deque, List, Optional, Protocol
from datetime import datetime

from llmteam.events.models import WorktrailEvent, EventType


class EventStore(Protocol):
    """Protocol for event storage."""

    async def append(self, event: WorktrailEvent) -> None:
        """Append event to store."""
        ...

    async def get_by_run(self, run_id: str) -> List[WorktrailEvent]:
        """Get all events for a run."""
        ...

    async def get_by_step(self, run_id: str, step_id: str) -> List[WorktrailEvent]:
        """Get events for a specific step."""
        ...

    async def get_by_type(
        self, run_id: str, event_type: EventType
    ) -> List[WorktrailEvent]:
        """Get events by type."""
        ...


class MemoryEventStore:
    """
    In-memory event store for testing and development.

    Args:
        max_events: Maximum number of events to keep (default: 100,000).
                   Oldest events are evicted when limit is reached.
    """

    DEFAULT_MAX_EVENTS = 100_000

    def __init__(self, max_events: int = DEFAULT_MAX_EVENTS) -> None:
        self._max_events = max_events
        self._events: Deque[WorktrailEvent] = deque(maxlen=max_events)

    async def append(self, event: WorktrailEvent) -> None:
        """Append event to store."""
        self._events.append(event)

    async def get_by_run(self, run_id: str) -> List[WorktrailEvent]:
        """Get all events for a run."""
        return [e for e in self._events if e.run_id == run_id]

    async def get_by_step(self, run_id: str, step_id: str) -> List[WorktrailEvent]:
        """Get events for a specific step."""
        return [
            e for e in self._events if e.run_id == run_id and e.step_id == step_id
        ]

    async def get_by_type(
        self, run_id: str, event_type: EventType
    ) -> List[WorktrailEvent]:
        """Get events by type."""
        return [
            e for e in self._events if e.run_id == run_id and e.event_type == event_type
        ]

    async def get_by_segment(self, segment_id: str) -> List[WorktrailEvent]:
        """Get all events for a segment."""
        return [e for e in self._events if e.segment_id == segment_id]

    async def get_by_tenant(self, tenant_id: str) -> List[WorktrailEvent]:
        """Get all events for a tenant."""
        return [e for e in self._events if e.tenant_id == tenant_id]

    async def get_by_time_range(
        self,
        run_id: str,
        start: datetime,
        end: datetime,
    ) -> List[WorktrailEvent]:
        """Get events within a time range."""
        return [
            e
            for e in self._events
            if e.run_id == run_id and start <= e.timestamp <= end
        ]

    async def get_errors(self, run_id: str) -> List[WorktrailEvent]:
        """Get all error events for a run."""
        return [
            e for e in self._events if e.run_id == run_id and e.error is not None
        ]

    async def count_by_run(self, run_id: str) -> int:
        """Count events for a run."""
        return len([e for e in self._events if e.run_id == run_id])

    async def clear(self) -> None:
        """Clear all events."""
        self._events.clear()

    async def clear_by_run(self, run_id: str) -> int:
        """Clear events for a run. Returns number of removed events."""
        original_count = len(self._events)
        self._events = [e for e in self._events if e.run_id != run_id]
        return original_count - len(self._events)
