"""
Event Stream for real-time event delivery to canvas UI.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Dict, List, Optional

from llmteam.events.models import WorktrailEvent
from llmteam.events.store import EventStore, MemoryEventStore


class EventStream:
    """Streaming events for canvas UI."""

    def __init__(self, store: Optional[EventStore] = None) -> None:
        self.store: EventStore = store or MemoryEventStore()
        self._subscribers: Dict[str, List[asyncio.Queue[Optional[WorktrailEvent]]]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, run_id: str) -> AsyncIterator[WorktrailEvent]:
        """
        Subscribe to events for a run.

        Yields events as they are published.
        """
        queue: asyncio.Queue[Optional[WorktrailEvent]] = asyncio.Queue()

        async with self._lock:
            if run_id not in self._subscribers:
                self._subscribers[run_id] = []
            self._subscribers[run_id].append(queue)

        try:
            while True:
                event = await queue.get()
                if event is None:  # Unsubscribe signal
                    break
                yield event
        finally:
            async with self._lock:
                if run_id in self._subscribers:
                    try:
                        self._subscribers[run_id].remove(queue)
                        if not self._subscribers[run_id]:
                            del self._subscribers[run_id]
                    except ValueError:
                        pass

    async def publish(self, event: WorktrailEvent) -> None:
        """Publish event to subscribers and store."""
        # Store event
        await self.store.append(event)

        # Notify subscribers
        run_id = event.run_id
        async with self._lock:
            if run_id in self._subscribers:
                for queue in self._subscribers[run_id]:
                    await queue.put(event)

    async def get_history(self, run_id: str) -> List[WorktrailEvent]:
        """Get historical events for a run."""
        return await self.store.get_by_run(run_id)

    async def unsubscribe_all(self, run_id: str) -> None:
        """Unsubscribe all listeners for a run."""
        async with self._lock:
            if run_id in self._subscribers:
                for queue in self._subscribers[run_id]:
                    await queue.put(None)  # Signal to stop
                del self._subscribers[run_id]

    async def get_subscriber_count(self, run_id: str) -> int:
        """Get number of subscribers for a run."""
        async with self._lock:
            return len(self._subscribers.get(run_id, []))

    async def has_subscribers(self, run_id: str) -> bool:
        """Check if run has any subscribers."""
        async with self._lock:
            return run_id in self._subscribers and len(self._subscribers[run_id]) > 0
