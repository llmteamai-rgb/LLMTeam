"""
In-Memory Snapshot Store.
"""

from collections import OrderedDict
from typing import Dict, List, Optional

from .base import SnapshotStore
from ..models import PipelineSnapshot


class MemorySnapshotStore(SnapshotStore):
    """
    In-memory implementation of snapshot store.

    Args:
        max_snapshots: Maximum number of snapshots to keep (default: 10,000).
                      Oldest snapshots are evicted when limit is reached.
    """

    DEFAULT_MAX_SNAPSHOTS = 10_000

    def __init__(self, max_snapshots: int = DEFAULT_MAX_SNAPSHOTS) -> None:
        self._max_snapshots = max_snapshots
        self._snapshots: OrderedDict[str, PipelineSnapshot] = OrderedDict()
        self._by_run: Dict[str, List[str]] = {}  # run_id -> [snapshot_ids]
        self._by_pipeline: Dict[str, List[str]] = {}  # pipeline_id -> [snapshot_ids]

    async def save(self, snapshot: PipelineSnapshot) -> None:
        """Save snapshot."""
        # Compute checksum before saving
        snapshot.checksum = snapshot.compute_checksum()

        # Evict oldest if at capacity
        while len(self._snapshots) >= self._max_snapshots:
            oldest_id, oldest_snapshot = self._snapshots.popitem(last=False)
            # Clean up indices for evicted snapshot
            if oldest_snapshot.run_id in self._by_run:
                try:
                    self._by_run[oldest_snapshot.run_id].remove(oldest_id)
                except ValueError:
                    pass
            if oldest_snapshot.pipeline_id in self._by_pipeline:
                try:
                    self._by_pipeline[oldest_snapshot.pipeline_id].remove(oldest_id)
                except ValueError:
                    pass

        # Store snapshot
        self._snapshots[snapshot.snapshot_id] = snapshot

        # Index by run_id
        if snapshot.run_id not in self._by_run:
            self._by_run[snapshot.run_id] = []
        self._by_run[snapshot.run_id].append(snapshot.snapshot_id)

        # Index by pipeline_id
        if snapshot.pipeline_id not in self._by_pipeline:
            self._by_pipeline[snapshot.pipeline_id] = []
        self._by_pipeline[snapshot.pipeline_id].append(snapshot.snapshot_id)

    async def load(self, snapshot_id: str) -> Optional[PipelineSnapshot]:
        """Load snapshot by ID."""
        snapshot = self._snapshots.get(snapshot_id)
        if snapshot and not snapshot.verify():
            # Checksum mismatch - data corruption
            return None
        return snapshot

    async def load_latest(self, run_id: str) -> Optional[PipelineSnapshot]:
        """Load latest snapshot for run."""
        snapshot_ids = self._by_run.get(run_id, [])
        if not snapshot_ids:
            return None

        # Get latest (last in list)
        latest_id = snapshot_ids[-1]
        return await self.load(latest_id)

    async def list(self, pipeline_id: str, limit: int = 10) -> List[PipelineSnapshot]:
        """List snapshots for pipeline."""
        snapshot_ids = self._by_pipeline.get(pipeline_id, [])

        # Get most recent snapshots
        recent_ids = snapshot_ids[-limit:] if len(snapshot_ids) > limit else snapshot_ids

        # Load snapshots
        snapshots = []
        for sid in reversed(recent_ids):  # Most recent first
            snapshot = await self.load(sid)
            if snapshot:
                snapshots.append(snapshot)

        return snapshots

    async def delete(self, snapshot_id: str) -> None:
        """Delete snapshot."""
        snapshot = self._snapshots.pop(snapshot_id, None)
        if not snapshot:
            return

        # Remove from indices
        if snapshot.run_id in self._by_run:
            try:
                self._by_run[snapshot.run_id].remove(snapshot_id)
            except ValueError:
                pass

        if snapshot.pipeline_id in self._by_pipeline:
            try:
                self._by_pipeline[snapshot.pipeline_id].remove(snapshot_id)
            except ValueError:
                pass
