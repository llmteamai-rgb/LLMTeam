"""
Base Snapshot Store.
"""

from typing import List, Optional

from ..models import PipelineSnapshot


class SnapshotStore:
    """Base class for snapshot storage."""

    async def save(self, snapshot: PipelineSnapshot) -> None:
        """Save snapshot."""
        raise NotImplementedError

    async def load(self, snapshot_id: str) -> Optional[PipelineSnapshot]:
        """Load snapshot by ID."""
        raise NotImplementedError

    async def load_latest(self, run_id: str) -> Optional[PipelineSnapshot]:
        """Load latest snapshot for run."""
        raise NotImplementedError

    async def list(self, pipeline_id: str, limit: int = 10) -> List[PipelineSnapshot]:
        """List snapshots for pipeline."""
        raise NotImplementedError

    async def delete(self, snapshot_id: str) -> None:
        """Delete snapshot."""
        raise NotImplementedError
