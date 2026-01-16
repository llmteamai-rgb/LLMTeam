"""
Persistence Module (v1.9.0).

Provides pipeline persistence for pause/resume:
- Pipeline snapshots
- Agent state snapshots
- Restore functionality
- Integrity verification

Usage:
    from llmteam.persistence import (
        SnapshotManager,
        PipelineSnapshot,
        PipelinePhase,
        SnapshotType,
    )

    # Create manager
    store = MemorySnapshotStore()
    manager = SnapshotManager(store, audit_trail)

    # Create snapshot
    snapshot = await manager.create_snapshot(
        pipeline_id="pipeline_1",
        run_id="run_123",
        phase=PipelinePhase.RUNNING,
        global_state={"step": 1, "data": "value"},
        snapshot_type=SnapshotType.CHECKPOINT,
    )

    # Restore from snapshot
    result = await manager.restore_snapshot(snapshot.snapshot_id)
"""

from .models import (
    SnapshotType,
    PipelinePhase,
    AgentSnapshot,
    PipelineSnapshot,
    RestoreResult,
)
from .manager import SnapshotManager
from .stores import SnapshotStore, MemorySnapshotStore

__all__ = [
    # Enums
    "SnapshotType",
    "PipelinePhase",
    # Models
    "AgentSnapshot",
    "PipelineSnapshot",
    "RestoreResult",
    # Manager & Store
    "SnapshotManager",
    "SnapshotStore",
    "MemorySnapshotStore",
]
