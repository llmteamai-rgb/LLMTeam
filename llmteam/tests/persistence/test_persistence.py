"""
Tests for Persistence module (v1.9.0).
"""

import pytest

from llmteam.persistence import (
    SnapshotType,
    PipelinePhase,
    AgentSnapshot,
    PipelineSnapshot,
    RestoreResult,
    SnapshotManager,
    MemorySnapshotStore,
)


class TestSnapshotModels:
    """Tests for snapshot data models."""

    def test_agent_snapshot(self):
        """Test AgentSnapshot creation."""
        snapshot = AgentSnapshot(
            agent_name="agent_1",
            state={"step": 1},
            context={"key": "value"},
        )

        assert snapshot.agent_name == "agent_1"
        assert snapshot.state["step"] == 1

    def test_pipeline_snapshot(self):
        """Test PipelineSnapshot creation."""
        snapshot = PipelineSnapshot(
            snapshot_id="snap_123",
            snapshot_type=SnapshotType.MANUAL,
            pipeline_id="pipeline_1",
            run_id="run_123",
            tenant_id="tenant_1",
            pipeline_version="1.0",
            phase=PipelinePhase.RUNNING,
            global_state={"data": "value"},
        )

        assert snapshot.snapshot_id == "snap_123"
        assert snapshot.phase == PipelinePhase.RUNNING

    def test_compute_checksum(self):
        """Test checksum computation."""
        snapshot = PipelineSnapshot(
            snapshot_id="snap_123",
            snapshot_type=SnapshotType.MANUAL,
            pipeline_id="pipeline_1",
            run_id="run_123",
            tenant_id="tenant_1",
            pipeline_version="1.0",
            phase=PipelinePhase.RUNNING,
            global_state={"key": "value"},
        )

        checksum1 = snapshot.compute_checksum()
        checksum2 = snapshot.compute_checksum()

        assert checksum1 == checksum2
        assert len(checksum1) == 64  # SHA-256


class TestMemorySnapshotStore:
    """Tests for MemorySnapshotStore."""

    @pytest.mark.asyncio
    async def test_save_and_load(self):
        """Test saving and loading snapshot."""
        store = MemorySnapshotStore()

        snapshot = PipelineSnapshot(
            snapshot_id="snap_123",
            snapshot_type=SnapshotType.MANUAL,
            pipeline_id="pipeline_1",
            run_id="run_123",
            tenant_id="tenant_1",
            pipeline_version="1.0",
            phase=PipelinePhase.RUNNING,
            global_state={"data": "value"},
        )

        await store.save(snapshot)
        loaded = await store.load("snap_123")

        assert loaded is not None
        assert loaded.snapshot_id == "snap_123"
        assert loaded.phase == PipelinePhase.RUNNING

    @pytest.mark.asyncio
    async def test_load_latest(self):
        """Test loading latest snapshot for run."""
        store = MemorySnapshotStore()

        # Create multiple snapshots
        for i in range(3):
            snapshot = PipelineSnapshot(
                snapshot_id=f"snap_{i}",
                snapshot_type=SnapshotType.AUTO,
                pipeline_id="pipeline_1",
                run_id="run_123",
                tenant_id="tenant_1",
                pipeline_version="1.0",
                phase=PipelinePhase.RUNNING,
                global_state={"step": i},
            )
            await store.save(snapshot)

        # Get latest
        latest = await store.load_latest("run_123")

        assert latest is not None
        assert latest.snapshot_id == "snap_2"
        assert latest.global_state["step"] == 2


class TestSnapshotManager:
    """Tests for SnapshotManager."""

    @pytest.fixture
    def manager(self):
        """Create manager with memory store."""
        store = MemorySnapshotStore()
        return SnapshotManager(store)

    @pytest.mark.asyncio
    async def test_create_snapshot(self, manager):
        """Test creating snapshot."""
        snapshot = await manager.create_snapshot(
            pipeline_id="pipeline_1",
            run_id="run_123",
            phase=PipelinePhase.RUNNING,
            global_state={"data": "value"},
            snapshot_type=SnapshotType.CHECKPOINT,
        )

        assert snapshot.phase == PipelinePhase.RUNNING
        assert snapshot.snapshot_type == SnapshotType.CHECKPOINT
        assert snapshot.checksum != ""  # Checksum should be computed

    @pytest.mark.asyncio
    async def test_restore_snapshot(self, manager):
        """Test restoring from snapshot."""
        # Create snapshot
        snapshot = await manager.create_snapshot(
            pipeline_id="pipeline_1",
            run_id="run_123",
            phase=PipelinePhase.PAUSED,
            global_state={"data": "value"},
        )

        # Restore
        result = await manager.restore_snapshot(snapshot.snapshot_id)

        assert result is not None
        assert result.success is True
        assert result.phase == PipelinePhase.PAUSED
        assert result.run_id == "run_123"
