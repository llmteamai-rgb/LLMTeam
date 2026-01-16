"""
Snapshot Manager.

Manages pipeline snapshots for pause/resume functionality:
- Create snapshots
- Restore from snapshots
- Auto-snapshot scheduling
"""

import uuid
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from .models import (
    PipelineSnapshot,
    AgentSnapshot,
    RestoreResult,
    SnapshotType,
    PipelinePhase,
)
from .stores.base import SnapshotStore

if TYPE_CHECKING:
    from ..audit import AuditTrail, AuditEventType


def generate_uuid() -> str:
    """Generate UUID4 string."""
    return str(uuid.uuid4())


class SnapshotManager:
    """
    Manager for pipeline snapshots.

    Integrates with:
    - AuditTrail from v1.7.0
    - TenantContext from v1.7.0
    """

    def __init__(
        self,
        store: SnapshotStore,
        audit_trail: Optional["AuditTrail"] = None,
    ):
        self.store = store
        self.audit_trail = audit_trail

    async def create_snapshot(
        self,
        pipeline_id: str,
        run_id: str,
        phase: PipelinePhase,
        global_state: dict,
        snapshot_type: SnapshotType = SnapshotType.MANUAL,
        **kwargs,
    ) -> PipelineSnapshot:
        """
        Create a pipeline snapshot.

        Args:
            pipeline_id: Pipeline identifier
            run_id: Run identifier
            phase: Current pipeline phase
            global_state: Pipeline state data
            snapshot_type: Type of snapshot
            **kwargs: Additional snapshot parameters

        Returns:
            Created snapshot
        """
        from ..tenancy import current_tenant

        snapshot = PipelineSnapshot(
            snapshot_id=generate_uuid(),
            snapshot_type=snapshot_type,
            pipeline_id=pipeline_id,
            run_id=run_id,
            tenant_id=current_tenant.get(),
            pipeline_version=kwargs.get("pipeline_version", "1.0"),
            phase=phase,
            global_state=global_state,
            **{k: v for k, v in kwargs.items() if k != "pipeline_version"},
        )

        # Compute and set checksum
        snapshot.checksum = snapshot.compute_checksum()

        # Save to store
        await self.store.save(snapshot)

        # Audit logging
        if self.audit_trail:
            from ..audit import AuditEventType

            await self.audit_trail.log(
                AuditEventType.PIPELINE_SNAPSHOT_CREATED,
                resource_type="pipeline_snapshot",
                resource_id=snapshot.snapshot_id,
                metadata={
                    "pipeline_id": pipeline_id,
                    "run_id": run_id,
                    "phase": phase.value,
                    "type": snapshot_type.value,
                },
            )

        return snapshot

    async def save_agent_snapshot(
        self,
        snapshot: PipelineSnapshot,
        agent_name: str,
        agent_state: dict,
        agent_context: dict,
        **kwargs,
    ) -> None:
        """
        Save agent state to snapshot.

        Args:
            snapshot: Pipeline snapshot
            agent_name: Agent name
            agent_state: Agent state data
            agent_context: Agent context data
            **kwargs: Additional agent snapshot parameters
        """
        agent_snapshot = AgentSnapshot(
            agent_name=agent_name,
            state=agent_state,
            context=agent_context,
            **kwargs,
        )

        snapshot.agent_snapshots[agent_name] = agent_snapshot

        # Update snapshot in store
        await self.store.save(snapshot)

    async def restore_snapshot(
        self,
        snapshot_id: str,
    ) -> Optional[RestoreResult]:
        """
        Restore pipeline from snapshot.

        Args:
            snapshot_id: Snapshot identifier

        Returns:
            RestoreResult with restored state, or None if not found
        """
        # Load snapshot
        snapshot = await self.store.load(snapshot_id)
        if not snapshot:
            return None

        # Verify integrity
        if not snapshot.verify():
            return RestoreResult(
                success=False,
                snapshot_id=snapshot_id,
                run_id=snapshot.run_id,
                phase=PipelinePhase.FAILED,
                current_step="",
                warnings=["Checksum verification failed - snapshot may be corrupted"],
            )

        # Create restore result
        result = RestoreResult(
            success=True,
            snapshot_id=snapshot_id,
            run_id=snapshot.run_id,
            phase=snapshot.phase,
            current_step=snapshot.current_step,
        )

        # Audit logging
        if self.audit_trail:
            from ..audit import AuditEventType

            await self.audit_trail.log(
                AuditEventType.PIPELINE_SNAPSHOT_RESTORED,
                resource_type="pipeline_snapshot",
                resource_id=snapshot_id,
                metadata={
                    "pipeline_id": snapshot.pipeline_id,
                    "run_id": snapshot.run_id,
                    "phase": snapshot.phase.value,
                },
            )

        return result

    async def get_snapshot(
        self,
        snapshot_id: str,
    ) -> Optional[PipelineSnapshot]:
        """Get snapshot by ID."""
        return await self.store.load(snapshot_id)

    async def get_latest_snapshot(
        self,
        run_id: str,
    ) -> Optional[PipelineSnapshot]:
        """Get latest snapshot for run."""
        return await self.store.load_latest(run_id)

    async def list_snapshots(
        self,
        pipeline_id: str,
        limit: int = 10,
    ) -> list[PipelineSnapshot]:
        """List snapshots for pipeline."""
        return await self.store.list(pipeline_id, limit)

    async def delete_snapshot(
        self,
        snapshot_id: str,
    ) -> None:
        """Delete snapshot."""
        await self.store.delete(snapshot_id)

        # Audit logging
        if self.audit_trail:
            from ..audit import AuditEventType

            await self.audit_trail.log(
                AuditEventType.PIPELINE_SNAPSHOT_DELETED,
                resource_type="pipeline_snapshot",
                resource_id=snapshot_id,
            )
