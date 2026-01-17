"""
Audit trail implementation.

This module provides AuditTrail - the main class for recording and
querying audit events.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from llmteam.observability import get_logger
from llmteam.audit.models import (
    AuditRecord,
    AuditQuery,
    AuditReport,
    AuditEventType,
    AuditSeverity,
    AuditIntegrityError,
    generate_audit_id,
)
from llmteam.tenancy.context import current_tenant


logger = get_logger(__name__)


@runtime_checkable
class AuditStore(Protocol):
    """
    Protocol for audit storage backends.
    
    Audit stores must be append-only for compliance requirements.
    """
    
    async def append(self, record: AuditRecord) -> None:
        """Append a record (no updates allowed)."""
        ...
    
    async def query(self, query: AuditQuery) -> List[AuditRecord]:
        """Query records with filters."""
        ...
    
    async def get_last_sequence(self, tenant_id: str) -> int:
        """Get the last sequence number for a tenant."""
        ...
    
    async def get_by_sequence_range(
        self, 
        tenant_id: str, 
        start: int, 
        end: int,
    ) -> List[AuditRecord]:
        """Get records by sequence number range."""
        ...


from llmteam.licensing import enterprise_only


@enterprise_only
class AuditTrail:
    """
    Main class for audit logging.
    
    Provides:
    - Thread-safe event logging
    - Query interface for audit records
    - Chain integrity verification
    - Report generation
    
    Example:
        from llmteam.audit import AuditTrail, AuditEventType
        from llmteam.audit.stores import MemoryAuditStore
        
        store = MemoryAuditStore()
        audit = AuditTrail(store, tenant_id="acme")
        
        # Log an event
        await audit.log(
            AuditEventType.PIPELINE_STARTED,
            actor_id="user@acme.com",
            resource_id="pipeline_123",
        )
        
        # Query events
        records = await audit.query(AuditQuery(
            event_types=[AuditEventType.PIPELINE_STARTED],
            limit=10,
        ))
    """
    
    def __init__(
        self,
        store: AuditStore,
        tenant_id: str = "",
        auto_flush: bool = True,
        buffer_size: int = 100,
        use_context_tenant: bool = True,
    ):
        """
        Initialize AuditTrail.
        
        Args:
            store: Storage backend for audit records
            tenant_id: Default tenant ID (can be overridden by context)
            auto_flush: Whether to auto-flush buffer when full
            buffer_size: Number of records to buffer before flushing
            use_context_tenant: Whether to use current_tenant context variable
        """
        self.store = store
        self._default_tenant_id = tenant_id
        self.auto_flush = auto_flush
        self.buffer_size = buffer_size
        self.use_context_tenant = use_context_tenant
        
        self._buffer: List[AuditRecord] = []
        self._sequence: int = 0
        self._last_checksum: str = ""
        self._lock = asyncio.Lock()
        self._initialized = False
        
        logger.debug(f"AuditTrail initialized (tenant={tenant_id}, buffer_size={buffer_size})")
    
    @property
    def tenant_id(self) -> str:
        """Get the effective tenant ID."""
        if self.use_context_tenant:
            ctx_tenant = current_tenant.get()
            if ctx_tenant:
                return ctx_tenant
        return self._default_tenant_id or "default"
    
    async def initialize(self) -> None:
        """
        Initialize the audit trail.
        
        Loads the last sequence number and checksum from storage.
        Should be called before first use.
        """
        if self._initialized:
            return
        
        async with self._lock:
            try:
                self._sequence = await self.store.get_last_sequence(self.tenant_id)
                logger.debug(f"Initialized audit trail sequence: {self._sequence}")
                
                # Load last checksum if there are existing records
                if self._sequence > 0:
                    records = await self.store.get_by_sequence_range(
                        self.tenant_id,
                        self._sequence,
                        self._sequence,
                    )
                    if records:
                        self._last_checksum = records[0].checksum
                
                self._initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize audit trail: {str(e)}")
                raise
    
    async def log(
        self,
        event_type: AuditEventType,
        *,
        severity: AuditSeverity = AuditSeverity.INFO,
        actor_type: str = "system",
        actor_id: str = "",
        actor_ip: str = "",
        action: str = "",
        resource_type: str = "",
        resource_id: str = "",
        old_value: Dict[str, Any] = None,
        new_value: Dict[str, Any] = None,
        success: bool = True,
        error_message: str = "",
        metadata: Dict[str, Any] = None,
        tags: List[str] = None,
        pipeline_id: str = "",
        run_id: str = "",
        agent_name: str = "",
        step_name: str = "",
    ) -> AuditRecord:
        """
        Log an audit event.
        
        Thread-safe. Records are buffered and flushed periodically.
        
        Args:
            event_type: Type of event
            severity: Event severity level
            actor_type: Type of actor (user, agent, system, external)
            actor_id: Identifier of the actor
            actor_ip: IP address of actor
            action: Description of the action
            resource_type: Type of resource affected
            resource_id: ID of resource affected
            old_value: Previous state (for changes)
            new_value: New state (for changes)
            success: Whether operation succeeded
            error_message: Error message if failed
            metadata: Additional event data
            tags: Tags for categorization
            pipeline_id: Associated pipeline ID
            run_id: Associated run ID
            agent_name: Associated agent name
            step_name: Associated step name
            
        Returns:
            The created AuditRecord
        """
        if not self._initialized:
            await self.initialize()
        
        async with self._lock:
            self._sequence += 1
            
            record = AuditRecord(
                record_id=generate_audit_id(),
                sequence_number=self._sequence,
                timestamp=datetime.now(timezone.utc),
                event_type=event_type,
                severity=severity,
                tenant_id=self.tenant_id,
                actor_type=actor_type,
                actor_id=actor_id,
                actor_ip=actor_ip,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                old_value=old_value,
                new_value=new_value,
                success=success,
                error_message=error_message,
                metadata=metadata or {},
                tags=tags or [],
                previous_checksum=self._last_checksum,
                pipeline_id=pipeline_id,
                run_id=run_id,
                agent_name=agent_name,
                step_name=step_name,
            )
            
            self._last_checksum = record.checksum
            self._buffer.append(record)
            
            if self.auto_flush and len(self._buffer) >= self.buffer_size:
                logger.debug(f"Buffer full ({len(self._buffer)} records), flushing")
                await self._flush_unlocked()
            
            return record
    
    async def _flush_unlocked(self) -> None:
        """Flush buffer without acquiring lock (caller must hold lock)."""
        if not self._buffer:
            return
        
        count = len(self._buffer)
        try:
            for record in self._buffer:
                await self.store.append(record)
            
            self._buffer.clear()
            logger.debug(f"Flushed {count} audit records")
            
        except Exception as e:
            logger.error(f"Failed to flush audit records: {str(e)}")
            # We don't clear the buffer on error so we can retry
            raise
    
    async def flush(self) -> None:
        """
        Flush buffered records to storage.
        
        Call this to ensure all records are persisted.
        """
        async with self._lock:
            await self._flush_unlocked()
    
    async def query(self, query: AuditQuery) -> List[AuditRecord]:
        """
        Query audit records.
        
        Args:
            query: Query filters
            
        Returns:
            List of matching AuditRecord objects
        """
        # Ensure buffer is flushed
        await self.flush()
        
        # Set tenant ID if not specified
        if query.tenant_id is None:
            query.tenant_id = self.tenant_id
        
        return await self.store.query(query)
    
    async def verify_chain(
        self, 
        start_sequence: int = 1, 
        end_sequence: int = None,
    ) -> Tuple[bool, List[int]]:
        """
        Verify integrity of the audit chain.
        
        Checks that:
        - All sequence numbers are present
        - Each record's checksum is valid
        - Each record's previous_checksum matches the prior record
        
        Args:
            start_sequence: Starting sequence number
            end_sequence: Ending sequence number (default: latest)
            
        Returns:
            Tuple of (is_valid, missing_sequence_numbers)
        """
        await self.flush()
        
        if end_sequence is None:
            end_sequence = await self.store.get_last_sequence(self.tenant_id)
        
        if end_sequence == 0:
            return True, []  # No records
        
        records = await self.store.get_by_sequence_range(
            self.tenant_id,
            start_sequence,
            end_sequence,
        )
        
        # Check for missing sequence numbers
        expected_sequences = set(range(start_sequence, end_sequence + 1))
        actual_sequences = {r.sequence_number for r in records}
        missing = sorted(expected_sequences - actual_sequences)
        
        if missing:
            logger.error(f"Integrity check failed: missing {len(missing)} records in range [{start_sequence}, {end_sequence}]")
            return False, missing
        
        # Sort by sequence
        records.sort(key=lambda r: r.sequence_number)
        
        # Verify chain
        prev_checksum = ""
        for i, record in enumerate(records):
            # Verify record integrity
            if not record.verify_integrity():
                logger.error(f"Integrity check failed: record {record.record_id} (seq {record.sequence_number}) corrupted")
                return False, []
            
            # Verify chain linkage (skip first record)
            if i > 0 and record.previous_checksum != prev_checksum:
                logger.error(f"Integrity check failed: chain broken at seq {record.sequence_number}")
                return False, []
            
            prev_checksum = record.checksum
            
        logger.info(f"Integrity check passed for range [{start_sequence}, {end_sequence}]")
        return True, []
    
    async def generate_report(
        self,
        start: datetime,
        end: datetime,
        include_records: bool = False,
    ) -> AuditReport:
        """
        Generate an audit report for compliance.
        
        Args:
            start: Report start time
            end: Report end time
            include_records: Whether to include raw records
            
        Returns:
            AuditReport with statistics and optionally records
        """
        logger.info(f"Generating audit report for period {start} to {end}")
        
        query = AuditQuery(
            start_time=start,
            end_time=end,
            tenant_id=self.tenant_id,
            limit=10000,  # Reasonable limit for report
        )
        
        records = await self.query(query)
        
        # Aggregate statistics
        events_by_type: Dict[str, int] = {}
        events_by_severity: Dict[str, int] = {}
        events_by_actor: Dict[str, int] = {}
        events_by_day: Dict[str, int] = {}
        security_incidents: List[AuditRecord] = []
        failed_operations: List[AuditRecord] = []
        
        for record in records:
            # By type
            type_key = record.event_type.value
            events_by_type[type_key] = events_by_type.get(type_key, 0) + 1
            
            # By severity
            sev_key = record.severity.value
            events_by_severity[sev_key] = events_by_severity.get(sev_key, 0) + 1
            
            # By actor
            if record.actor_id:
                events_by_actor[record.actor_id] = events_by_actor.get(record.actor_id, 0) + 1
            
            # By day
            day_key = record.timestamp.strftime("%Y-%m-%d")
            events_by_day[day_key] = events_by_day.get(day_key, 0) + 1
            
            # Security incidents
            if record.event_type.value.startswith("security.") and not record.success:
                security_incidents.append(record)
            
            # Failed operations
            if not record.success:
                failed_operations.append(record)
        
        # Verify chain integrity
        chain_valid = True
        missing_sequences: List[int] = []
        
        if records:
            min_seq = min(r.sequence_number for r in records)
            max_seq = max(r.sequence_number for r in records)
            chain_valid, missing_sequences = await self.verify_chain(min_seq, max_seq)
        
        logger.info(f"Audit report generated: {len(records)} events, valid={chain_valid}")

        return AuditReport(
            report_id=generate_audit_id(),
            generated_at=datetime.now(timezone.utc),
            generated_by="system",
            period_start=start,
            period_end=end,
            tenant_id=self.tenant_id,
            total_events=len(records),
            events_by_type=events_by_type,
            events_by_severity=events_by_severity,
            events_by_actor=events_by_actor,
            events_by_day=events_by_day,
            security_incidents=security_incidents[:100],  # Limit
            failed_operations=failed_operations[:100],  # Limit
            records=records if include_records else [],
            chain_valid=chain_valid,
            missing_sequence_numbers=missing_sequences,
        )
    
    # Convenience methods
    
    async def log_pipeline_started(
        self,
        pipeline_id: str,
        run_id: str,
        actor_id: str,
        input_summary: Dict[str, Any] = None,
    ) -> AuditRecord:
        """Log pipeline start event."""
        return await self.log(
            AuditEventType.PIPELINE_STARTED,
            actor_type="user",
            actor_id=actor_id,
            action="start",
            resource_type="pipeline",
            resource_id=pipeline_id,
            pipeline_id=pipeline_id,
            run_id=run_id,
            new_value={"input_summary": input_summary} if input_summary else None,
        )
    
    async def log_pipeline_completed(
        self,
        pipeline_id: str,
        run_id: str,
        output_summary: Dict[str, Any] = None,
    ) -> AuditRecord:
        """Log pipeline completion event."""
        return await self.log(
            AuditEventType.PIPELINE_COMPLETED,
            action="complete",
            resource_type="pipeline",
            resource_id=pipeline_id,
            pipeline_id=pipeline_id,
            run_id=run_id,
            new_value={"output_summary": output_summary} if output_summary else None,
        )
    
    async def log_pipeline_failed(
        self,
        pipeline_id: str,
        run_id: str,
        error: str,
    ) -> AuditRecord:
        """Log pipeline failure event."""
        return await self.log(
            AuditEventType.PIPELINE_FAILED,
            severity=AuditSeverity.ERROR,
            action="fail",
            resource_type="pipeline",
            resource_id=pipeline_id,
            pipeline_id=pipeline_id,
            run_id=run_id,
            success=False,
            error_message=error,
        )
    
    async def log_access_denied(
        self,
        actor_id: str,
        resource_type: str,
        resource_id: str,
        reason: str,
    ) -> AuditRecord:
        """Log access denied event."""
        return await self.log(
            AuditEventType.ACCESS_DENIED,
            severity=AuditSeverity.WARNING,
            actor_type="user",
            actor_id=actor_id,
            action="access_denied",
            resource_type=resource_type,
            resource_id=resource_id,
            success=False,
            error_message=reason,
        )
    
    async def log_approval(
        self,
        approved: bool,
        approver_id: str,
        request_id: str,
        reason: str = "",
    ) -> AuditRecord:
        """Log approval decision."""
        event_type = AuditEventType.APPROVAL_GRANTED if approved else AuditEventType.APPROVAL_REJECTED
        
        return await self.log(
            event_type,
            severity=AuditSeverity.INFO if approved else AuditSeverity.WARNING,
            actor_type="user",
            actor_id=approver_id,
            action="approve" if approved else "reject",
            resource_type="approval_request",
            resource_id=request_id,
            success=approved,
            metadata={"reason": reason} if reason else {},
        )
    
    async def log_config_change(
        self,
        actor_id: str,
        resource_type: str,
        resource_id: str,
        old_value: Dict[str, Any],
        new_value: Dict[str, Any],
    ) -> AuditRecord:
        """Log configuration change."""
        return await self.log(
            AuditEventType.CONFIG_CHANGED,
            actor_type="user",
            actor_id=actor_id,
            action="update",
            resource_type=resource_type,
            resource_id=resource_id,
            old_value=old_value,
            new_value=new_value,
        )


class TenantAwareAuditTrail(AuditTrail):
    """
    AuditTrail that automatically uses the current tenant context.
    
    Always uses the tenant from current_tenant context variable.
    """
    
    def __init__(self, store: AuditStore):
        super().__init__(
            store=store,
            tenant_id="",
            use_context_tenant=True,
        )
