"""
Audit trail for llmteam.

This module provides compliance-ready audit logging with:
- Immutable audit records with chain integrity
- Query interface for audit search
- Report generation for compliance

Quick Start:
    from llmteam.audit import AuditTrail, AuditEventType
    from llmteam.audit.stores import MemoryAuditStore
    
    # Create audit trail
    store = MemoryAuditStore()
    audit = AuditTrail(store, tenant_id="acme")
    
    # Log events
    await audit.log(
        AuditEventType.PIPELINE_STARTED,
        actor_id="user@acme.com",
        resource_id="pipeline_123",
    )
    
    # Query events
    records = await audit.query(AuditQuery(
        start_time=datetime(2024, 1, 1),
        event_types=[AuditEventType.PIPELINE_STARTED],
    ))
    
    # Generate compliance report
    report = await audit.generate_report(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 12, 31),
    )
"""

from llmteam.audit.models import (
    AuditEventType,
    AuditSeverity,
    AuditRecord,
    AuditQuery,
    AuditReport,
    AuditError,
    AuditIntegrityError,
    AuditStorageError,
    generate_audit_id,
)

from llmteam.audit.trail import (
    AuditStore,
    AuditTrail,
    TenantAwareAuditTrail,
)

__all__ = [
    # Event types and severity
    "AuditEventType",
    "AuditSeverity",
    
    # Data structures
    "AuditRecord",
    "AuditQuery",
    "AuditReport",
    
    # Exceptions
    "AuditError",
    "AuditIntegrityError",
    "AuditStorageError",
    
    # Utilities
    "generate_audit_id",
    
    # Trail
    "AuditStore",
    "AuditTrail",
    "TenantAwareAuditTrail",
]
