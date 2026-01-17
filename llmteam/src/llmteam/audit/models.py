"""
Audit trail models and data structures.

This module defines the core data structures for audit logging:
- AuditEventType: Types of auditable events
- AuditSeverity: Event severity levels
- AuditRecord: Immutable audit record
- AuditQuery: Query filters for audit search
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict
import hashlib
import json
import uuid


class AuditEventType(Enum):
    """
    Types of audit events.
    
    Events are grouped by category:
    - pipeline.*: Pipeline lifecycle events
    - agent.*: Agent execution events
    - security.*: Security-related events
    - config.*: Configuration changes
    - data.*: Data operations
    """
    
    # Pipeline lifecycle
    PIPELINE_CREATED = "pipeline.created"
    PIPELINE_STARTED = "pipeline.started"
    PIPELINE_COMPLETED = "pipeline.completed"
    PIPELINE_FAILED = "pipeline.failed"
    PIPELINE_PAUSED = "pipeline.paused"
    PIPELINE_RESUMED = "pipeline.resumed"
    PIPELINE_CANCELLED = "pipeline.cancelled"
    
    # Agent execution
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    AGENT_RETRIED = "agent.retried"
    
    # Human interaction (for future use)
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_GRANTED = "approval.granted"
    APPROVAL_REJECTED = "approval.rejected"
    APPROVAL_TIMEOUT = "approval.timeout"
    APPROVAL_ESCALATED = "approval.escalated"
    
    # External actions (for future use)
    ACTION_INVOKED = "action.invoked"
    ACTION_COMPLETED = "action.completed"
    ACTION_FAILED = "action.failed"
    
    # Security events
    ACCESS_GRANTED = "security.access_granted"
    ACCESS_DENIED = "security.access_denied"
    CONTEXT_ACCESSED = "security.context_accessed"
    SEALED_DATA_ACCESSED = "security.sealed_data_accessed"
    AUTHENTICATION_SUCCESS = "security.auth_success"
    AUTHENTICATION_FAILURE = "security.auth_failure"
    
    # Configuration changes
    CONFIG_CHANGED = "config.changed"
    POLICY_CHANGED = "config.policy_changed"
    TENANT_CREATED = "config.tenant_created"
    TENANT_UPDATED = "config.tenant_updated"
    TENANT_DELETED = "config.tenant_deleted"
    
    # Data operations
    DATA_CREATED = "data.created"
    DATA_UPDATED = "data.updated"
    DATA_DELETED = "data.deleted"
    DATA_EXPORTED = "data.exported"
    DATA_IMPORTED = "data.imported"


class AuditSeverity(Enum):
    """
    Severity levels for audit events.
    
    Used for filtering and alerting on important events.
    """
    DEBUG = "debug"      # Development/debugging only
    INFO = "info"        # Normal operations
    WARNING = "warning"  # Potential issues
    ERROR = "error"      # Errors that didn't stop execution
    CRITICAL = "critical"  # Critical errors or security events


def generate_audit_id() -> str:
    """Generate a unique audit record ID."""
    return str(uuid.uuid4())


class AuditRecordDict(TypedDict, total=False):
    """Dictionary representation of AuditRecord."""
    record_id: str
    sequence_number: int
    timestamp: str  # ISO format
    event_type: str
    severity: str
    tenant_id: str
    pipeline_id: str
    run_id: str
    agent_name: str
    step_name: str
    actor_type: str
    actor_id: str
    actor_ip: str
    actor_user_agent: str
    action: str
    resource_type: str
    resource_id: str
    old_value: Optional[Dict[str, Any]]
    new_value: Optional[Dict[str, Any]]
    success: bool
    error_message: str
    metadata: Dict[str, Any]
    tags: List[str]
    checksum: str
    previous_checksum: str


@dataclass
class AuditRecord:
    """
    Immutable audit record.
    
    All fields are set at creation time and cannot be modified.
    The checksum field ensures tamper detection.
    
    Attributes:
        record_id: Unique identifier for this record
        sequence_number: Monotonically increasing number for ordering
        timestamp: When the event occurred
        event_type: Type of event
        severity: Event severity
        tenant_id: Associated tenant
        pipeline_id: Associated pipeline (if any)
        run_id: Associated pipeline run (if any)
        agent_name: Associated agent (if any)
        step_name: Associated step (if any)
        actor_type: Type of actor (user, agent, system, external)
        actor_id: Identifier of the actor
        actor_ip: IP address of the actor (if applicable)
        actor_user_agent: User agent string (if applicable)
        action: Description of what happened
        resource_type: Type of resource affected
        resource_id: ID of resource affected
        old_value: Previous state (for changes)
        new_value: New state (for changes)
        success: Whether the operation succeeded
        error_message: Error message if failed
        metadata: Additional event-specific data
        tags: Tags for categorization
        checksum: SHA-256 checksum for integrity
        previous_checksum: Checksum of previous record (for chaining)
    """
    
    # Identity
    record_id: str
    sequence_number: int
    timestamp: datetime
    
    # Event
    event_type: AuditEventType
    severity: AuditSeverity = AuditSeverity.INFO
    
    # Context
    tenant_id: str = ""
    pipeline_id: str = ""
    run_id: str = ""
    agent_name: str = ""
    step_name: str = ""
    
    # Actor
    actor_type: str = ""  # "user", "agent", "system", "external"
    actor_id: str = ""
    actor_ip: str = ""
    actor_user_agent: str = ""
    
    # Details
    action: str = ""
    resource_type: str = ""
    resource_id: str = ""
    
    # State changes
    old_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None
    
    # Result
    success: bool = True
    error_message: str = ""
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Integrity
    checksum: str = ""
    previous_checksum: str = ""
    
    def __post_init__(self):
        """Compute checksum after initialization."""
        if not self.checksum:
            self.checksum = self._compute_checksum()
    
    def _compute_checksum(self) -> str:
        """
        Compute SHA-256 checksum of record fields.
        
        Only includes fields that should be immutable for integrity checking.
        """
        data = {
            "record_id": self.record_id,
            "sequence_number": self.sequence_number,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "tenant_id": self.tenant_id,
            "actor_type": self.actor_type,
            "actor_id": self.actor_id,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "success": self.success,
            "previous_checksum": self.previous_checksum,
        }
        
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """
        Verify that the record has not been tampered with.
        
        Returns:
            True if checksum matches computed value
        """
        return self.checksum == self._compute_checksum()
    
    def to_dict(self) -> AuditRecordDict:
        """Convert to dictionary for serialization."""
        return {
            "record_id": self.record_id,
            "sequence_number": self.sequence_number,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "tenant_id": self.tenant_id,
            "pipeline_id": self.pipeline_id,
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "step_name": self.step_name,
            "actor_type": self.actor_type,
            "actor_id": self.actor_id,
            "actor_ip": self.actor_ip,
            "actor_user_agent": self.actor_user_agent,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "tags": self.tags,
            "checksum": self.checksum,
            "previous_checksum": self.previous_checksum,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditRecord":
        """Create from dictionary."""
        return cls(
            record_id=data["record_id"],
            sequence_number=data["sequence_number"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_type=AuditEventType(data["event_type"]),
            severity=AuditSeverity(data.get("severity", "info")),
            tenant_id=data.get("tenant_id", ""),
            pipeline_id=data.get("pipeline_id", ""),
            run_id=data.get("run_id", ""),
            agent_name=data.get("agent_name", ""),
            step_name=data.get("step_name", ""),
            actor_type=data.get("actor_type", ""),
            actor_id=data.get("actor_id", ""),
            actor_ip=data.get("actor_ip", ""),
            actor_user_agent=data.get("actor_user_agent", ""),
            action=data.get("action", ""),
            resource_type=data.get("resource_type", ""),
            resource_id=data.get("resource_id", ""),
            old_value=data.get("old_value"),
            new_value=data.get("new_value"),
            success=data.get("success", True),
            error_message=data.get("error_message", ""),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            checksum=data.get("checksum", ""),
            previous_checksum=data.get("previous_checksum", ""),
        )


@dataclass
class AuditQuery:
    """
    Query filters for searching audit records.
    
    All filters are optional - only non-None values are applied.
    Multiple filters are ANDed together.
    """
    
    # Time range
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Filters
    tenant_id: Optional[str] = None
    event_types: Optional[List[AuditEventType]] = None
    severities: Optional[List[AuditSeverity]] = None
    actor_id: Optional[str] = None
    actor_type: Optional[str] = None
    pipeline_id: Optional[str] = None
    run_id: Optional[str] = None
    agent_name: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    success: Optional[bool] = None
    
    # Search
    search_text: Optional[str] = None
    tags: Optional[List[str]] = None
    
    # Pagination
    limit: int = 100
    offset: int = 0
    
    # Ordering
    order_by: str = "timestamp"
    order_desc: bool = True


@dataclass
class AuditReport:
    """
    Aggregated audit report for compliance.
    
    Contains summary statistics and optionally the raw records.
    """
    
    report_id: str
    generated_at: datetime
    generated_by: str
    
    # Period
    period_start: datetime
    period_end: datetime
    tenant_id: str
    
    # Summary statistics
    total_events: int
    events_by_type: Dict[str, int]
    events_by_severity: Dict[str, int]
    events_by_actor: Dict[str, int]
    events_by_day: Dict[str, int]
    
    # Notable events
    security_incidents: List[AuditRecord]
    failed_operations: List[AuditRecord]
    
    # Raw records (optional)
    records: List[AuditRecord] = field(default_factory=list)
    
    # Integrity
    chain_valid: bool = True
    missing_sequence_numbers: List[int] = field(default_factory=list)


# Exceptions

class AuditError(Exception):
    """Base exception for audit-related errors."""
    pass


class AuditIntegrityError(AuditError):
    """Raised when audit chain integrity is compromised."""
    
    def __init__(self, message: str, invalid_records: List[str] = None):
        self.invalid_records = invalid_records or []
        super().__init__(message)


class AuditStorageError(AuditError):
    """Raised when there's a storage error."""
    pass
