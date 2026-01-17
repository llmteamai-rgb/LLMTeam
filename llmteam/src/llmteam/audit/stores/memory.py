"""
In-memory audit store implementation.

This store is useful for testing. Data is lost when the process exits.
"""

from collections import deque
from datetime import datetime
from typing import Deque, Dict, List, Optional

from llmteam.audit.models import AuditRecord, AuditQuery


class MemoryAuditStore:
    """
    In-memory implementation of AuditStore.

    Stores audit records in a bounded deque. Useful for:
    - Unit testing
    - Development and prototyping
    - Temporary audit needs

    Note: This store does NOT provide the immutability guarantees
    required for compliance. Use PostgresAuditStore for production.

    Args:
        max_records: Maximum number of records to keep (default: 100,000).
                    Oldest records are evicted when limit is reached.

    Example:
        store = MemoryAuditStore(max_records=10_000)
        audit = AuditTrail(store)

        await audit.log(AuditEventType.PIPELINE_STARTED, ...)
    """

    DEFAULT_MAX_RECORDS = 100_000

    def __init__(self, max_records: int = DEFAULT_MAX_RECORDS):
        """Initialize empty store with bounded capacity."""
        self._max_records = max_records
        self._records: Deque[AuditRecord] = deque(maxlen=max_records)
        self._sequence_by_tenant: Dict[str, int] = {}
    
    async def append(self, record: AuditRecord) -> None:
        """
        Append a record.
        
        Args:
            record: Audit record to store
        """
        self._records.append(record)
        
        # Update sequence tracker
        tenant_id = record.tenant_id
        current_seq = self._sequence_by_tenant.get(tenant_id, 0)
        if record.sequence_number > current_seq:
            self._sequence_by_tenant[tenant_id] = record.sequence_number
    
    async def query(self, query: AuditQuery) -> List[AuditRecord]:
        """
        Query records with filters.
        
        Args:
            query: Query filters
            
        Returns:
            List of matching AuditRecord objects
        """
        results = []
        
        for record in self._records:
            if not self._matches_query(record, query):
                continue
            results.append(record)
        
        # Sort
        reverse = query.order_desc
        if query.order_by == "timestamp":
            results.sort(key=lambda r: r.timestamp, reverse=reverse)
        elif query.order_by == "sequence_number":
            results.sort(key=lambda r: r.sequence_number, reverse=reverse)
        
        # Paginate
        start = query.offset
        end = start + query.limit
        
        return results[start:end]
    
    def _matches_query(self, record: AuditRecord, query: AuditQuery) -> bool:
        """Check if a record matches query filters."""
        # Tenant filter
        if query.tenant_id is not None and record.tenant_id != query.tenant_id:
            return False
        
        # Time range
        if query.start_time is not None and record.timestamp < query.start_time:
            return False
        if query.end_time is not None and record.timestamp > query.end_time:
            return False
        
        # Event types
        if query.event_types is not None and record.event_type not in query.event_types:
            return False
        
        # Severities
        if query.severities is not None and record.severity not in query.severities:
            return False
        
        # Actor
        if query.actor_id is not None and record.actor_id != query.actor_id:
            return False
        if query.actor_type is not None and record.actor_type != query.actor_type:
            return False
        
        # Pipeline/Run
        if query.pipeline_id is not None and record.pipeline_id != query.pipeline_id:
            return False
        if query.run_id is not None and record.run_id != query.run_id:
            return False
        
        # Agent
        if query.agent_name is not None and record.agent_name != query.agent_name:
            return False
        
        # Resource
        if query.resource_type is not None and record.resource_type != query.resource_type:
            return False
        if query.resource_id is not None and record.resource_id != query.resource_id:
            return False
        
        # Success
        if query.success is not None and record.success != query.success:
            return False
        
        # Tags
        if query.tags is not None:
            if not any(tag in record.tags for tag in query.tags):
                return False
        
        # Text search (simple substring match)
        if query.search_text is not None:
            search_lower = query.search_text.lower()
            searchable = f"{record.action} {record.resource_id} {record.error_message}".lower()
            if search_lower not in searchable:
                return False
        
        return True
    
    async def get_last_sequence(self, tenant_id: str) -> int:
        """
        Get the last sequence number for a tenant.
        
        Args:
            tenant_id: The tenant ID
            
        Returns:
            Last sequence number or 0 if no records
        """
        return self._sequence_by_tenant.get(tenant_id, 0)
    
    async def get_by_sequence_range(
        self,
        tenant_id: str,
        start: int,
        end: int,
    ) -> List[AuditRecord]:
        """
        Get records by sequence number range.
        
        Args:
            tenant_id: The tenant ID
            start: Starting sequence number (inclusive)
            end: Ending sequence number (inclusive)
            
        Returns:
            List of AuditRecord objects in the range
        """
        results = []
        
        for record in self._records:
            if record.tenant_id != tenant_id:
                continue
            if start <= record.sequence_number <= end:
                results.append(record)
        
        results.sort(key=lambda r: r.sequence_number)
        return results
    
    async def count(self, tenant_id: str = None) -> int:
        """
        Count records.
        
        Args:
            tenant_id: Optional tenant filter
            
        Returns:
            Number of records
        """
        if tenant_id is None:
            return len(self._records)
        return sum(1 for r in self._records if r.tenant_id == tenant_id)
    
    def clear(self) -> None:
        """Clear all records (useful for testing)."""
        self._records.clear()
        self._sequence_by_tenant.clear()
    
    def get_all(self) -> List[AuditRecord]:
        """Get all records (useful for testing)."""
        return list(self._records)
