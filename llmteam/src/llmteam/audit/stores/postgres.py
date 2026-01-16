"""
PostgreSQL audit store implementation.

This store provides compliant, immutable audit logging with:
- Append-only operations (no UPDATE/DELETE)
- Partitioning support for large datasets
- Efficient querying with indexes
"""

import json
from datetime import datetime
from typing import Any, List, Optional

from llmteam.audit.models import (
    AuditRecord,
    AuditQuery,
    AuditEventType,
    AuditSeverity,
)


class PostgresAuditStore:
    """
    PostgreSQL implementation of AuditStore.
    
    Features:
    - Append-only table (compliance requirement)
    - Partitioning by month for performance
    - Full-text search support
    - Efficient indexes for common queries
    
    Required table schema: See POSTGRES_SCHEMA at the end of this file.
    
    Example:
        import asyncpg
        
        pool = await asyncpg.create_pool(dsn)
        store = PostgresAuditStore(pool)
        audit = AuditTrail(store)
    """
    
    def __init__(
        self,
        pool: Any,  # asyncpg.Pool
        table_name: str = "audit_log",
    ):
        """
        Initialize PostgresAuditStore.
        
        Args:
            pool: asyncpg connection pool
            table_name: Name of the audit table
        """
        self.pool = pool
        self.table_name = table_name
    
    async def append(self, record: AuditRecord) -> None:
        """
        Append a record (INSERT only).
        
        Args:
            record: Audit record to store
        """
        query = f"""
            INSERT INTO {self.table_name} (
                record_id, sequence_number, timestamp, event_type, severity,
                tenant_id, pipeline_id, run_id, agent_name, step_name,
                actor_type, actor_id, actor_ip, actor_user_agent,
                action, resource_type, resource_id,
                old_value, new_value, success, error_message,
                metadata, tags, checksum, previous_checksum
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9, $10,
                $11, $12, $13, $14,
                $15, $16, $17,
                $18, $19, $20, $21,
                $22, $23, $24, $25
            )
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                record.record_id,
                record.sequence_number,
                record.timestamp,
                record.event_type.value,
                record.severity.value,
                record.tenant_id,
                record.pipeline_id,
                record.run_id,
                record.agent_name,
                record.step_name,
                record.actor_type,
                record.actor_id,
                record.actor_ip,
                record.actor_user_agent,
                record.action,
                record.resource_type,
                record.resource_id,
                json.dumps(record.old_value) if record.old_value else None,
                json.dumps(record.new_value) if record.new_value else None,
                record.success,
                record.error_message,
                json.dumps(record.metadata),
                record.tags,
                record.checksum,
                record.previous_checksum,
            )
    
    async def query(self, query: AuditQuery) -> List[AuditRecord]:
        """
        Query records with filters.
        
        Args:
            query: Query filters
            
        Returns:
            List of matching AuditRecord objects
        """
        conditions = []
        params = []
        param_idx = 1
        
        # Build WHERE clause
        if query.tenant_id is not None:
            conditions.append(f"tenant_id = ${param_idx}")
            params.append(query.tenant_id)
            param_idx += 1
        
        if query.start_time is not None:
            conditions.append(f"timestamp >= ${param_idx}")
            params.append(query.start_time)
            param_idx += 1
        
        if query.end_time is not None:
            conditions.append(f"timestamp <= ${param_idx}")
            params.append(query.end_time)
            param_idx += 1
        
        if query.event_types is not None:
            placeholders = ", ".join(f"${param_idx + i}" for i in range(len(query.event_types)))
            conditions.append(f"event_type IN ({placeholders})")
            params.extend(et.value for et in query.event_types)
            param_idx += len(query.event_types)
        
        if query.severities is not None:
            placeholders = ", ".join(f"${param_idx + i}" for i in range(len(query.severities)))
            conditions.append(f"severity IN ({placeholders})")
            params.extend(s.value for s in query.severities)
            param_idx += len(query.severities)
        
        if query.actor_id is not None:
            conditions.append(f"actor_id = ${param_idx}")
            params.append(query.actor_id)
            param_idx += 1
        
        if query.actor_type is not None:
            conditions.append(f"actor_type = ${param_idx}")
            params.append(query.actor_type)
            param_idx += 1
        
        if query.pipeline_id is not None:
            conditions.append(f"pipeline_id = ${param_idx}")
            params.append(query.pipeline_id)
            param_idx += 1
        
        if query.run_id is not None:
            conditions.append(f"run_id = ${param_idx}")
            params.append(query.run_id)
            param_idx += 1
        
        if query.agent_name is not None:
            conditions.append(f"agent_name = ${param_idx}")
            params.append(query.agent_name)
            param_idx += 1
        
        if query.resource_type is not None:
            conditions.append(f"resource_type = ${param_idx}")
            params.append(query.resource_type)
            param_idx += 1
        
        if query.resource_id is not None:
            conditions.append(f"resource_id = ${param_idx}")
            params.append(query.resource_id)
            param_idx += 1
        
        if query.success is not None:
            conditions.append(f"success = ${param_idx}")
            params.append(query.success)
            param_idx += 1
        
        if query.tags is not None:
            conditions.append(f"tags && ${param_idx}")
            params.append(query.tags)
            param_idx += 1
        
        if query.search_text is not None:
            conditions.append(f"search_vector @@ plainto_tsquery('english', ${param_idx})")
            params.append(query.search_text)
            param_idx += 1
        
        # Build query
        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        order_dir = "DESC" if query.order_desc else "ASC"
        
        sql = f"""
            SELECT 
                record_id, sequence_number, timestamp, event_type, severity,
                tenant_id, pipeline_id, run_id, agent_name, step_name,
                actor_type, actor_id, actor_ip, actor_user_agent,
                action, resource_type, resource_id,
                old_value, new_value, success, error_message,
                metadata, tags, checksum, previous_checksum
            FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY {query.order_by} {order_dir}
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        
        params.extend([query.limit, query.offset])
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        
        return [self._row_to_record(row) for row in rows]
    
    async def get_last_sequence(self, tenant_id: str) -> int:
        """
        Get the last sequence number for a tenant.
        
        Args:
            tenant_id: The tenant ID
            
        Returns:
            Last sequence number or 0 if no records
        """
        query = f"""
            SELECT COALESCE(MAX(sequence_number), 0)
            FROM {self.table_name}
            WHERE tenant_id = $1
        """
        
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, tenant_id)
    
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
        query = f"""
            SELECT 
                record_id, sequence_number, timestamp, event_type, severity,
                tenant_id, pipeline_id, run_id, agent_name, step_name,
                actor_type, actor_id, actor_ip, actor_user_agent,
                action, resource_type, resource_id,
                old_value, new_value, success, error_message,
                metadata, tags, checksum, previous_checksum
            FROM {self.table_name}
            WHERE tenant_id = $1 AND sequence_number BETWEEN $2 AND $3
            ORDER BY sequence_number ASC
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, tenant_id, start, end)
        
        return [self._row_to_record(row) for row in rows]
    
    async def count(self, tenant_id: str = None) -> int:
        """
        Count records.
        
        Args:
            tenant_id: Optional tenant filter
            
        Returns:
            Number of records
        """
        if tenant_id:
            query = f"SELECT COUNT(*) FROM {self.table_name} WHERE tenant_id = $1"
            async with self.pool.acquire() as conn:
                return await conn.fetchval(query, tenant_id)
        else:
            query = f"SELECT COUNT(*) FROM {self.table_name}"
            async with self.pool.acquire() as conn:
                return await conn.fetchval(query)
    
    def _row_to_record(self, row: Any) -> AuditRecord:
        """Convert a database row to AuditRecord."""
        old_value = row["old_value"]
        if isinstance(old_value, str):
            old_value = json.loads(old_value)
        
        new_value = row["new_value"]
        if isinstance(new_value, str):
            new_value = json.loads(new_value)
        
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        
        return AuditRecord(
            record_id=row["record_id"],
            sequence_number=row["sequence_number"],
            timestamp=row["timestamp"],
            event_type=AuditEventType(row["event_type"]),
            severity=AuditSeverity(row["severity"]),
            tenant_id=row["tenant_id"],
            pipeline_id=row["pipeline_id"] or "",
            run_id=row["run_id"] or "",
            agent_name=row["agent_name"] or "",
            step_name=row["step_name"] or "",
            actor_type=row["actor_type"] or "",
            actor_id=row["actor_id"] or "",
            actor_ip=row["actor_ip"] or "",
            actor_user_agent=row["actor_user_agent"] or "",
            action=row["action"] or "",
            resource_type=row["resource_type"] or "",
            resource_id=row["resource_id"] or "",
            old_value=old_value,
            new_value=new_value,
            success=row["success"],
            error_message=row["error_message"] or "",
            metadata=metadata or {},
            tags=row["tags"] or [],
            checksum=row["checksum"],
            previous_checksum=row["previous_checksum"] or "",
        )


# SQL schema for reference
POSTGRES_SCHEMA = """
-- Audit log table (append-only)
CREATE TABLE IF NOT EXISTS audit_log (
    record_id VARCHAR(36) PRIMARY KEY,
    sequence_number BIGINT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL DEFAULT 'info',
    
    -- Context
    tenant_id VARCHAR(255) NOT NULL,
    pipeline_id VARCHAR(255),
    run_id VARCHAR(255),
    agent_name VARCHAR(255),
    step_name VARCHAR(255),
    
    -- Actor
    actor_type VARCHAR(50),
    actor_id VARCHAR(255),
    actor_ip VARCHAR(45),
    actor_user_agent TEXT,
    
    -- Details
    action VARCHAR(255),
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    
    -- State
    old_value JSONB,
    new_value JSONB,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_message TEXT,
    
    -- Metadata
    metadata JSONB NOT NULL DEFAULT '{}',
    tags TEXT[] NOT NULL DEFAULT '{}',
    
    -- Integrity
    checksum VARCHAR(64) NOT NULL,
    previous_checksum VARCHAR(64),
    
    -- Full-text search
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', COALESCE(action, '') || ' ' || 
                              COALESCE(resource_id, '') || ' ' || 
                              COALESCE(error_message, ''))
    ) STORED
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_audit_tenant_time 
    ON audit_log(tenant_id, timestamp DESC);
    
CREATE INDEX IF NOT EXISTS idx_audit_tenant_sequence 
    ON audit_log(tenant_id, sequence_number);
    
CREATE INDEX IF NOT EXISTS idx_audit_event_type 
    ON audit_log(event_type);
    
CREATE INDEX IF NOT EXISTS idx_audit_actor 
    ON audit_log(actor_id);
    
CREATE INDEX IF NOT EXISTS idx_audit_pipeline 
    ON audit_log(pipeline_id) WHERE pipeline_id IS NOT NULL;
    
CREATE INDEX IF NOT EXISTS idx_audit_run 
    ON audit_log(run_id) WHERE run_id IS NOT NULL;
    
CREATE INDEX IF NOT EXISTS idx_audit_resource 
    ON audit_log(resource_type, resource_id);
    
CREATE INDEX IF NOT EXISTS idx_audit_success 
    ON audit_log(success) WHERE success = FALSE;
    
CREATE INDEX IF NOT EXISTS idx_audit_search 
    ON audit_log USING GIN(search_vector);
    
CREATE INDEX IF NOT EXISTS idx_audit_tags 
    ON audit_log USING GIN(tags);

-- Ensure sequence uniqueness per tenant
CREATE UNIQUE INDEX IF NOT EXISTS idx_audit_tenant_sequence_unique 
    ON audit_log(tenant_id, sequence_number);

-- Partitioning by month (optional, for large datasets)
-- To enable, create table as PARTITION BY RANGE (timestamp)
-- and create monthly partitions

-- Revoke UPDATE/DELETE to enforce append-only
-- REVOKE UPDATE, DELETE ON audit_log FROM app_user;

-- Trigger to prevent updates/deletes (alternative to REVOKE)
CREATE OR REPLACE FUNCTION audit_prevent_modify()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'Audit log is append-only. Updates and deletes are not allowed.';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS audit_no_update ON audit_log;
CREATE TRIGGER audit_no_update
    BEFORE UPDATE ON audit_log
    FOR EACH ROW
    EXECUTE FUNCTION audit_prevent_modify();

DROP TRIGGER IF EXISTS audit_no_delete ON audit_log;
CREATE TRIGGER audit_no_delete
    BEFORE DELETE ON audit_log
    FOR EACH ROW
    EXECUTE FUNCTION audit_prevent_modify();
"""
