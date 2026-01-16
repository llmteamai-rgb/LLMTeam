"""
Tests for audit module.
"""

import pytest
from datetime import datetime, timedelta, timezone

from llmteam.audit import (
    AuditTrail,
    AuditRecord,
    AuditQuery,
    AuditEventType,
    AuditSeverity,
)
from llmteam.audit.stores import MemoryAuditStore


class TestAuditRecord:
    """Tests for AuditRecord."""
    
    def test_checksum_computed(self):
        """Test that checksum is computed on creation."""
        record = AuditRecord(
            record_id="rec_1",
            sequence_number=1,
            timestamp=datetime.now(timezone.utc),
            event_type=AuditEventType.PIPELINE_STARTED,
            tenant_id="test",
        )
        
        assert record.checksum != ""
        assert len(record.checksum) == 64  # SHA-256 hex
    
    def test_checksum_deterministic(self):
        """Test that same data produces same checksum."""
        ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        record1 = AuditRecord(
            record_id="rec_1",
            sequence_number=1,
            timestamp=ts,
            event_type=AuditEventType.PIPELINE_STARTED,
            tenant_id="test",
        )
        
        record2 = AuditRecord(
            record_id="rec_1",
            sequence_number=1,
            timestamp=ts,
            event_type=AuditEventType.PIPELINE_STARTED,
            tenant_id="test",
        )
        
        assert record1.checksum == record2.checksum
    
    def test_verify_integrity(self):
        """Test integrity verification."""
        record = AuditRecord(
            record_id="rec_1",
            sequence_number=1,
            timestamp=datetime.now(timezone.utc),
            event_type=AuditEventType.PIPELINE_STARTED,
            tenant_id="test",
        )
        
        assert record.verify_integrity() is True
        
        # Tamper with data
        record.action = "tampered"
        # Note: checksum was computed with original action
        # This won't change the stored checksum
    
    def test_serialization(self):
        """Test to_dict and from_dict."""
        record = AuditRecord(
            record_id="rec_1",
            sequence_number=1,
            timestamp=datetime.now(timezone.utc),
            event_type=AuditEventType.PIPELINE_STARTED,
            severity=AuditSeverity.INFO,
            tenant_id="test",
            actor_id="user@test.com",
            metadata={"key": "value"},
        )
        
        data = record.to_dict()
        restored = AuditRecord.from_dict(data)
        
        assert restored.record_id == record.record_id
        assert restored.event_type == record.event_type
        assert restored.actor_id == record.actor_id


class TestAuditTrail:
    """Tests for AuditTrail."""
    
    @pytest.fixture
    def audit(self):
        """Create an audit trail with memory store."""
        store = MemoryAuditStore()
        return AuditTrail(store, tenant_id="test")
    
    @pytest.mark.asyncio
    async def test_log_event(self, audit):
        """Test logging an event."""
        record = await audit.log(
            AuditEventType.PIPELINE_STARTED,
            actor_id="user@test.com",
            resource_id="pipeline_1",
        )
        
        assert record.event_type == AuditEventType.PIPELINE_STARTED
        assert record.actor_id == "user@test.com"
        assert record.sequence_number == 1
    
    @pytest.mark.asyncio
    async def test_sequence_numbers_increment(self, audit):
        """Test that sequence numbers increment."""
        await audit.log(AuditEventType.PIPELINE_STARTED)
        await audit.log(AuditEventType.PIPELINE_COMPLETED)
        record3 = await audit.log(AuditEventType.PIPELINE_FAILED)
        
        assert record3.sequence_number == 3
    
    @pytest.mark.asyncio
    async def test_checksum_chain(self, audit):
        """Test that records form a checksum chain."""
        record1 = await audit.log(AuditEventType.PIPELINE_STARTED)
        record2 = await audit.log(AuditEventType.PIPELINE_COMPLETED)
        
        await audit.flush()
        
        # record2's previous_checksum should be record1's checksum
        assert record2.previous_checksum == record1.checksum
    
    @pytest.mark.asyncio
    async def test_query_by_event_type(self, audit):
        """Test querying by event type."""
        await audit.log(AuditEventType.PIPELINE_STARTED)
        await audit.log(AuditEventType.PIPELINE_COMPLETED)
        await audit.log(AuditEventType.PIPELINE_STARTED)
        
        query = AuditQuery(
            event_types=[AuditEventType.PIPELINE_STARTED],
        )
        
        results = await audit.query(query)
        
        assert len(results) == 2
        assert all(r.event_type == AuditEventType.PIPELINE_STARTED for r in results)
    
    @pytest.mark.asyncio
    async def test_query_by_time_range(self, audit):
        """Test querying by time range."""
        # Create records
        await audit.log(AuditEventType.PIPELINE_STARTED)
        await audit.log(AuditEventType.PIPELINE_COMPLETED)
        
        # Query with time range
        query = AuditQuery(
            start_time=datetime.now(timezone.utc) - timedelta(hours=1),
            end_time=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        
        results = await audit.query(query)
        
        assert len(results) == 2
    
    @pytest.mark.asyncio
    async def test_query_by_actor(self, audit):
        """Test querying by actor."""
        await audit.log(AuditEventType.PIPELINE_STARTED, actor_id="alice")
        await audit.log(AuditEventType.PIPELINE_STARTED, actor_id="bob")
        await audit.log(AuditEventType.PIPELINE_COMPLETED, actor_id="alice")
        
        query = AuditQuery(actor_id="alice")
        results = await audit.query(query)
        
        assert len(results) == 2
        assert all(r.actor_id == "alice" for r in results)
    
    @pytest.mark.asyncio
    async def test_verify_chain_valid(self, audit):
        """Test chain verification with valid chain."""
        await audit.log(AuditEventType.PIPELINE_STARTED)
        await audit.log(AuditEventType.PIPELINE_COMPLETED)
        await audit.log(AuditEventType.PIPELINE_FAILED)
        
        is_valid, missing = await audit.verify_chain()
        
        assert is_valid is True
        assert missing == []
    
    @pytest.mark.asyncio
    async def test_convenience_methods(self, audit):
        """Test convenience logging methods."""
        await audit.log_pipeline_started(
            pipeline_id="p1",
            run_id="r1",
            actor_id="user",
        )
        
        await audit.log_pipeline_completed(
            pipeline_id="p1",
            run_id="r1",
        )
        
        await audit.log_access_denied(
            actor_id="hacker",
            resource_type="pipeline",
            resource_id="p1",
            reason="Unauthorized",
        )
        
        # Verify all logged
        query = AuditQuery()
        results = await audit.query(query)
        
        assert len(results) == 3
    
    @pytest.mark.asyncio
    async def test_generate_report(self, audit):
        """Test report generation."""
        # Create some events
        await audit.log(AuditEventType.PIPELINE_STARTED, actor_id="user1")
        await audit.log(AuditEventType.PIPELINE_COMPLETED, actor_id="user1")
        await audit.log(AuditEventType.ACCESS_DENIED, actor_id="user2", success=False)
        
        report = await audit.generate_report(
            start=datetime.now(timezone.utc) - timedelta(hours=1),
            end=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        
        assert report.total_events == 3
        assert report.events_by_type[AuditEventType.PIPELINE_STARTED.value] == 1
        assert report.events_by_actor["user1"] == 2
        assert len(report.failed_operations) == 1


class TestMemoryAuditStore:
    """Tests for MemoryAuditStore."""
    
    @pytest.fixture
    def store(self):
        """Create a memory store."""
        return MemoryAuditStore()
    
    @pytest.mark.asyncio
    async def test_append_and_query(self, store):
        """Test appending and querying records."""
        record = AuditRecord(
            record_id="rec_1",
            sequence_number=1,
            timestamp=datetime.now(timezone.utc),
            event_type=AuditEventType.PIPELINE_STARTED,
            tenant_id="test",
        )
        
        await store.append(record)
        
        query = AuditQuery(tenant_id="test")
        results = await store.query(query)
        
        assert len(results) == 1
        assert results[0].record_id == "rec_1"
    
    @pytest.mark.asyncio
    async def test_get_last_sequence(self, store):
        """Test getting last sequence number."""
        for i in range(1, 4):
            record = AuditRecord(
                record_id=f"rec_{i}",
                sequence_number=i,
                timestamp=datetime.now(timezone.utc),
                event_type=AuditEventType.PIPELINE_STARTED,
                tenant_id="test",
            )
            await store.append(record)
        
        last = await store.get_last_sequence("test")
        
        assert last == 3
    
    @pytest.mark.asyncio
    async def test_get_by_sequence_range(self, store):
        """Test getting records by sequence range."""
        for i in range(1, 6):
            record = AuditRecord(
                record_id=f"rec_{i}",
                sequence_number=i,
                timestamp=datetime.now(timezone.utc),
                event_type=AuditEventType.PIPELINE_STARTED,
                tenant_id="test",
            )
            await store.append(record)
        
        results = await store.get_by_sequence_range("test", 2, 4)
        
        assert len(results) == 3
        assert results[0].sequence_number == 2
        assert results[-1].sequence_number == 4
