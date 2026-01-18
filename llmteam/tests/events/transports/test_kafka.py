"""Tests for KafkaTransport."""

import pytest
import sys
import json
import datetime
from unittest.mock import MagicMock, AsyncMock, patch, ANY
from llmteam.events.models import WorktrailEvent, EventType

# Create mock aiokafka module
mock_aiokafka = MagicMock()
mock_aiokafka.AIOKafkaProducer = MagicMock()
mock_aiokafka.AIOKafkaConsumer = MagicMock()

@pytest.fixture(autouse=True)
def patch_aiokafka():
    with patch.dict(sys.modules, {"aiokafka": mock_aiokafka}):
        yield

from llmteam.events.transports.kafka import KafkaTransport

@pytest.fixture
def mock_producer():
    producer = AsyncMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer.send_and_wait = AsyncMock()
    # Mock AIOKafkaProducer constructor to return this instance
    mock_aiokafka.AIOKafkaProducer.return_value = producer
    return producer

@pytest.fixture
def mock_consumer():
    consumer = AsyncMock()
    consumer.start = AsyncMock()
    consumer.stop = AsyncMock()
    
    # Mock iteration
    async def mock_iter():
        msg = MagicMock()
        msg.value = {
            "event_id": "evt_1",
            "event_type": "step.started",
            "timestamp": datetime.datetime.now().isoformat(),
            "tenant_id": "tenant_1",
            "instance_id": "inst_1",
            "run_id": "test_run",
            "segment_id": "seg_1",
        }
        yield msg
    
    consumer.__aiter__.side_effect = mock_iter
    
    mock_aiokafka.AIOKafkaConsumer.return_value = consumer
    return consumer

@pytest.fixture
def transport(mock_producer, mock_consumer):
    return KafkaTransport(bootstrap_servers="kafka:9092")

@pytest.mark.asyncio
async def test_connect(transport, mock_producer):
    await transport.connect()
    
    mock_aiokafka.AIOKafkaProducer.assert_called_with(
        bootstrap_servers="kafka:9092",
        client_id=transport.client_id,
        value_serializer=ANY
    )
    mock_producer.start.assert_called_once()

@pytest.mark.asyncio
async def test_publish(transport, mock_producer):
    await transport.connect()
    
    event = WorktrailEvent(
        event_id="evt_1",
        run_id="test_run",
        event_type=EventType.STEP_STARTED,
        timestamp=datetime.datetime.now(),
        tenant_id="tenant_1",
        instance_id="inst_1",
        segment_id="seg_1"
    )
    
    await transport.publish(event)
    
    mock_producer.send_and_wait.assert_called_once()
    assert mock_producer.send_and_wait.call_args[0][0] == "llmteam-events"
    assert mock_producer.send_and_wait.call_args[1]["key"] == b"test_run"

@pytest.mark.asyncio
async def test_consume(transport, mock_consumer):
    events = []
    async for event in transport.consume():
        events.append(event)
        
    assert len(events) == 1
    assert events[0].run_id == "test_run"
    
    mock_aiokafka.AIOKafkaConsumer.assert_called_once()
    mock_consumer.start.assert_called_once()
