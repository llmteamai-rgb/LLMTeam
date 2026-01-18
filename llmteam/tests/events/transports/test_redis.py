"""Tests for RedisTransport."""

import pytest
import sys
import json
import datetime
from unittest.mock import MagicMock, AsyncMock, patch
from llmteam.events.models import WorktrailEvent, EventType, EventSeverity

# Create a mock Redis module
mock_redis = MagicMock()
mock_redis.asyncio = MagicMock()
# Set from_url to be an AsyncMock so it can be awaited
mock_redis.asyncio.from_url = AsyncMock()

@pytest.fixture(autouse=True)
def patch_redis():
    """Patch redis module for all tests in this file."""
    with patch.dict(sys.modules, {"redis": mock_redis, "redis.asyncio": mock_redis.asyncio}):
        yield

from llmteam.events.transports.redis import RedisTransport

@pytest.fixture
def mock_redis_client():
    client = AsyncMock()
    # Mock methods
    client.publish = AsyncMock(return_value=1)
    client.close = AsyncMock()
    
    # PubSub setup
    pubsub = AsyncMock() 
    # Important: pubsub() in redis-py is synchronous, returning a PubSub instance
    # But often people mock it as AsyncMock if they expect await.
    # In our code: self._redis.pubsub() -> sync call.
    # So client.pubsub should be a MagicMock (or AsyncMock with return_value if called sync? No)
    # Let's make client.pubsub a standard Mock that returns our pubsub mock.
    client.pubsub = MagicMock(return_value=pubsub)
    
    # PubSub methods
    pubsub.psubscribe = AsyncMock()
    pubsub.punsubscribe = AsyncMock()
    pubsub.close = AsyncMock()
    
    # Mock listen as async iterator
    async def mock_listen():
        yield {
            "type": "pmessage",
            "data": json.dumps({
                "event_id": "evt_1",
                "event_type": "step.started",
                "timestamp": datetime.datetime.now().isoformat(),
                "tenant_id": "tenant_1",
                "instance_id": "inst_1",
                "run_id": "test_run",
                "segment_id": "seg_1",
            })
        }
    pubsub.listen = mock_listen
    
    return client

@pytest.fixture
def transport(mock_redis_client):
    # We mock the connection part manually or use the fixture
    t = RedisTransport(redis_url="redis://test")
    # Inject the mock client directly to avoid connect() call in some tests
    t._redis = mock_redis_client
    return t

@pytest.mark.asyncio
async def test_connect(mock_redis_client):
    # Configure the global mock to return our client
    mock_redis.asyncio.from_url.return_value = mock_redis_client
    
    # Re-import to handle module patching if needed (but patch_redis fixture handles it)
    transport = RedisTransport(redis_url="redis://test")
    await transport.connect()
    
    mock_redis.asyncio.from_url.assert_called_once_with(
        "redis://test",
        encoding="utf-8",
        decode_responses=True
    )
    assert transport._redis == mock_redis_client

@pytest.mark.asyncio
async def test_publish(transport, mock_redis_client):
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
    
    mock_redis_client.publish.assert_called_once()
    assert mock_redis_client.publish.call_args[0][0] == "llmteam:events:test_run"

@pytest.mark.asyncio
async def test_subscribe(transport):
    # transport._redis is already set by fixture
    events = []
    async for event in transport.subscribe("test_*"):
        events.append(event)
        # Consuming all (mock yields only 1) ensures natural finish and finally execution
    
    assert len(events) == 1
    assert events[0].run_id == "test_run"
    
    # Verify subscription
    pubsub = transport._redis.pubsub.return_value
    pubsub.psubscribe.assert_called_with("llmteam:events:test_*")
    
    # Verify unsubscription (cleanup)
    # Note: explicit break might require a moment or ensuring the generator is finalized.
    # But usually 'break' in 'async for' calls aclose().
    pubsub.punsubscribe.assert_called_with("llmteam:events:test_*")
