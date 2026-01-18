"""Tests for HealthChecker."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from llmteam.api.health import HealthChecker, HealthStatus


@pytest.fixture
def checker():
    return HealthChecker()


@pytest.mark.asyncio
async def test_health_check_basic(checker):
    status = await checker.check()
    
    assert status.status == "healthy"
    assert status.version is not None
    assert status.uptime_seconds >= 0
    assert status.checks == {}


@pytest.mark.asyncio
async def test_register_sync_check_success(checker):
    checker.register_check("db", lambda: True)
    
    status = await checker.check()
    
    assert status.status == "healthy"
    assert status.checks["db"] is True


@pytest.mark.asyncio
async def test_register_sync_check_failure(checker):
    checker.register_check("db", lambda: False)
    
    status = await checker.check()
    
    assert status.status == "unhealthy"
    assert status.checks["db"] is False


@pytest.mark.asyncio
async def test_register_async_check_success(checker):
    async def check_redis():
        return True
        
    checker.register_check("redis", check_redis)
    
    status = await checker.check()
    
    assert status.status == "healthy"
    assert status.checks["redis"] is True


@pytest.mark.asyncio
async def test_register_async_check_failure(checker):
    async def check_redis():
        return False
        
    checker.register_check("redis", check_redis)
    
    status = await checker.check()
    
    assert status.status == "unhealthy"
    assert status.checks["redis"] is False


@pytest.mark.asyncio
async def test_check_exception_handling(checker):
    def check_broken():
        raise RuntimeError("Oops")
        
    checker.register_check("broken", check_broken)
    
    status = await checker.check()
    
    assert status.status == "unhealthy"
    assert status.checks["broken"] is False


def test_status_to_dict(checker):
    status = HealthStatus(
        status="healthy",
        version="1.0.0",
        uptime_seconds=100.0,
        checks={"db": True},
        timestamp="2024-01-01"
    )
    
    data = status.to_dict()
    assert data["status"] == "healthy"
    assert data["checks"] == {"db": True}
