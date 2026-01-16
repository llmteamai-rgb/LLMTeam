"""
Shared pytest configuration and fixtures for all tests.

This module provides:
- Centralized cleanup to prevent memory leaks
- Shared fixtures for common test objects
- Test isolation and resource management
"""

import gc
import pytest
from typing import AsyncGenerator, Generator

# Store references to all created stores for cleanup
_test_stores: list = []
_test_managers: list = []
_test_engines: list = []


def _register_for_cleanup(obj: object) -> None:
    """Register an object for cleanup after test."""
    if hasattr(obj, 'close') or hasattr(obj, 'cleanup'):
        if obj not in _test_stores:
            _test_stores.append(obj)


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatically cleanup after each test to prevent memory leaks."""
    # Setup (before test)
    yield

    # Teardown (after test)
    global _test_stores, _test_managers, _test_engines

    # Clear all registered objects
    _test_stores.clear()
    _test_managers.clear()
    _test_engines.clear()

    # Force garbage collection
    gc.collect()


@pytest.fixture(autouse=True)
def cleanup_async_resources():
    """Cleanup async resources after each test."""
    # Setup (before test)
    yield

    # Teardown (after test)
    # Just trigger garbage collection for async resources
    # Don't wait for tasks as it can cause hangs
    gc.collect()


@pytest.fixture
def memory_tenant_store():
    """Create a fresh MemoryTenantStore for testing."""
    from llmteam.tenancy.stores import MemoryTenantStore

    store = MemoryTenantStore()
    _register_for_cleanup(store)
    return store


@pytest.fixture
def memory_audit_store():
    """Create a fresh MemoryAuditStore for testing."""
    from llmteam.audit.stores import MemoryAuditStore

    store = MemoryAuditStore()
    _register_for_cleanup(store)
    return store


@pytest.fixture
def memory_kv_store():
    """Create a fresh MemoryKeyValueStore for testing."""
    from llmteam.tenancy.stores import MemoryKeyValueStore

    store = MemoryKeyValueStore()
    _register_for_cleanup(store)
    return store


@pytest.fixture
async def tenant_manager(memory_tenant_store):
    """Create a TenantManager with fresh store."""
    from llmteam.tenancy import TenantManager

    manager = TenantManager(memory_tenant_store)
    _register_for_cleanup(manager)
    return manager


@pytest.fixture
async def audit_trail(memory_audit_store):
    """Create an AuditTrail with fresh store."""
    from llmteam.audit import AuditTrail

    trail = AuditTrail(memory_audit_store, tenant_id="test")
    _register_for_cleanup(trail)
    return trail


@pytest.fixture
def process_mining_engine():
    """Create a fresh ProcessMiningEngine."""
    from llmteam.roles import ProcessMiningEngine

    engine = ProcessMiningEngine()
    _register_for_cleanup(engine)
    return engine


# Test markers for grouping tests
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (slower, may use external services)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests that take significant time"
    )
    config.addinivalue_line(
        "markers", "memory_intensive: Tests that use significant memory"
    )
