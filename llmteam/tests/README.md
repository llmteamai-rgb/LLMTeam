# llmteam Test Suite

## Quick Start

```bash
cd llmteam

# Install dependencies (first time only)
pip install -e ".[dev]"

# Run all tests (memory-safe)
python run_tests.py
```

## Problem Solved

This test suite prevents out-of-memory (OOM) errors that occur when running many tests in parallel.

## Features

- **Automatic cleanup** after each test (no memory leaks)
- **Sequential execution** by default (safest)
- **Controlled parallelism** when needed
- **Test isolation** via fixtures
- **Timeout protection** (30s per test)

## Test Organization

```
tests/
├── conftest.py           # Global fixtures with auto-cleanup
├── pytest.ini            # Safety configuration
├── tenancy/              # Multi-tenant isolation tests
├── audit/                # Audit trail tests
├── context/              # Context security + hierarchical tests
├── ratelimit/            # Rate limiting tests
├── licensing/            # License management tests
├── execution/            # Pipeline executor tests
└── roles/                # Orchestration + process mining tests
```

## Common Commands

```bash
# Safe sequential run
python run_tests.py

# Specific module
python run_tests.py --module tenancy

# With coverage report
python run_tests.py --coverage

# Parallel (2 workers max recommended)
python run_tests.py --parallel 2

# Fast tests only
python run_tests.py --fast
```

## Writing Tests

Use fixtures from `conftest.py`:

```python
import pytest

def test_with_manager(tenant_manager):
    """Use shared fixture with auto-cleanup."""
    # tenant_manager is automatically cleaned up
    await tenant_manager.create_tenant(...)

@pytest.mark.asyncio
async def test_async(audit_trail):
    """Async test with auto-cleanup."""
    # audit_trail is automatically cleaned up
    await audit_trail.log(...)

@pytest.mark.unit
def test_unit():
    """Mark fast unit tests."""
    assert True

@pytest.mark.memory_intensive
def test_memory():
    """Mark memory-intensive tests."""
    # These can be skipped on low-memory systems
    pass
```

## Markers

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow tests
- `@pytest.mark.memory_intensive` - Memory-heavy tests

Run specific markers:

```bash
# Only unit tests
PYTHONPATH=src pytest -m unit

# Exclude slow tests
PYTHONPATH=src pytest -m "not slow"
```

## Memory-Safe Fixtures

All stores and managers are created via fixtures:

- `memory_tenant_store` - Fresh MemoryTenantStore
- `memory_audit_store` - Fresh MemoryAuditStore
- `memory_kv_store` - Fresh MemoryKeyValueStore
- `tenant_manager` - TenantManager with cleanup
- `audit_trail` - AuditTrail with cleanup
- `process_mining_engine` - ProcessMiningEngine with cleanup

These fixtures automatically:
1. Create fresh instances
2. Register for cleanup
3. Clear data after test
4. Force garbage collection

## Troubleshooting

### Still getting OOM errors?

```bash
# Run modules one at a time
python run_tests.py --module tenancy
python run_tests.py --module audit
python run_tests.py --module context
```

### Tests hanging?

Each test has 30s timeout. Check:
```bash
# Run with verbose output
PYTHONPATH=src pytest tests/tenancy/ -vv -s
```

### Need to debug one test?

```bash
# Run specific test
PYTHONPATH=src pytest tests/tenancy/test_tenancy.py::TestTenantConfig::test_default_config -vv
```

## Coverage Reports

```bash
# Generate HTML coverage
python run_tests.py --coverage

# View report
# Open: htmlcov/index.html
```

## CI/CD

For continuous integration:

```yaml
- name: Run tests
  run: |
    cd llmteam
    pip install -e ".[dev]"
    python run_tests.py --coverage
```

## More Info

See `TESTING.md` in the repository root for detailed documentation.
