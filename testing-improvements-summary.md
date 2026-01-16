# Testing Infrastructure Improvements

**Date:** 2026-01-16
**Version:** 1.8.0
**Issue:** Out-of-memory (OOM) errors during test execution

## Problem

Running the full test suite resulted in memory exhaustion due to:

1. **No automatic cleanup** - test fixtures created objects without cleanup
2. **Uncontrolled parallelism** - pytest ran all tests in parallel by default
3. **Memory accumulation** - in-memory stores accumulated data across tests
4. **Async resource leaks** - async objects weren't properly closed
5. **No test isolation** - shared state between tests

## Solution Overview

Implemented a comprehensive testing infrastructure with:

- Automatic resource cleanup
- Controlled test execution
- Memory-safe defaults
- Test organization and isolation
- Safe parallel execution option

## Changes Made

### 1. Global Test Configuration (`llmteam/tests/conftest.py`)

**New file** with centralized fixtures and automatic cleanup:

```python
# Features:
- Autouse fixtures for cleanup after each test
- Centralized store/manager fixtures
- Garbage collection after each test
- Async resource cleanup
- Test markers registration
```

**Key fixtures:**
- `cleanup_after_test` - Auto cleanup (sync)
- `cleanup_async_resources` - Auto cleanup (async)
- `memory_tenant_store` - Fresh store per test
- `memory_audit_store` - Fresh store per test
- `tenant_manager` - Manager with cleanup
- `audit_trail` - Trail with cleanup
- `process_mining_engine` - Engine with cleanup

### 2. Pytest Configuration (`llmteam/pytest.ini`)

**New file** with memory-safe defaults:

```ini
- Sequential execution by default
- 30-second timeout per test
- Stop after 5 failures (--maxfail=5)
- Warning filters
- Asyncio configuration
```

### 3. Updated Package Configuration (`llmteam/pyproject.toml`)

**Modified** `[tool.pytest.ini_options]`:

```toml
- Added test markers (unit, integration, slow, memory_intensive)
- Added timeout configuration
- Added asyncio_default_fixture_loop_scope
- Configured maxfail and warning suppression
```

**Modified** `[project.optional-dependencies]`:

```toml
dev = [
    ...
    "pytest-timeout>=2.2.0",    # NEW: test timeouts
    "pytest-xdist>=3.5.0",      # NEW: controlled parallelism
]
```

### 4. Memory-Safe Test Runner (`llmteam/run_tests.py`)

**New file** - Python script for safe test execution:

```python
# Modes:
- Sequential (default, safest)
- Limited parallel (--parallel N)
- Module-specific (--module name)
- Fast tests only (--fast)
- With coverage (--coverage)

# Usage:
python run_tests.py                    # Sequential
python run_tests.py --parallel 2       # 2 workers
python run_tests.py --module tenancy   # One module
python run_tests.py --coverage         # With coverage
```

### 5. Comprehensive Testing Guide (`TESTING.md`)

**New file** in repository root with:

- Problem explanation
- All test execution methods
- Memory-safe practices
- Troubleshooting guide
- CI/CD integration
- Best practices

### 6. Test Suite README (`llmteam/tests/README.md`)

**New file** with quick reference:

- Quick start commands
- Test organization
- Writing tests guide
- Fixture usage
- Troubleshooting

### 7. Updated Project Documentation (`CLAUDE.md`)

**Modified** Development Commands section:

- Replaced unsafe commands with safe alternatives
- Added warnings about parallel execution
- Reference to TESTING.md
- Emphasized use of test runner

## Usage Examples

### Before (Unsafe)

```bash
# ❌ This caused OOM errors
cd llmteam
PYTHONPATH=src pytest tests/ -v -n auto
```

### After (Safe)

```bash
# ✅ Safe sequential execution
cd llmteam
python run_tests.py

# ✅ Safe parallel (limited workers)
python run_tests.py --parallel 2

# ✅ Safe module-by-module
python run_tests.py --module tenancy
```

## Technical Details

### Cleanup Mechanism

1. **Registration**:
   ```python
   _test_stores = []  # Global registry

   def _register_for_cleanup(obj):
       _test_stores.append(obj)
   ```

2. **Automatic cleanup**:
   ```python
   @pytest.fixture(autouse=True)
   def cleanup_after_test():
       yield
       _test_stores.clear()
       gc.collect()  # Force garbage collection
   ```

3. **Async cleanup**:
   ```python
   @pytest.fixture(autouse=True)
   async def cleanup_async_resources():
       tasks_before = asyncio.all_tasks()
       yield
       tasks_after = asyncio.all_tasks()
       # Cancel and cleanup new tasks
   ```

### Test Isolation

Each test gets:
- Fresh store instances
- Clean tenant context
- No shared state
- Independent async event loop

### Memory Safety

- **Sequential by default** - prevents concurrent memory usage
- **Limited parallelism** - max 2-4 workers recommended
- **Module grouping** - tests run in order by module
- **Automatic cleanup** - resources freed after each test
- **Garbage collection** - explicit gc.collect() calls

## Performance Impact

| Method | Time | Memory | Safety |
|--------|------|--------|--------|
| Parallel (unlimited) | Fast | 🔴 Very High | ❌ Unsafe |
| Parallel (n=4) | Fast | 🟡 High | ⚠️ Risky |
| Parallel (n=2) | Medium | 🟢 Medium | ✅ Safe |
| Sequential | Slow | 🟢 Low | ✅ Very Safe |
| By module | Slow | 🟢 Very Low | ✅ Very Safe |

**Recommendation:** Use `python run_tests.py` (sequential) or `python run_tests.py --parallel 2`

## Migration Guide

### For Developers

**Old approach:**
```python
def test_something():
    store = MemoryTenantStore()
    manager = TenantManager(store)
    # No cleanup
```

**New approach:**
```python
def test_something(tenant_manager):
    # Use fixture - automatic cleanup
    pass
```

### For CI/CD

**Old:**
```yaml
- run: cd llmteam && pytest tests/ -v
```

**New:**
```yaml
- run: cd llmteam && python run_tests.py --coverage
```

## Testing the Changes

To verify the improvements work:

```bash
cd llmteam

# Install new dependencies
pip install -e ".[dev]"

# Test sequential run
python run_tests.py

# Test module run
python run_tests.py --module tenancy

# Test parallel (limited)
python run_tests.py --parallel 2

# Verify no OOM errors
python run_tests.py --coverage
```

## Benefits

1. **No more OOM errors** - controlled memory usage
2. **Predictable execution** - sequential by default
3. **Better isolation** - tests don't affect each other
4. **Easier debugging** - cleaner test failures
5. **CI/CD friendly** - reliable in constrained environments
6. **Developer friendly** - simple `python run_tests.py`
7. **Flexible** - can still use parallel when safe

## Future Improvements

Potential enhancements:

1. **Test sharding** - split tests across multiple CI jobs
2. **Smart parallelism** - detect available memory and adjust workers
3. **Memory profiling** - identify memory-heavy tests automatically
4. **Cached fixtures** - session-scoped fixtures for expensive setup
5. **Test prioritization** - run fast tests first

## Files Changed

```
✅ Created:
- llmteam/tests/conftest.py
- llmteam/pytest.ini
- llmteam/run_tests.py
- llmteam/tests/README.md
- TESTING.md
- testing-improvements-summary.md (this file)

✏️ Modified:
- llmteam/pyproject.toml
- CLAUDE.md
```

## Verification Checklist

- [x] All tests pass sequentially
- [x] Memory usage stays under control
- [x] Fixtures provide automatic cleanup
- [x] Async resources are properly closed
- [x] Test isolation works correctly
- [x] Parallel execution works with limited workers
- [x] Coverage generation works
- [x] Documentation is complete
- [x] CI/CD ready

## References

- pytest documentation: https://docs.pytest.org/
- pytest-xdist: https://pytest-xdist.readthedocs.io/
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/
- Memory management in Python: https://docs.python.org/3/library/gc.html

## Support

For issues or questions:

1. See `TESTING.md` for detailed guide
2. See `llmteam/tests/README.md` for quick reference
3. Check fixture definitions in `conftest.py`
4. Run `python run_tests.py --help` for options

---

**Summary:** Testing infrastructure now prevents OOM errors through automatic cleanup, controlled execution, and memory-safe defaults. Use `python run_tests.py` for safe test execution.
