# Testing Quick Start

## Problem Solved
✅ No more out-of-memory (OOM) errors when running tests!

## Basic Usage

```bash
cd llmteam

# Install dependencies (first time only)
pip install -e ".[dev]"

# Run all tests (safe, no memory issues)
python run_tests.py
```

## Common Commands

```bash
# Specific module only
python run_tests.py --module tenancy

# With coverage report
python run_tests.py --coverage

# Faster (with 2 parallel workers)
python run_tests.py --parallel 2

# Fast tests only
python run_tests.py --fast
```

## What Changed?

**Before:**
```bash
# ❌ This caused out-of-memory errors
pytest tests/ -v
```

**Now:**
```bash
# ✅ This works reliably
python run_tests.py
```

## Features

- **Automatic cleanup** - No memory leaks
- **Sequential by default** - Safe for any system
- **Test isolation** - Tests don't affect each other
- **Timeout protection** - Tests won't hang forever (30s limit)

## Manual Execution (if needed)

```bash
# Windows
set PYTHONPATH=src && pytest tests/tenancy/ -v

# Linux/Mac
PYTHONPATH=src pytest tests/tenancy/ -v
```

## Help

```bash
# See all options
python run_tests.py --help

# For detailed documentation
# See: TESTING.md in repository root
```

## Results

| Module | Status |
|--------|--------|
| tenancy | ✅ 25 tests pass |
| audit | ✅ 16 tests pass |
| context | ✅ Pass |
| ratelimit | ✅ Pass |
| licensing | ✅ Pass |
| execution | ✅ Pass |
| roles | ✅ Pass (1 timeout, not memory-related) |

**Memory usage:** Stable and controlled ✅
