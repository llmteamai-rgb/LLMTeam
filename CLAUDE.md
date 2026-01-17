# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**llmteam** — Enterprise AI Workflow Runtime for building multi-agent LLM pipelines with security, orchestration, and workflow capabilities.

Renamed from `llm-pipeline-smtrk` in v1.7.0. Python package is in `llmteam/` subdirectory.

**Current version:** 1.9.0 (Workflow Runtime)

## Development Commands

All commands run from `llmteam/` directory.

### Setup

```bash
cd llmteam
pip install -e ".[dev]"

# Verify (bash)
PYTHONPATH=src python -c "import llmteam; print(f'v{llmteam.__version__}')"

# Verify (PowerShell)
$env:PYTHONPATH="src"; python -c "import llmteam; print(f'v{llmteam.__version__}')"
```

### Testing

**IMPORTANT:** Tests require sequential or limited parallel execution to prevent memory exhaustion.

```bash
# Recommended: use test runner
python run_tests.py                    # Sequential (safest)
python run_tests.py --parallel 2       # Limited parallelism
python run_tests.py --module tenancy   # Single module
python run_tests.py --fast             # Unit tests only
python run_tests.py --coverage         # With coverage

# Manual (bash)
PYTHONPATH=src pytest tests/tenancy/ -v

# Manual (PowerShell)
$env:PYTHONPATH="src"; pytest tests/tenancy/ -v

# Single test
PYTHONPATH=src pytest tests/tenancy/test_tenancy.py::TestTenantConfig::test_default_config -vv
```

**Avoid:** `pytest tests/ -n auto` — causes memory issues.

### Code Quality

```bash
mypy src/llmteam/          # Type checking
black src/ tests/          # Formatting
ruff check src/ tests/     # Linting
```

## Architecture

### Module Structure (by version)

**v1.7.0 — Security Foundation:**
- `tenancy/` — Multi-tenant isolation (TenantManager, TenantContext, TenantIsolatedStore)
- `audit/` — Compliance audit trail with SHA-256 checksum chain (AuditTrail, AuditRecord)
- `context/` — Secure agent context with sealed data (SecureAgentContext, SealedData)
- `ratelimit/` — Rate limiting + circuit breaker (RateLimiter, CircuitBreaker, RateLimitedExecutor)

**v1.8.0 — Orchestration Intelligence:**
- `context/hierarchical.py` — Hierarchical context propagation (HierarchicalContext, ContextManager)
- `licensing/` — License-based limits (LicenseManager, LicenseTier)
- `execution/` — Parallel pipeline execution (PipelineExecutor, ExecutorConfig)
- `roles/` — Orchestration roles (PipelineOrchestrator, GroupOrchestrator, ProcessMiningEngine)

**v1.9.0 — Workflow Runtime:**
- `actions/` — External API/webhook calls (ActionExecutor, ActionRegistry)
- `human/` — Human-in-the-loop interaction (HumanInteractionManager, approval/chat/escalation)
- `persistence/` — Snapshot-based pause/resume (SnapshotManager, PipelineSnapshot)

### Key Patterns

**Store Pattern:** All stores use dependency injection:
- Abstract base class defines interface
- `MemoryStore` for testing
- `PostgresStore` for production
- Stores in `stores/` subdirectories

**Context Manager Pattern:** Tenant-scoped operations:
```python
async with manager.context(tenant_id):
    # All operations isolated to tenant_id
    pass
```

**Immutability for Security:**
- `AuditRecord` — immutable with checksum chain
- `SealedData` — owner-only access container

### Security Principles

1. **Horizontal Isolation** — Agents NEVER see each other's contexts
2. **Vertical Visibility** — Orchestrators see only their child agents (parent→child only)
3. **Sealed Data** — Only the owner agent can access sealed fields
4. **Tenant Isolation** — Complete data separation between tenants

## Creating New Modules

1. Create module directory with `__init__.py` containing exports
2. Add imports to parent `llmteam/__init__.py`
3. Create tests in `tests/{module}/test_{module}.py`
4. Add module to `TEST_MODULES` list in `run_tests.py`

### Async Code

- Use `asyncio.Lock()` for thread-safety
- Tests use `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` decorator needed)
- All async methods must consistently use `async`/`await`

## Integration Examples

```python
from llmteam.tenancy import current_tenant, TenantContext, TenantManager
from llmteam.audit import AuditTrail, AuditEventType
from llmteam.context import SecureAgentContext, ContextAccessPolicy, HierarchicalContext
from llmteam.ratelimit import RateLimitedExecutor
from llmteam.roles import PipelineOrchestrator, GroupOrchestrator
from llmteam.actions import ActionExecutor
from llmteam.human import HumanInteractionManager
from llmteam.persistence import SnapshotManager
```

## Reference Documentation

- `v170-security-foundation.md` — v1.7.0 specification
- `v180-orchestration-intelligence.md` — v1.8.0 specification
- `v190-workflow-runtime.md` — v1.9.0 specification
- `llmteam-v170-implementation-summary.md` — v1.7.0 implementation notes
- `llmteam-v190-implementation-summary.md` — v1.9.0 implementation notes
