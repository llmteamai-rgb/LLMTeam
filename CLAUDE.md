# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**llmteam** — Enterprise AI Workflow Runtime for building multi-agent LLM pipelines with security, orchestration, and workflow capabilities.

Renamed from `llm-pipeline-smtrk` in v1.7.0. Python package is in `llmteam/` subdirectory.

**Current version:** 2.0.0 (Canvas Integration)

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
python run_tests.py --module canvas    # Canvas module
python run_tests.py --fast             # Unit tests only
python run_tests.py --coverage         # With coverage

# Manual (bash)
PYTHONPATH=src pytest tests/tenancy/ -v

# Manual (PowerShell)
$env:PYTHONPATH="src"; pytest tests/tenancy/ -v

# Single test
PYTHONPATH=src pytest tests/canvas/test_runner.py::TestSegmentRunner::test_simple_run -vv
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

**v2.0.0 — Canvas Integration:**
- `runtime/` — Runtime context injection (RuntimeContext, StepContext, registries for Store/Client/LLM/Secrets)
- `events/` — Worktrail events for UI (EventEmitter, WorktrailEvent, EventStore)
- `canvas/` — Canvas segment execution:
  - `models.py` — Segment JSON contract (SegmentDefinition, StepDefinition, EdgeDefinition, PortDefinition)
  - `catalog.py` — Step type catalog (StepCatalog, StepTypeMetadata, 7 built-in types)
  - `runner.py` — Segment execution engine (SegmentRunner, SegmentResult, RunConfig)
  - `handlers.py` — Built-in step handlers (HumanTaskHandler)
  - `exceptions.py` — Canvas-specific exceptions

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

**RuntimeContext Pattern (v2.0.0):** Step execution context:
```python
from llmteam.runtime import RuntimeContext, RuntimeContextManager

manager = RuntimeContextManager()
manager.register_store("redis", redis_store)
manager.register_client("http", http_client)

runtime = manager.create_runtime(tenant_id, instance_id)
step_ctx = runtime.child_context("step_1")

# Access resources in step
store = step_ctx.get_store("redis")
secret = step_ctx.get_secret("api_key")
```

**Immutability for Security:**
- `AuditRecord` — immutable with checksum chain
- `SealedData` — owner-only access container

### Security Principles

1. **Horizontal Isolation** — Agents NEVER see each other's contexts
2. **Vertical Visibility** — Orchestrators see only their child agents (parent→child only)
3. **Sealed Data** — Only the owner agent can access sealed fields
4. **Tenant Isolation** — Complete data separation between tenants
5. **Instance Namespacing** — Workflow instances are isolated within tenant

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
# Core security (v1.7.0)
from llmteam.tenancy import current_tenant, TenantContext, TenantManager
from llmteam.audit import AuditTrail, AuditEventType
from llmteam.context import SecureAgentContext, ContextAccessPolicy, HierarchicalContext
from llmteam.ratelimit import RateLimitedExecutor

# Orchestration (v1.8.0)
from llmteam.roles import PipelineOrchestrator, GroupOrchestrator
from llmteam.licensing import LicenseManager

# Workflow runtime (v1.9.0)
from llmteam.actions import ActionExecutor
from llmteam.human import HumanInteractionManager
from llmteam.persistence import SnapshotManager

# Canvas integration (v2.0.0)
from llmteam.runtime import RuntimeContext, StepContext, RuntimeContextManager
from llmteam.events import EventEmitter, WorktrailEvent, EventType
from llmteam.canvas import (
    SegmentDefinition, StepDefinition, EdgeDefinition,
    StepCatalog, SegmentRunner, RunConfig,
    HumanTaskHandler, create_human_task_handler,
)
```

## Canvas Segment Example (v2.0.0)

```python
from llmteam.canvas import SegmentDefinition, StepDefinition, EdgeDefinition, SegmentRunner

# Define segment
segment = SegmentDefinition(
    segment_id="example",
    name="Example Workflow",
    steps=[
        StepDefinition(step_id="start", step_type="transform", config={"expression": "input"}),
        StepDefinition(step_id="process", step_type="llm_agent", config={"model": "gpt-4"}),
        StepDefinition(step_id="end", step_type="transform", config={"expression": "output"}),
    ],
    edges=[
        EdgeDefinition(source_step="start", source_port="output", target_step="process", target_port="input"),
        EdgeDefinition(source_step="process", source_port="output", target_step="end", target_port="input"),
    ],
)

# Run segment
runner = SegmentRunner()
result = await runner.run(
    segment=segment,
    input_data={"query": "Hello"},
    runtime=runtime_context,
)
```

## Reference Documentation

- `v170-security-foundation.md` — v1.7.0 specification
- `v180-orchestration-intelligence.md` — v1.8.0 specification
- `v190-workflow-runtime.md` — v1.9.0 specification
- `rfc-v200-canvas-integration.md` — v2.0.0 specification
- `llmteam-v170-implementation-summary.md` — v1.7.0 implementation notes
- `llmteam-v190-implementation-summary.md` — v1.9.0 implementation notes
