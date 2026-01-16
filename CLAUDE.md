# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**llmteam** — Enterprise AI Workflow Runtime for building multi-agent LLM pipelines with security, orchestration, and workflow capabilities.

Renamed from `llm-pipeline-smtrk` in v1.7.0. The actual Python package is located in the `llmteam/` subdirectory.

## Version Status

| Version | Name | Status |
|---------|------|--------|
| **v1.7.0** | Security Foundation | ✅ IMPLEMENTED |
| **v1.8.0** | Orchestration Intelligence | 🔄 NEXT |
| **v1.9.0** | Workflow Runtime | ⏳ PLANNED |

## Directory Structure

```
LLMTeam/                      # Repository root
├── CLAUDE.md                 # This file
├── llmteam/                  # Python package directory
│   ├── src/llmteam/
│   │   ├── __init__.py       # Main exports
│   │   ├── _compat.py        # Backward compatibility with llm_pipeline_smtrk
│   │   │
│   │   ├── tenancy/          # Multi-tenant isolation
│   │   │   ├── models.py     # TenantConfig, TenantTier, TenantLimits
│   │   │   ├── context.py    # TenantContext, current_tenant
│   │   │   ├── manager.py    # TenantManager
│   │   │   ├── isolation.py  # TenantIsolatedStore
│   │   │   └── stores/       # MemoryTenantStore, PostgresTenantStore
│   │   │
│   │   ├── audit/            # Compliance audit trail
│   │   │   ├── models.py     # AuditRecord, AuditQuery, AuditEventType
│   │   │   ├── trail.py      # AuditTrail
│   │   │   └── stores/       # MemoryAuditStore, PostgresAuditStore
│   │   │
│   │   ├── context/          # Context security
│   │   │   ├── visibility.py # VisibilityLevel, SensitivityLevel
│   │   │   ├── security.py   # ContextAccessPolicy, SealedData
│   │   │   └── secure_context.py  # SecureAgentContext
│   │   │
│   │   └── ratelimit/        # Rate limiting + Circuit Breaker
│   │       ├── config.py     # RateLimitConfig, CircuitBreakerConfig
│   │       ├── limiter.py    # RateLimiter
│   │       ├── circuit.py    # CircuitBreaker
│   │       └── executor.py   # RateLimitedExecutor
│   │
│   ├── tests/                # Test suite (pytest + pytest-asyncio)
│   ├── pyproject.toml        # Package configuration
│   └── README.md             # Package README
│
├── v170-security-foundation.md      # v1.7.0 spec
├── v180-orchestration-intelligence.md  # v1.8.0 spec
└── v190-workflow-runtime.md         # v1.9.0 spec
```

## Implemented Features (v1.7.0)

### Tenancy
- `TenantConfig` — configuration with tier, limits, features
- `TenantTier` — FREE, STARTER, PROFESSIONAL, ENTERPRISE
- `TenantContext` — context manager (sync/async)
- `current_tenant` — ContextVar for current tenant
- `TenantManager` — CRUD, check limits/features
- `TenantIsolatedStore` — automatic namespace per tenant

### Audit
- `AuditRecord` — immutable record with SHA-256 checksum chain
- `AuditEventType` — 30+ event types
- `AuditTrail` — logging, query, verify_chain, generate_report
- PostgreSQL store with append-only protection

### Context Security
- `SensitivityLevel` — PUBLIC → TOP_SECRET
- `ContextAccessPolicy` — access rules
- `SealedData` — owner-only container
- `SecureAgentContext` — context with sealed fields
- **Key principle:** horizontal access between agents is FORBIDDEN

### Rate Limiting
- `RateLimiter` — token bucket with per-second/minute/hour limits
- `CircuitBreaker` — CLOSED → OPEN → HALF_OPEN states
- `RateLimitedExecutor` — combination + retry + fallback

## Next Phase: v1.8.0 Orchestration Intelligence

### Components
1. **Hierarchical Context** — hierarchical context propagation
2. **Pipeline Orchestrator Roles** — Orchestration + Process Mining
3. **Group Orchestrator Roles** — managing pipeline groups
4. **Parallel Execution** — parallel agent execution
5. **Licensing** — license-based limits

### New Modules to Create
```
llmteam/src/llmteam/
├── context/
│   ├── hierarchical.py       # HierarchicalContext, ContextManager
│   └── propagation.py        # ContextPropagationConfig
│
├── roles/                    # NEW
│   ├── __init__.py
│   ├── orchestration.py      # OrchestrationStrategy, OrchestrationContext
│   ├── process_mining.py     # ProcessMiningEngine, ProcessMetrics
│   ├── pipeline_orch.py      # PipelineOrchestrator
│   └── group_orch.py         # GroupOrchestrator
│
├── execution/                # NEW
│   ├── __init__.py
│   ├── executor.py           # PipelineExecutor
│   ├── config.py             # ExecutorConfig
│   └── stats.py              # ExecutionStats
│
└── licensing/                # NEW
    ├── __init__.py
    ├── models.py             # LicenseTier, LicenseLimits
    └── manager.py            # LicenseManager
```

See `v180-orchestration-intelligence.md` for full specification.

## Planned: v1.9.0 Workflow Runtime

### Components
1. **External Actions** — external API/webhook calls
2. **Human Interaction** — approval, chat, escalation
3. **Persistence** — snapshot for pause/resume

See `v190-workflow-runtime.md` for full specification.

## Development Commands

All commands should be run from the `llmteam/` directory.

### Setup

```bash
# Navigate to package directory
cd llmteam

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Verify package imports
PYTHONPATH=src python -c "import llmteam; print(f'v{llmteam.__version__}')"
```

### Testing (Memory-Safe)

**IMPORTANT:** Tests are optimized to prevent memory exhaustion. Use the provided test runner:

```bash
# Recommended: Sequential run (safest, prevents OOM)
python run_tests.py

# With limited parallelism (2 workers)
python run_tests.py --parallel 2

# Run specific module only
python run_tests.py --module tenancy
python run_tests.py --module audit

# Fast tests only
python run_tests.py --fast

# With coverage
python run_tests.py --coverage
```

### Manual Test Execution

If you need to run tests manually:

```bash
# Linux/Mac - Sequential by module (safe)
PYTHONPATH=src pytest tests/tenancy/ -v
PYTHONPATH=src pytest tests/audit/ -v
PYTHONPATH=src pytest tests/context/ -v
PYTHONPATH=src pytest tests/ratelimit/ -v

# Windows - Sequential by module (safe)
set PYTHONPATH=src && pytest tests/tenancy/ -v
set PYTHONPATH=src && pytest tests/audit/ -v

# PowerShell
$env:PYTHONPATH="src"; pytest tests/tenancy/ -v
```

**WARNING:** Do NOT run all tests at once with high parallelism - this causes memory issues!

```bash
# ❌ AVOID: This may cause out-of-memory errors
PYTHONPATH=src pytest tests/ -n auto

# ✅ SAFE: Use the test runner or limit workers
python run_tests.py --parallel 2
```

### Code Quality

```bash
# Type checking
mypy src/llmteam/

# Code formatting
black src/ tests/

# Linting
ruff check src/ tests/
```

### Coverage

```bash
# Using test runner (recommended)
python run_tests.py --coverage

# Manual
PYTHONPATH=src pytest tests/ -v --cov=llmteam --cov-report=html
# Report available at: htmlcov/index.html
```

For detailed testing documentation, see `TESTING.md`.

## Architecture Principles

### Security
1. **Horizontal Isolation** — agents NEVER see each other's contexts
2. **Sealed Data** — only the owner has access
3. **Audit Everything** — all actions are logged
4. **Tenant Isolation** — complete data separation between tenants

### Reliability
1. **Rate Limiting** — protect external APIs from overload
2. **Circuit Breaker** — prevent cascading failures
3. **Retry with Backoff** — automatic retry for transient failures

### Orchestration (v1.8.0)
1. **Vertical Visibility** — orchestrator sees its agents (parent-child only)
2. **Process Mining** — XES export for ProM/Celonis
3. **Smart Routing** — rule-based and LLM-based strategies

## Version Dependencies

```
v1.7.0 Security Foundation
    │
    ▼
v1.8.0 Orchestration Intelligence
    │   - uses TenantContext
    │   - uses SecureAgentContext
    │   - uses AuditTrail
    ▼
v1.9.0 Workflow Runtime
        - uses everything from v1.7.0 and v1.8.0
        - External Actions → RateLimitedExecutor
        - Human Interaction → AuditTrail
        - Persistence → TenantIsolatedStore
```

## Development Guidelines

### When Creating a New Module
1. Create `__init__.py` with exports
2. Add imports to parent `__init__.py`
3. Create tests in `tests/{module}/test_{module}.py`
4. Follow the existing module structure pattern

### Working with Async Code
- Use `asyncio.Lock()` for thread-safety
- Mark tests with `@pytest.mark.asyncio`
- All async methods should have `async`/`await` consistently

### Integration with Existing Modules
```python
from llmteam.tenancy import current_tenant, TenantContext
from llmteam.audit import AuditTrail, AuditEventType
from llmteam.context import SecureAgentContext, ContextAccessPolicy
from llmteam.ratelimit import RateLimitedExecutor
```

## Key Architecture Patterns

### Store Pattern
All persistence layers follow the store pattern:
- Abstract base class defines interface
- `MemoryStore` for testing and development
- `PostgresStore` for production (when available)
- Stores are injected into managers/trails

### Context Manager Pattern
Multi-tenant operations use context managers:
```python
async with manager.context(tenant_id):
    # All operations isolated to tenant_id
    pass
```

### Immutability for Security
- `AuditRecord` is immutable with checksum chain
- `SealedData` uses owner-only access pattern
- Context security prevents unauthorized access

## Reference Documentation

- v1.7.0 spec: `v170-security-foundation.md`
- v1.8.0 spec: `v180-orchestration-intelligence.md`
- v1.9.0 spec: `v190-workflow-runtime.md`
- Implementation summary: `llmteam-v170-implementation-summary.md`
