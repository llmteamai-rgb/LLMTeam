# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**llmteam-ai** — Enterprise AI Workflow Runtime for building multi-agent LLM pipelines with security, orchestration, and workflow capabilities.

- **PyPI package:** `llmteam-ai` (install via `pip install llmteam-ai`)
- **Import as:** `import llmteam`
- **Current version:** 2.0.3
- **Python:** >=3.10
- **License:** Apache-2.0

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
python run_tests.py --module canvas    # Single module
python run_tests.py --coverage         # With coverage

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

### Makefile (alternative)

```bash
make test           # Run all tests
make test-module MODULE=canvas  # Single module
make lint           # Ruff + mypy
make format         # Black
make build          # Build package
```

### CLI

```bash
llmteam --version
llmteam catalog              # List step types
llmteam validate segment.json
llmteam run segment.json --input data.json
llmteam serve --port 8000    # Start API server
```

## Architecture

### Module Structure

| Version | Module | Purpose |
|---------|--------|---------|
| v1.7.0 | `tenancy/` | Multi-tenant isolation |
| v1.7.0 | `audit/` | Compliance audit trail (SHA-256 chain) |
| v1.7.0 | `context/` | Secure agent context, sealed data |
| v1.7.0 | `ratelimit/` | Rate limiting + circuit breaker |
| v1.8.0 | `licensing/` | License tiers (Community/Professional/Enterprise) |
| v1.8.0 | `execution/` | Parallel pipeline execution |
| v1.8.0 | `roles/` | Orchestrators, process mining |
| v1.9.0 | `actions/` | External API/webhook calls |
| v1.9.0 | `human/` | Human-in-the-loop |
| v1.9.0 | `persistence/` | Snapshot pause/resume |
| v2.0.0 | `runtime/` | RuntimeContext, RuntimeContextFactory, StepContext |
| v2.0.0 | `events/` | Worktrail events for UI |
| v2.0.0 | `canvas/` | Segment execution engine |
| v2.0.0 | `canvas/handlers/` | Built-in step handlers (LLM, HTTP, Transform, Condition, Parallel) |
| v2.0.0 | `canvas/validation` | Segment validation with JSON Schema |
| v2.0.0 | `observability/` | Structured logging (structlog) |
| v2.0.0 | `cli/` | Command-line interface |
| v2.0.0 | `api/` | REST + WebSocket API (FastAPI) |
| v2.0.0 | `patterns/` | Workflow patterns (fan-out, aggregation) |
| v2.0.0 | `ports/` | Port definitions for step I/O |

### Key Patterns

**Store Pattern:** All stores use dependency injection:
- Abstract base class defines interface
- `MemoryStore` for testing, `PostgresStore` for production
- Located in `stores/` subdirectories

**RuntimeContext Pattern:** Resource injection for step execution:
```python
from llmteam.runtime import RuntimeContextFactory

factory = RuntimeContextFactory()
factory.register_store("redis", redis_store)
factory.register_llm("gpt4", openai_provider)

runtime = factory.create_runtime(tenant_id="acme", instance_id="run-123")
step_ctx = runtime.child_context("step_1")
```

**Context Manager Pattern:** Tenant-scoped operations:
```python
async with manager.context(tenant_id):
    # All operations isolated to tenant_id
    pass
```

### Security Principles

1. **Horizontal Isolation** — Agents NEVER see each other's contexts
2. **Vertical Visibility** — Orchestrators see only their child agents
3. **Sealed Data** — Only the owner agent can access sealed fields
4. **Tenant Isolation** — Complete data separation between tenants

## Creating New Modules

1. Create module directory with `__init__.py` containing exports
2. Add imports to `llmteam/__init__.py`
3. Create tests in `tests/{module}/test_{module}.py`
4. Add module to `TEST_MODULES` in `run_tests.py`

### Async Code

- Use `asyncio.Lock()` for thread-safety
- Tests use `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` needed)
- All async methods must consistently use `async`/`await`

### Validation

Validate segment definitions before execution:

```python
from llmteam.canvas import validate_segment, SegmentDefinition

result = validate_segment(segment)
if not result.is_valid:
    for msg in result.errors:
        print(f"{msg.severity}: {msg.message}")
```

## Canvas Segment Example

```python
from llmteam.canvas import SegmentDefinition, StepDefinition, EdgeDefinition, SegmentRunner

segment = SegmentDefinition(
    segment_id="example",
    name="Example Workflow",
    entrypoint="start",
    steps=[
        StepDefinition(step_id="start", type="transform", config={}),
        StepDefinition(step_id="process", type="llm_agent", config={"llm_ref": "gpt4"}),
    ],
    edges=[
        EdgeDefinition(from_step="start", to_step="process"),
    ],
)

runner = SegmentRunner()
result = await runner.run(segment=segment, input_data={"query": "Hello"}, runtime=runtime)
```

### Built-in Step Handlers

| Handler | Step Type | Purpose |
|---------|-----------|---------|
| `LLMAgentHandler` | `llm_agent` | LLM completion with prompt templating and variable substitution |
| `HTTPActionHandler` | `http_action` | HTTP requests (GET/POST/PUT/PATCH/DELETE) with headers/timeout |
| `TransformHandler` | `transform` | Data transformation with expressions, field mappings, functions |
| `ConditionHandler` | `condition` | Conditional branching (eq/ne/gt/lt/contains/and/or) |
| `ParallelSplitHandler` | `parallel_split` | Fan-out to parallel branches with branch_ids |
| `ParallelJoinHandler` | `parallel_join` | Merge parallel results (all/any/first strategies) |
| `HumanTaskHandler` | `human_task` | Human approval/input with timeout, requires HumanInteractionManager |

### Custom Step Handlers

Implement the handler protocol and register with `SegmentRunner.register_handler()`:

```python
from llmteam.canvas import SegmentRunner
from llmteam.runtime import StepContext

async def my_handler(step: StepDefinition, input_data: dict, context: StepContext) -> dict:
    # Your custom logic
    return {"result": "processed"}

runner = SegmentRunner()
runner.register_handler("my_step_type", my_handler)
```

## Publishing to PyPI

```bash
cd llmteam
python -m build
python -m twine upload dist/* -u __token__ -p <pypi-token>
```

## Documentation

```
docs/
├── specs/                              # Version specifications (RFC)
│   ├── v170-security-foundation.md
│   ├── v180-orchestration-intelligence.md
│   ├── v190-workflow-runtime.md
│   └── rfc-v200-canvas-integration.md
├── testing/                            # Testing documentation
│   └── TESTING.md                      # Main testing guide
├── llmteam-v*-implementation-summary.md  # Implementation notes
└── llmteam-v200-P*.md                  # Priority task lists
```

## Repository Structure

```
LLMTeam/
├── CLAUDE.md              # This file
├── README.md              # Project overview
├── llmteam/               # Python package (pip install -e ".[dev]")
│   ├── src/llmteam/       # Source code
│   ├── tests/             # Test suite
│   ├── Makefile           # Build commands
│   └── run_tests.py       # Test runner
├── docs/                  # Documentation
└── open-core-changes/     # Open Core licensing utilities
```
