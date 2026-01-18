# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**llmteam-ai** — Enterprise AI Workflow Runtime for building multi-agent LLM pipelines with security, orchestration, and workflow capabilities.

- **PyPI package:** `llmteam-ai` (install via `pip install llmteam-ai`)
- **Import as:** `import llmteam`
- **Current version:** 2.3.0
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

# Single test (PowerShell)
$env:PYTHONPATH="src"; pytest tests/canvas/test_runner.py::TestSegmentRunner::test_simple_run -vv

# Single test (bash)
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
llmteam check segment.json   # Comprehensive validation
llmteam run segment.json --input data.json
llmteam providers            # List LLM providers
llmteam serve --port 8000    # Start API server
```

## Architecture

### Core Concept: Teams as Canvas Steps

LLMTeam orchestrates AI agents within teams. Teams are invoked as Canvas workflow steps:

```
Canvas (SegmentRunner)         — Routing logic (edges, conditions, workflow)
       │
       ▼
GroupOrchestrator              — Coordination (escalations, metrics, supervision)
       │
       ▼
LLMTeam (PipelineOrchestrator) — Agent orchestration (internal pipeline)
       │
       ▼
Agents                         — LLM calls, tools, actions
```

**Key Principle:** Routing between teams is defined in Canvas, not in GroupOrchestrator.

### Module Structure

| Version | Module | Purpose |
|---------|--------|---------|
| v1.7.0 | `tenancy/` | Multi-tenant isolation |
| v1.7.0 | `audit/` | Compliance audit trail (SHA-256 chain) |
| v1.7.0 | `context/` | Secure agent context, sealed data |
| v1.7.0 | `ratelimit/` | Rate limiting + circuit breaker |
| v1.8.0 | `licensing/` | License tiers (Community/Professional/Enterprise) |
| v1.8.0 | `execution/` | Parallel pipeline execution |
| v1.8.0 | `roles/` | Orchestrators, process mining, contracts, escalation |
| v1.9.0 | `actions/` | External API/webhook calls |
| v1.9.0 | `human/` | Human-in-the-loop |
| v1.9.0 | `persistence/` | Snapshot pause/resume |
| v2.0.0 | `runtime/` | RuntimeContext, RuntimeContextFactory, StepContext |
| v2.0.0 | `events/` | Worktrail events for UI |
| v2.0.0 | `canvas/` | Segment execution engine |
| v2.0.0 | `canvas/handlers/` | Built-in step handlers |
| v2.0.0 | `canvas/validation` | Segment validation with JSON Schema |
| v2.0.0 | `observability/` | Structured logging (structlog) |
| v2.0.0 | `cli/` | Command-line interface |
| v2.0.0 | `api/` | REST + WebSocket API (FastAPI) |
| v2.0.0 | `ports/` | Port definitions for step I/O |
| v2.0.3 | `providers/` | LLM providers (OpenAI, Anthropic, Azure, Bedrock, Vertex, Ollama, LiteLLM) |
| v2.0.3 | `testing/` | Mock providers, SegmentTestRunner, StepTestHarness |
| v2.0.3 | `events/transports/` | WebSocketTransport, SSETransport |
| v2.0.4 | `middleware/` | Step execution middleware (logging, timing, retry, caching, auth) |
| v2.0.4 | `auth/` | OIDC, JWT, API key authentication + RBAC |
| v2.0.4 | `clients/` | HTTP, GraphQL, gRPC clients with retry and circuit breaker |
| v2.1.0 | `secrets/` | Secrets management (Vault, AWS, Azure, env fallback) |
| v2.2.0 | `canvas/handlers/subworkflow_handler` | Nested workflow execution |
| v2.2.0 | `canvas/handlers/switch_handler` | Multi-way branching (switch/case) |
| v2.2.0 | `events/transports/redis` | Redis Pub/Sub transport |
| v2.2.0 | `events/transports/kafka` | Kafka enterprise streaming |
| v2.3.0 | `roles/contract.py` | TeamContract with input/output validation |
| v2.3.0 | `canvas/handlers/team_handler.py` | Execute agent teams as Canvas steps |
| v2.3.0 | `transport/bus.py` | SecureBus for event-driven communication |

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

**TeamContract Pattern (v2.3.0):** Formal input/output contracts for teams:
```python
from llmteam.roles import TeamContract, PipelineOrchestrator

contract = TeamContract(
    name="triage_team",
    inputs=[TypedPort(name="ticket", data_type="object", required=True)],
    outputs=[TypedPort(name="category", data_type="string", required=True)],
    strict=True,
)

team = PipelineOrchestrator(pipeline_id="triage", contract=contract)
```

**Escalation Pattern (v2.3.0):** Structured escalation handling:
```python
from llmteam.roles import GroupOrchestrator, Escalation, EscalationLevel

group = GroupOrchestrator("support_group")
decision = await group.handle_escalation(Escalation(
    level=EscalationLevel.WARNING,
    source_pipeline="billing_team",
    reason="Refund exceeds threshold",
))
```

### Security Principles

1. **Horizontal Isolation** — Agents NEVER see each other's contexts
2. **Vertical Visibility** — Orchestrators see only their child agents
3. **Sealed Data** — Only the owner agent can access sealed fields
4. **Tenant Isolation** — Complete data separation between tenants

## Creating New Modules

1. Create module directory with `__init__.py` containing exports
2. Add imports to `llmteam/__init__.py` (or use lazy import for optional deps)
3. Create tests in `tests/{module}/test_{module}.py`
4. Add module to `TEST_MODULES` in `run_tests.py`

### Async Code

- Use `asyncio.Lock()` for thread-safety
- Tests use `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` needed)
- All async methods must consistently use `async`/`await`

## Built-in Step Handlers

| Handler | Step Type | Purpose |
|---------|-----------|---------|
| `LLMAgentHandler` | `llm_agent` | LLM completion with prompt templating |
| `TeamHandler` | `team` | **v2.3.0** Execute agent teams as steps |
| `HTTPActionHandler` | `http_action` | HTTP requests with headers/timeout |
| `TransformHandler` | `transform` | Data transformation with expressions |
| `ConditionHandler` | `condition` | Conditional branching |
| `SwitchHandler` | `switch` | Multi-way branching |
| `ParallelSplitHandler` | `parallel_split` | Fan-out to parallel branches |
| `ParallelJoinHandler` | `parallel_join` | Merge parallel results |
| `HumanTaskHandler` | `human_task` | Human approval/input |
| `SubworkflowHandler` | `subworkflow` | Nested workflow execution |
| `RAGHandler` | `rag` | Retrieval-augmented generation |

## Canvas Segment Example

```python
from llmteam.canvas import SegmentDefinition, StepDefinition, EdgeDefinition, SegmentRunner

segment = SegmentDefinition(
    segment_id="example",
    name="Example Workflow",
    entrypoint="start",
    steps=[
        StepDefinition(step_id="start", type="transform", config={}),
        StepDefinition(step_id="triage", type="team", config={"team_ref": "triage_team"}),
    ],
    edges=[
        EdgeDefinition(from_step="start", to_step="triage"),
    ],
)

runner = SegmentRunner()
result = await runner.run(segment=segment, input_data={"query": "Hello"}, runtime=runtime)
```

## Publishing to PyPI

```bash
cd llmteam
python -m build
python -m twine upload dist/* -u __token__ -p <pypi-token>
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
