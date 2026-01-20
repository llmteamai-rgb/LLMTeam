# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**llmteam-ai** — Enterprise AI Workflow Runtime for building multi-agent LLM pipelines with security, orchestration, and workflow capabilities.

- **PyPI package:** `llmteam-ai` (install via `pip install llmteam-ai`)
- **Import as:** `import llmteam`
- **Current version:** 4.0.0 (Agent Architecture Refactoring)
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

### Core Concept: Teams with Typed Agents (v4.0.0)

LLMTeam orchestrates typed AI agents (LLM, RAG, KAG). Teams use SegmentRunner internally:

```
Canvas (SegmentRunner)         — Routing logic (edges, conditions, workflow)
       │
       ▼
LLMTeam                        — Agent container, uses SegmentRunner internally
       │
       ▼
LLMGroup                       — Multi-team coordination (group orchestrator = LLMAgent)
       │
       ▼
Typed Agents (LLM/RAG/KAG)     — LLM calls, retrieval, knowledge graphs
```

**Key Principles (v4.0.0):**
- Only 3 agent types: LLM, RAG, KAG (no custom Agent classes)
- Agents are created through `LLMTeam.add_agent(config)` using `AgentFactory`
- Orchestrator is just an LLMAgent with a specialized prompt
- Flow supports DAG: string ("a -> b -> c") or dict with edges/conditions
- Context modes: SHARED (one mailbox) vs NOT_SHARED (per-agent mailbox)

### Module Structure

**Core (v4.0.0):**
| Module | Purpose |
|--------|---------|
| `agents/` | Typed agents (LLMAgent, RAGAgent, KAGAgent), AgentFactory, configs, presets |
| `team/` | LLMTeam container, LLMGroup, RunResult, TeamSnapshot |
| `contract.py` | TeamContract, ContractValidationResult |
| `registry/` | BaseRegistry[T], AgentRegistry, TeamRegistry |
| `escalation/` | EscalationLevel, handlers (Default, Threshold, Chain) |
| `mining/` | ProcessMiningEngine, ProcessEvent, ProcessMetrics |
| `roles/` | Legacy module (deprecated, re-exports from new locations) |

**Canvas & Runtime:**
| Module | Purpose |
|--------|---------|
| `canvas/` | Segment execution engine, StepDefinition, EdgeDefinition |
| `canvas/handlers/` | 12 built-in step handlers (llm_agent, team, transform, etc.) |
| `runtime/` | RuntimeContext, StepContext, resource injection |
| `events/` | Worktrail events, EventEmitter, transports |

**Enterprise:**
| Module | Purpose |
|--------|---------|
| `providers/` | LLM providers (OpenAI, Anthropic, Azure, Bedrock, Vertex, Ollama) |
| `middleware/` | Step middleware (logging, timing, retry, caching, auth) |
| `auth/` | OIDC, JWT, API key + RBAC |
| `clients/` | HTTP, GraphQL, gRPC clients |
| `secrets/` | Vault, AWS, Azure secrets |
| `tenancy/` | Multi-tenant isolation |
| `audit/` | Compliance audit trail |

### Key Patterns

**LLMTeam Pattern (v4.0.0):** Team container with typed agents:
```python
from llmteam import LLMTeam

# Create team with agents (only LLM, RAG, KAG types)
team = LLMTeam(
    team_id="content",
    agents=[
        {"type": "rag", "role": "retriever", "collection": "docs"},
        {"type": "llm", "role": "writer", "prompt": "Write about: {query}"},
    ],
    flow="retriever -> writer",  # DAG flow
)

# Execute team
result = await team.run({"query": "AI trends"})
print(result.output)  # Combined agent outputs
```

**LLMGroup Pattern (v4.0.0):** Multi-team coordination:
```python
from llmteam import LLMTeam

support_team = LLMTeam(team_id="support", agents=[...])
billing_team = LLMTeam(team_id="billing", agents=[...])

# Create group with support_team as leader
group = support_team.create_group(
    group_id="customer_service",
    teams=[billing_team]
)

result = await group.run({"query": "Help with my bill"})
```

**Agent Configuration (v4.0.0):** Use configs or dicts:
```python
from llmteam.agents import LLMAgentConfig, RAGAgentConfig

# Config objects
team.add_agent(LLMAgentConfig(role="writer", prompt="...", model="gpt-4o"))
team.add_agent(RAGAgentConfig(role="retriever", collection="docs", top_k=5))

# Or dict shorthand
team.add_agent({"type": "llm", "role": "writer", "prompt": "..."})
team.add_agent({"type": "rag", "role": "retriever", "collection": "docs"})
```

**Orchestrator Presets (v4.0.0):** Pre-built agent configs:
```python
from llmteam.agents import create_orchestrator_config, create_summarizer_config

# Add orchestrator for adaptive flow
team = LLMTeam(team_id="adaptive", orchestration=True)  # Auto-adds orchestrator

# Or manually with presets
config = create_orchestrator_config(["agent1", "agent2"], model="gpt-4o")
team.add_agent(config)
```

**Escalation Pattern (v4.0.0):** Structured escalation handling:
```python
from llmteam.escalation import Escalation, EscalationLevel, ChainHandler, DefaultHandler

handler = ChainHandler([ThresholdHandler(threshold=3), DefaultHandler()])
decision = handler.handle(Escalation(
    level=EscalationLevel.WARNING,
    source_team="billing_team",
    reason="Refund exceeds threshold",
))
```

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
factory.register_team(support_team)  # v3.0.0: Register teams

runtime = factory.create_runtime(tenant_id="acme", instance_id="run-123")
step_ctx = runtime.child_context("step_1")
team = step_ctx.get_team("support")  # v3.0.0: Access teams
```

### Security Principles

1. **Horizontal Isolation** — Agents NEVER see each other's contexts
2. **Vertical Visibility** — Orchestrators see only their child agents
3. **Sealed Data** — Only the owner agent can access sealed fields
4. **Tenant Isolation** — Complete data separation between tenants

## Migration from v3.x to v4.0.0

```python
# v3.x (deprecated)
from llmteam import LLMTeam, Agent

class MyAgent(Agent):  # Custom Agent class
    async def process(self, state): ...

team = LLMTeam(team_id="support")
team.register_agent(MyAgent("triage"))
result = await team.run(data)

# v4.0.0 (recommended)
from llmteam import LLMTeam

# Use typed agents (LLM, RAG, KAG) via config dicts
team = LLMTeam(
    team_id="support",
    agents=[
        {"type": "llm", "role": "triage", "prompt": "Triage: {input}"},
        {"type": "llm", "role": "resolver", "prompt": "Resolve: {input}"},
    ],
    flow="triage -> resolver",
)
result = await team.run(data)

# Key changes in v4.0.0:
# - No custom Agent classes (use LLM/RAG/KAG types)
# - Agents created via add_agent(config) not register_agent()
# - Flow defined as string or dict DAG
# - Orchestrator is just LLMAgent with preset prompt
# - compat/ module removed (no backwards compatibility layer)
```

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
| `TeamHandler` | `team` | Execute LLMTeam as canvas step (v3.0.0 API) |
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
from llmteam import LLMTeam
from llmteam.runtime import RuntimeContextFactory

# Create team with typed agents (v4.0.0)
team = LLMTeam(
    team_id="triage_team",
    agents=[
        {"type": "llm", "role": "triage", "prompt": "Categorize: {query}"},
    ],
)

factory = RuntimeContextFactory()
factory.register_team(team)
runtime = factory.create_runtime(tenant_id="acme", instance_id="run-123")

# Define segment using team step
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
# result includes team_metadata with agents_called, iterations, escalations
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
