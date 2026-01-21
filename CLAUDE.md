# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**llmteam-ai** — Enterprise AI Workflow Runtime for building multi-agent LLM pipelines with security, orchestration, and workflow capabilities.

- **PyPI package:** `llmteam-ai` (install via `pip install llmteam-ai`)
- **Import as:** `import llmteam`
- **Current version:** 4.2.0 (Quality Slider - RFC-008)
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

### Core Concept: Teams with Typed Agents and TeamOrchestrator (v4.1.0)

LLMTeam orchestrates typed AI agents (LLM, RAG, KAG). TeamOrchestrator is a **separate supervisor entity** (NOT an agent):

```
Canvas (SegmentRunner)         — Routing logic (edges, conditions, workflow)
       │
       ▼
LLMTeam + TeamOrchestrator     — Agent container + separate supervisor
       │                         (orchestrator NOT in _agents dict)
       ▼
LLMGroup                       — Multi-team coordination
       │
       ▼
Typed Agents (LLM/RAG/KAG)     — LLM calls, retrieval, knowledge graphs
       │
       ▼
AgentReport                    — Automatic reporting to orchestrator
```

**Key Principles (v4.1.0):**
- Only 3 agent types: LLM, RAG, KAG (no custom Agent classes)
- Agents are created through `LLMTeam.add_agent(config)` using `AgentFactory`
- **TeamOrchestrator is separate** — NOT an agent, NOT in `_agents` dict
- **OrchestratorMode** — PASSIVE (default) vs ACTIVE (ROUTER) vs FULL
- **AgentReport** — Agents automatically report to orchestrator after execution
- Flow supports DAG: string ("a -> b -> c") or dict with edges/conditions
- Context modes: SHARED (one mailbox) vs NOT_SHARED (per-agent mailbox)
- **Reserved roles** — Roles starting with `_` are reserved for internal use

### Module Structure

**Core (v4.1.0):**
| Module | Purpose |
|--------|---------|
| `agents/` | Typed agents (LLMAgent, RAGAgent, KAGAgent), AgentFactory, configs, presets |
| `agents/orchestrator.py` | **TeamOrchestrator**, OrchestratorMode, OrchestratorConfig, RoutingDecision |
| `agents/report.py` | **AgentReport** model for agent-to-orchestrator reporting |
| `agents/prompts.py` | Routing and recovery prompts for orchestrator LLM |
| `team/` | LLMTeam container, LLMGroup, RunResult, TeamSnapshot |
| `contract.py` | TeamContract, ContractValidationResult |
| `registry/` | BaseRegistry[T], AgentRegistry, TeamRegistry |
| `escalation/` | EscalationLevel, handlers (Default, Threshold, Chain) |
| `mining/` | ProcessMiningEngine, ProcessEvent, ProcessMetrics |
| `quality/` | **QualityManager**, QualityPreset, CostEstimate, CostEstimator (RFC-008) |
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

**TeamOrchestrator Pattern (v4.1.0):** Separate supervisor with modes:
```python
from llmteam import LLMTeam
from llmteam.agents.orchestrator import (
    TeamOrchestrator, OrchestratorMode, OrchestratorConfig
)

# PASSIVE mode (default) - Canvas controls flow, all agents called sequentially
team = LLMTeam(
    team_id="support",
    agents=[
        {"type": "llm", "role": "billing", "prompt": "Handle billing: {query}"},
        {"type": "llm", "role": "technical", "prompt": "Handle tech: {query}"},
    ],
)
# team.get_orchestrator() returns TeamOrchestrator in PASSIVE mode
# team.is_router_mode == False

# ROUTER mode (ACTIVE) - Orchestrator LLM selects ONE agent per task
team = LLMTeam(
    team_id="support",
    agents=[
        {"type": "llm", "role": "billing", "prompt": "Handle billing: {query}"},
        {"type": "llm", "role": "technical", "prompt": "Handle tech: {query}"},
    ],
    orchestration=True,  # Enables ROUTER mode
)
# team.is_router_mode == True
# Orchestrator decides: billing question → billing agent only

# Or configure explicitly
team = LLMTeam(
    team_id="support",
    agents=[...],
    orchestrator=OrchestratorConfig(
        mode=OrchestratorMode.ACTIVE,  # SUPERVISOR | REPORTER | ROUTER
        model="gpt-4o",
        auto_retry=True,
        max_retries=2,
    ),
)
```

**OrchestratorMode Flags (v4.1.0):**
```python
from llmteam.agents.orchestrator import OrchestratorMode

# Individual flags
OrchestratorMode.SUPERVISOR  # Observe, receive reports
OrchestratorMode.REPORTER    # Generate execution reports
OrchestratorMode.ROUTER      # Control agent selection (LLM decides)
OrchestratorMode.RECOVERY    # Decide on error recovery

# Presets
OrchestratorMode.PASSIVE  # SUPERVISOR | REPORTER (default)
OrchestratorMode.ACTIVE   # SUPERVISOR | REPORTER | ROUTER
OrchestratorMode.FULL     # All flags enabled
```

**AgentReport Pattern (v4.1.0):** Automatic reporting to orchestrator:
```python
from llmteam.agents.report import AgentReport

# Agents automatically report to orchestrator after execution
# AgentReport contains:
# - agent_id, agent_role, agent_type
# - started_at, completed_at, duration_ms
# - input_summary, output_summary
# - success, error, tokens_used, model

# Access reports via orchestrator
orchestrator = team.get_orchestrator()
reports = orchestrator.reports  # List[AgentReport]
summary = orchestrator.get_summary()  # Dict with stats
report_text = orchestrator.generate_report()  # Markdown/JSON/text
```

**LLMTeam Pattern (v4.2.0):** Team container with typed agents and quality control:
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
    quality=70,  # RFC-008: Quality slider 0-100
    max_cost_per_run=1.00,  # RFC-008: Optional cost limit
)

# Execute team
result = await team.run({"query": "AI trends"})
print(result.output)  # Combined agent outputs
print(result.report)  # Execution report (v4.1.0)
print(result.summary)  # Execution summary (v4.1.0)

# RFC-008: Get cost estimate and quality info
estimate = await team.estimate_cost()
manager = team.get_quality_manager()
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

**Orchestrator Presets (v4.1.0):** Pre-built configs for common patterns:
```python
from llmteam.agents import create_orchestrator_config, create_summarizer_config

# ROUTER mode (v4.1.0) - orchestrator is separate, not added as agent
team = LLMTeam(team_id="adaptive", orchestration=True)
# team.get_orchestrator() is TeamOrchestrator (NOT an agent)
# team.is_router_mode == True

# Presets still available for specialized LLMAgents
config = create_summarizer_config(role="summarizer", model="gpt-4o")
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

**Quality Slider Pattern (RFC-008):** Control quality/cost tradeoff with single 0-100 parameter:
```python
from llmteam import LLMTeam, QualityManager
from llmteam.quality import QualityPreset, CostEstimator

# Create team with quality setting (0-100 or preset name)
team = LLMTeam(
    team_id="content",
    agents=[
        {"type": "llm", "role": "writer", "prompt": "Write: {query}"},
    ],
    quality=70,  # Higher = better quality, higher cost
    max_cost_per_run=1.00,  # Optional cost limit
)

# Or use preset names: draft(20), economy(30), balanced(50), production(75), best(95)
team = LLMTeam(team_id="fast", quality="draft")
team = LLMTeam(team_id="prod", quality="production")

# Get cost estimate before running
estimate = await team.estimate_cost(complexity="medium")
print(f"Estimated: ${estimate.min_cost:.2f} - ${estimate.max_cost:.2f}")

# Run with quality override or importance
result = await team.run({"query": "..."}, quality=80)  # Override quality
result = await team.run({"query": "..."}, importance="high")  # +20 quality

# Use QualityManager directly for model selection
manager = QualityManager(quality=70)
model = manager.get_model("complex")  # Returns appropriate model
params = manager.get_generation_params()  # {max_tokens, temperature}
depth = manager.get_pipeline_depth()  # SHALLOW | MEDIUM | DEEP

# Auto mode: quality adjusts based on budget usage
manager = QualityManager(quality="auto")
manager.set_daily_budget(10.0)
manager.record_spend(5.0)  # Tracks spending, adjusts quality
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

## Migration from v4.1.0 to v4.2.0

```python
# v4.1.0 (still works)
team = LLMTeam(team_id="support", agents=[...])
result = await team.run(data)

# v4.2.0 (RFC-008: Quality Slider)
from llmteam import LLMTeam, QualityManager

team = LLMTeam(
    team_id="support",
    agents=[...],
    quality=70,  # NEW: Quality slider 0-100
    max_cost_per_run=1.00,  # NEW: Optional cost limit
)

# NEW: Get cost estimate before running
estimate = await team.estimate_cost(complexity="medium")

# NEW: Run with quality override or importance
result = await team.run(data, quality=80)  # Override quality
result = await team.run(data, importance="high")  # +20 quality

# NEW: Access QualityManager
manager = team.get_quality_manager()

# Key additions in v4.2.0:
# - LLMTeam.quality property (0-100 or preset name)
# - LLMTeam.max_cost_per_run property
# - LLMTeam.estimate_cost() method
# - LLMTeam.get_quality_manager() method
# - run() accepts quality= and importance= parameters
# - QualityManager for model selection based on quality
# - CostEstimator for cost prediction
```

## Migration from v4.0.0 to v4.1.0

```python
# v4.0.0 (deprecated)
team = LLMTeam(team_id="support", orchestration=True)
# Orchestrator was added as agent in _agents dict
# team._has_orchestrator attribute used for checking

# v4.1.0 (recommended)
team = LLMTeam(team_id="support", orchestration=True)
# TeamOrchestrator is now SEPARATE entity (not an agent)
orchestrator = team.get_orchestrator()  # Returns TeamOrchestrator
is_routing = team.is_router_mode  # Check if ROUTER mode enabled

# Key changes in v4.1.0:
# - TeamOrchestrator is separate class, NOT in _agents dict
# - Use get_orchestrator() instead of _has_orchestrator
# - Use is_router_mode property to check ROUTER mode
# - Roles starting with _ are reserved (will raise error)
# - RunResult now has .report and .summary fields
# - AgentReport model for automatic reporting
```

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

# Create team with typed agents (v4.1.0)
team = LLMTeam(
    team_id="triage_team",
    agents=[
        {"type": "llm", "role": "triage", "prompt": "Categorize: {query}"},
    ],
)
# team.get_orchestrator() returns TeamOrchestrator in PASSIVE mode

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
# v4.1.0: result.report and result.summary available from orchestrator
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
