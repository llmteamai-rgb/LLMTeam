# llmteam-ai

Enterprise AI Workflow Runtime for building multi-agent LLM pipelines with security, orchestration, and workflow capabilities.

[![PyPI version](https://badge.fury.io/py/llmteam-ai.svg)](https://pypi.org/project/llmteam-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Current Version: v2.0.0 — Canvas Integration

### New Features in v2.0.0

- **RuntimeContext Injection** — Unified access point for enterprise resources (stores, clients, LLMs, secrets) with dependency injection
- **Worktrail Events** — Real-time event streaming for Canvas UI integration with EventEmitter and EventStore
- **Segment JSON Contract** — Declarative workflow definition with SegmentDefinition, StepDefinition, EdgeDefinition
- **Step Catalog API** — Registry of 7 built-in step types (llm_agent, transform, human_task, conditional, parallel, loop, api_call)
- **Segment Runner** — Async execution engine for canvas segments with topological ordering and port-based data flow

## Installation

```bash
pip install llmteam-ai

# With PostgreSQL support
pip install llmteam-ai[postgres]

# With API server (FastAPI)
pip install llmteam-ai[api]

# With all optional dependencies
pip install llmteam-ai[all]
```

## Quick Start

### Define and Run a Segment

```python
from llmteam.canvas import SegmentDefinition, StepDefinition, EdgeDefinition, SegmentRunner
from llmteam.runtime import RuntimeContextFactory

# Create runtime context
factory = RuntimeContextFactory()
runtime = factory.create_runtime(
    tenant_id="acme",
    instance_id="workflow-1",
)

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
    runtime=runtime,
)
print(result.status)  # SegmentStatus.COMPLETED
```

### CLI Usage

```bash
# Validate segment definition
llmteam validate segment.json

# Run segment
llmteam run segment.json --input-json '{"query": "Hello"}'

# List available step types
llmteam catalog

# Start API server
llmteam serve --port 8000
```

## Features

### v2.0.0 — Canvas Integration

#### Runtime Context

Inject runtime resources (stores, clients, secrets, LLMs) into step execution.

```python
from llmteam.runtime import RuntimeContextFactory

# Create runtime factory with registries
factory = RuntimeContextFactory()
factory.register_store("redis", redis_store)
factory.register_client("http", http_client)
factory.set_secrets_provider(vault_provider)

# Create runtime for workflow instance
runtime = factory.create_runtime(
    tenant_id="acme",
    instance_id="workflow_123",
)

# Create step context for step execution
step_ctx = runtime.child_context("process_data")

# Access resources in step handler
store = step_ctx.get_store("redis")
secret = await step_ctx.get_secret("api_key")
llm = step_ctx.get_llm("openai")
```

#### Worktrail Events

Emit events for Canvas UI integration.

```python
from llmteam.events import EventEmitter, EventType, MemoryEventStore

store = MemoryEventStore()
emitter = EventEmitter(store)

# Emit step events
await emitter.emit_step_started(run_id, step_id, input_data)
await emitter.emit_step_completed(run_id, step_id, output_data, duration_ms)
await emitter.emit_step_failed(run_id, step_id, error_info)

# Query events
events = await store.query(run_id=run_id, event_types=[EventType.STEP_COMPLETED])
```

#### Segment Definition (JSON Contract)

Define workflow segments with steps and edges.

```python
from llmteam.canvas import SegmentDefinition, StepDefinition, EdgeDefinition, PortDefinition

segment = SegmentDefinition(
    segment_id="data_pipeline",
    name="Data Processing Pipeline",
    version="1.0.0",
    steps=[
        StepDefinition(
            step_id="fetch",
            step_type="api_call",
            config={"url": "https://api.example.com/data", "method": "GET"},
            outputs=[PortDefinition(port_id="output", data_type="json")],
        ),
        StepDefinition(
            step_id="transform",
            step_type="transform",
            config={"expression": "data.items"},
            inputs=[PortDefinition(port_id="input", data_type="json")],
            outputs=[PortDefinition(port_id="output", data_type="json")],
        ),
        StepDefinition(
            step_id="analyze",
            step_type="llm_agent",
            config={"model": "gpt-4", "system_prompt": "Analyze the data"},
            inputs=[PortDefinition(port_id="input", data_type="json")],
        ),
    ],
    edges=[
        EdgeDefinition(source_step="fetch", source_port="output", target_step="transform", target_port="input"),
        EdgeDefinition(source_step="transform", source_port="output", target_step="analyze", target_port="input"),
    ],
)
```

#### Step Catalog

Registry of available step types with metadata.

```python
from llmteam.canvas import StepCatalog, StepCategory

catalog = StepCatalog.instance()

# Get all step types
all_types = catalog.list_all()

# Get types by category
ai_types = catalog.list_by_category(StepCategory.AI)

# Get step metadata
llm_agent = catalog.get("llm_agent")
print(llm_agent.display_name)  # "LLM Agent"
```

Built-in step types:

| Type | Category | Description |
|------|----------|-------------|
| `llm_agent` | AI | LLM-powered agent step |
| `transform` | Data | Data transformation |
| `human_task` | Human | Human approval/input |
| `conditional` | Control | Conditional branching |
| `parallel` | Control | Parallel execution |
| `loop` | Control | Iterative processing |
| `api_call` | Integration | External API calls |

#### Segment Runner

Execute workflow segments with handlers.

```python
from llmteam.canvas import SegmentRunner, RunConfig

# Create runner
runner = SegmentRunner()

# Run segment
result = await runner.run(
    segment=segment,
    input_data={"query": "Process this data"},
    runtime=runtime_context,
    config=RunConfig(timeout_seconds=300, max_retries=3),
)

# Check result
if result.status == SegmentStatus.COMPLETED:
    print(result.outputs)
elif result.status == SegmentStatus.FAILED:
    print(result.error)
```

### Previous Versions

#### v1.9.0 — Workflow Runtime
- External Actions (API/webhook calls with retry)
- Human-in-the-loop interaction (approval, chat, escalation)
- Snapshot-based pause/resume for long-running workflows

#### v1.8.0 — Orchestration Intelligence
- Hierarchical context propagation
- Pipeline orchestration with smart routing
- Process mining with XES export
- License-based feature management

#### v1.7.0 — Security Foundation
- Multi-tenant isolation with configurable limits
- Compliance audit trail with SHA-256 chain integrity
- Secure agent context with sealed data
- Rate limiting with circuit breaker

## Architecture

```
llmteam/
├── runtime/          # Runtime context injection (v2.0.0)
├── events/           # Worktrail events (v2.0.0)
├── canvas/           # Canvas segment execution (v2.0.0)
│   ├── models.py     # SegmentDefinition, StepDefinition, EdgeDefinition
│   ├── catalog.py    # StepCatalog with 7 built-in types
│   ├── runner.py     # SegmentRunner execution engine
│   └── handlers.py   # HumanTaskHandler
├── actions/          # External API/webhook calls (v1.9.0)
├── human/            # Human-in-the-loop (v1.9.0)
├── persistence/      # Snapshot pause/resume (v1.9.0)
├── roles/            # Orchestration roles (v1.8.0)
├── execution/        # Parallel pipeline execution (v1.8.0)
├── licensing/        # License management (v1.8.0)
├── tenancy/          # Multi-tenant isolation (v1.7.0)
├── audit/            # Compliance audit trail (v1.7.0)
├── context/          # Context security (v1.7.0)
├── ratelimit/        # Rate limiting + circuit breaker (v1.7.0)
├── cli/              # Command-line interface
├── api/              # REST API with FastAPI
└── observability/    # Structured logging
```

## Key Principles

### Security

1. **Horizontal Isolation**: Agents NEVER see each other's contexts
2. **Vertical Visibility**: Orchestrators see only their child agents
3. **Sealed Data**: Only the owning agent can access sealed fields
4. **Tenant Isolation**: Complete data separation between tenants
5. **Instance Namespacing**: Workflow instances isolated within tenant

### Canvas Integration

1. **JSON Contract**: Segments defined as portable JSON
2. **Step Catalog**: Extensible registry of step types
3. **Event-Driven**: UI updates via Worktrail events
4. **Resource Injection**: Runtime context provides stores, clients, secrets

## Links

- [PyPI Package](https://pypi.org/project/llmteam-ai/)
- [GitHub Repository](https://github.com/llmteamai-rgb/LLMTeam)
- [Changelog](https://github.com/llmteamai-rgb/LLMTeam/blob/main/llmteam/CHANGELOG.md)

## License

Apache 2.0 License
