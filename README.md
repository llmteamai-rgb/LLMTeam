# LLMTeam

**Enterprise AI Workflow Runtime** for building multi-agent LLM pipelines with security, orchestration, and workflow capabilities.

[![PyPI version](https://badge.fury.io/py/llmteam-ai.svg)](https://pypi.org/project/llmteam-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Current Version: v2.1.0 — Extended Providers & Secrets

### New Features in v2.0.0

- **RuntimeContext Injection** — Unified access point for enterprise resources (stores, clients, LLMs, secrets) with dependency injection
- **Worktrail Events** — Real-time event streaming for Canvas UI integration with EventEmitter and EventStore
- **Segment JSON Contract** — Declarative workflow definition with SegmentDefinition, StepDefinition, EdgeDefinition
- **Step Catalog API** — Registry of 7 built-in step types (llm_agent, transform, human_task, conditional, parallel, loop, api_call)
- **Segment Runner** — Async execution engine for canvas segments with topological ordering and port-based data flow

## Installation

```bash
pip install llmteam-ai
```

With optional dependencies:

```bash
pip install llmteam-ai[api]        # FastAPI server
pip install llmteam-ai[postgres]   # PostgreSQL stores
pip install llmteam-ai[all]        # All features
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

## Architecture

### Module Structure

| Version | Module | Description |
|---------|--------|-------------|
| **v2.0.0** | `runtime/` | RuntimeContext, StepContext, resource registries |
| **v2.0.0** | `events/` | EventEmitter, WorktrailEvent, EventStore |
| **v2.0.0** | `canvas/` | SegmentDefinition, StepCatalog, SegmentRunner |
| v1.9.0 | `actions/` | External API/webhook calls |
| v1.9.0 | `human/` | Human-in-the-loop interaction |
| v1.9.0 | `persistence/` | Snapshot-based pause/resume |
| v1.8.0 | `roles/` | Pipeline/Group orchestrators |
| v1.8.0 | `execution/` | Parallel pipeline execution |
| v1.7.0 | `tenancy/` | Multi-tenant isolation |
| v1.7.0 | `audit/` | Compliance audit trail |
| v1.7.0 | `context/` | Secure agent context |
| v1.7.0 | `ratelimit/` | Rate limiting + circuit breaker |

### Step Types (v2.0.0)

| Type | Category | Description |
|------|----------|-------------|
| `llm_agent` | AI | LLM-powered agent step |
| `transform` | Data | Data transformation |
| `human_task` | Human | Human approval/input |
| `conditional` | Control | Conditional branching |
| `parallel` | Control | Parallel execution |
| `loop` | Control | Iterative processing |
| `api_call` | Integration | External API calls |

## Development

```bash
cd llmteam

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
python run_tests.py

# Type checking
mypy src/llmteam/

# Formatting
black src/ tests/
ruff check src/ tests/
```

## License

Apache 2.0 — see [LICENSE](llmteam/LICENSE) for details.

## Links

- [PyPI Package](https://pypi.org/project/llmteam-ai/)
- [GitHub Repository](https://github.com/llmteamai-rgb/LLMTeam)
- [Changelog](llmteam/CHANGELOG.md)
