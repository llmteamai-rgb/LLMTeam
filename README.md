# llmteam-ai

Enterprise AI Workflow Runtime for building multi-agent LLM pipelines with security, orchestration, and workflow capabilities.

[![PyPI version](https://badge.fury.io/py/llmteam-ai.svg)](https://pypi.org/project/llmteam-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Current Version: v2.3.0 — Team Contracts & Canvas Integration

### New Features in v2.3.0

- **TeamContract** — Formal interface definitions for team inputs/outputs with type validation
- **TeamHandler** — Execute agent teams as Canvas workflow steps
- **Escalation System** — Structured escalation handling in GroupOrchestrator (INFO → WARNING → CRITICAL → EMERGENCY)
- **Coordinator Role** — GroupOrchestrator transitioned from Router to Coordinator/Supervisor

> **Migration Note:** `GroupOrchestrator.orchestrate()` is deprecated. Use Canvas with TeamHandler for routing between teams.

---

## What LLMTeam Is

LLMTeam is an **AI agent orchestration library** that enables:

- Creating teams of AI agents with defined roles
- Orchestrating agent interactions within pipelines
- Integrating agent teams into visual workflow builders (Canvas)
- Enterprise-grade security, audit, and multi-tenancy

## What LLMTeam is NOT

| NOT | Explanation |
|-----|-------------|
| ❌ **Workflow Engine** | LLMTeam orchestrates agents, not business processes. Use Canvas for workflow routing. |
| ❌ **BPM System** | No BPMN, no process definitions, no human task management (that's Canvas responsibility) |
| ❌ **Rule Engine** | Conditions are soft-routing hints, not business rules |
| ❌ **Execution History Owner** | Platforms (like KorpOS) own execution history; LLMTeam provides events |

### Architecture: Who Does What

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

---

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

---

## Quick Start

### 1. Define a Team with Contract

```python
from llmteam.roles import PipelineOrchestrator, TeamContract, RuleBasedStrategy
from llmteam.ports import TypedPort, PortLevel, PortDirection

# Define formal contract
contract = TeamContract(
    name="triage_team",
    inputs=[
        TypedPort(
            name="ticket",
            level=PortLevel.WORKFLOW,
            direction=PortDirection.INPUT,
            data_type="object",
            required=True,
            description="Support ticket to classify",
        ),
    ],
    outputs=[
        TypedPort(
            name="category",
            level=PortLevel.WORKFLOW,
            direction=PortDirection.OUTPUT,
            data_type="string",
            required=True,
            description="Ticket category (billing/technical/general)",
        ),
        TypedPort(
            name="priority",
            level=PortLevel.WORKFLOW,
            direction=PortDirection.OUTPUT,
            data_type="string",
            required=True,
            description="Priority level (low/medium/high/critical)",
        ),
    ],
    strict=True,  # Reject unknown fields
    description="Triage team for ticket classification",
)

# Create team with contract
triage_team = PipelineOrchestrator(
    pipeline_id="triage",
    contract=contract,
    strategy=RuleBasedStrategy(),
    strict_validation=True,  # Raise errors on validation failure
)

# Register agents
triage_team.register_agent("classifier", classifier_agent)
triage_team.register_agent("prioritizer", prioritizer_agent)

# Execute
result = await triage_team.orchestrate("run_123", {
    "ticket": {"id": "T-001", "subject": "Cannot login", "body": "..."}
})
# result = {"category": "technical", "priority": "high", ...}
```

### 2. Use Team in Canvas Workflow

```python
from llmteam.canvas import SegmentDefinition, StepDefinition, EdgeDefinition, SegmentRunner
from llmteam.runtime import RuntimeContextFactory

# Register team in runtime
factory = RuntimeContextFactory()
runtime = factory.create_runtime(tenant_id="acme", instance_id="support-flow-1")
runtime.register_team("triage_team", triage_team)
runtime.register_team("billing_team", billing_team)
runtime.register_team("technical_team", technical_team)

# Define workflow with team steps
segment = SegmentDefinition(
    segment_id="support_workflow",
    name="Support Ticket Processing",
    steps=[
        StepDefinition(
            step_id="triage",
            step_type="team",
            config={
                "team_ref": "triage_team",
                "input_mapping": {"ticket": "input.ticket"},
            },
        ),
        StepDefinition(
            step_id="billing",
            step_type="team",
            config={
                "team_ref": "billing_team",
                "input_mapping": {"issue": "steps.triage.output"},
            },
        ),
        StepDefinition(
            step_id="technical",
            step_type="team",
            config={
                "team_ref": "technical_team",
                "input_mapping": {"issue": "steps.triage.output"},
            },
        ),
    ],
    edges=[
        EdgeDefinition(
            source_step="triage",
            target_step="billing",
            condition="steps.triage.output.category == 'billing'",
        ),
        EdgeDefinition(
            source_step="triage",
            target_step="technical",
            condition="steps.triage.output.category == 'technical'",
        ),
    ],
)

# Run workflow
runner = SegmentRunner()
result = await runner.run(
    segment=segment,
    input_data={"ticket": {"id": "T-001", "subject": "Billing question"}},
    runtime=runtime,
)
```

### 3. Handle Escalations

```python
from llmteam.roles import (
    GroupOrchestrator,
    Escalation,
    EscalationLevel,
    EscalationAction,
    EscalationDecision,
)

# Create coordinator group
group = GroupOrchestrator("support_group")
group.register_pipeline(triage_team)
group.register_pipeline(billing_team)
group.register_pipeline(technical_team)

# Handle escalation from a team
escalation = Escalation(
    escalation_id="esc-001",
    level=EscalationLevel.WARNING,
    source_pipeline="billing_team",
    reason="Refund amount exceeds threshold",
    context={"amount": 5000, "customer_tier": "standard"},
)

# Default handling based on level
decision = await group.handle_escalation(escalation)
# decision.action = EscalationAction.REDIRECT (to another team)

# Or use custom handler
def custom_handler(esc: Escalation) -> EscalationDecision:
    if esc.context.get("amount", 0) > 10000:
        return EscalationDecision(
            action=EscalationAction.HUMAN_REVIEW,
            message="Large refund requires manager approval",
        )
    return EscalationDecision(
        action=EscalationAction.ACKNOWLEDGE,
        message="Proceeding with standard refund",
    )

decision = await group.handle_escalation(escalation, handler=custom_handler)

# Collect metrics for monitoring
metrics = group.collect_metrics()
print(f"Health score: {metrics['health_score']}")
print(f"Escalations (24h): {metrics['escalations']['total']}")
```

---

## CLI Usage

```bash
# Validate segment definition
llmteam validate segment.json

# Run segment
llmteam run segment.json --input-json '{"ticket": {"id": "T-001"}}'

# List available step types (includes 'team' in v2.3.0)
llmteam catalog

# Start API server
llmteam serve --port 8000
```

---

## Features by Version

### v2.3.0 — Team Contracts & Canvas Integration (Current)

| Feature | Description |
|---------|-------------|
| **TeamContract** | Formal input/output contracts with validation |
| **TeamHandler** | Canvas step type for executing teams |
| **Escalation System** | Structured escalation with levels and actions |
| **Coordinator Role** | GroupOrchestrator as supervisor, not router |

### v2.2.0 — Extended Handlers

- SubworkflowHandler, SwitchHandler
- Redis/Kafka event transports
- JSONPath in transforms

### v2.1.0 — Extended Providers

- Vertex AI, Ollama, LiteLLM providers
- Enterprise secrets (Vault, AWS, Azure)
- GraphQL/gRPC clients

### v2.0.0 — Canvas Integration

- RuntimeContext injection
- Worktrail events
- SegmentRunner execution

### v1.7.0–v1.9.0 — Security & Workflow Foundation

- Multi-tenant isolation
- Audit trail with SHA-256 chain
- Human-in-the-loop, pause/resume

---

## Step Types

| Type | Category | Description |
|------|----------|-------------|
| `llm_agent` | AI | LLM-powered agent step |
| `team` | AI | **v2.3.0** Execute agent team |
| `transform` | Data | Data transformation |
| `human_task` | Human | Human approval/input |
| `condition` | Control | Conditional branching |
| `switch` | Control | Multi-way branching |
| `parallel_split` | Control | Fan-out to parallel |
| `parallel_join` | Control | Merge parallel results |
| `loop` | Control | Iterative processing |
| `subworkflow` | Control | Nested workflow |
| `http_action` | Integration | External API calls |

---

## Architecture

```
llmteam/
├── roles/            # Orchestration (v2.3.0: TeamContract, Escalation)
│   ├── contract.py   # TeamContract, ValidationResult
│   ├── pipeline_orch.py  # PipelineOrchestrator with contract support
│   └── group_orch.py # GroupOrchestrator with escalation handling
├── canvas/           # Canvas segment execution
│   ├── handlers/     # Step handlers (v2.3.0: TeamHandler)
│   ├── catalog.py    # StepCatalog with 11 built-in types
│   └── runner.py     # SegmentRunner
├── runtime/          # Runtime context injection
├── events/           # Worktrail events + transports
├── ports/            # TypedPort, PortLevel, PortDirection
├── providers/        # LLM providers (OpenAI, Anthropic, etc.)
├── tenancy/          # Multi-tenant isolation
├── audit/            # Compliance audit trail
└── observability/    # Structured logging, tracing
```

---

## Key Principles

### Security

1. **Horizontal Isolation** — Agents never see each other's contexts
2. **Vertical Visibility** — Orchestrators see only their child agents
3. **Tenant Isolation** — Complete data separation between tenants
4. **Contract Validation** — Type-safe boundaries between components

### Canvas Integration

1. **JSON Contract** — Segments defined as portable JSON
2. **Step Catalog** — Extensible registry of step types
3. **Team as Step** — Agent teams are first-class workflow steps
4. **Event-Driven** — UI updates via Worktrail events

### Orchestration

1. **Config-Driven** — Behavior defined by configuration, not code
2. **Execution-First** — Events are source of truth
3. **Escalation-Aware** — Structured handling of exceptional cases
4. **Stateless** — No global state, instance-scoped

---

## Links

- [PyPI Package](https://pypi.org/project/llmteam-ai/)
- [GitHub Repository](https://github.com/llmteamai-rgb/LLMTeam)
- [Changelog](https://github.com/llmteamai-rgb/LLMTeam/blob/main/CHANGELOG.md)

## License

Apache 2.0 License
