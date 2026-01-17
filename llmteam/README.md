# llmteam

Enterprise AI Workflow Runtime for building multi-agent LLM pipelines with security, orchestration, and workflow capabilities.

**Current Version: 2.0.0 (Canvas Integration)**

## Installation

```bash
pip install llmteam

# With PostgreSQL support
pip install llmteam[postgres]

# With all optional dependencies
pip install llmteam[all]
```

## Features

### v1.7.0 — Security Foundation

#### Multi-Tenant Isolation

Complete data isolation between tenants with configurable limits and features.

```python
from llmteam.tenancy import TenantManager, TenantConfig, TenantTier
from llmteam.tenancy.stores import MemoryTenantStore

# Create manager
store = MemoryTenantStore()
manager = TenantManager(store)

# Create tenant
await manager.create_tenant(TenantConfig(
    tenant_id="acme",
    name="Acme Corporation",
    tier=TenantTier.PROFESSIONAL,
    data_region="eu-west-1",
))

# Use tenant context
async with manager.context("acme"):
    # All operations are isolated to "acme"
    pass
```

#### Audit Trail

Compliance-ready audit logging with SHA-256 chain integrity verification.

```python
from llmteam.audit import AuditTrail, AuditEventType
from llmteam.audit.stores import MemoryAuditStore

# Create audit trail
store = MemoryAuditStore()
audit = AuditTrail(store, tenant_id="acme")

# Log events
await audit.log(
    AuditEventType.PIPELINE_STARTED,
    actor_id="user@acme.com",
    resource_id="pipeline_123",
)

# Generate compliance report
report = await audit.generate_report(
    start=datetime(2024, 1, 1),
    end=datetime(2024, 12, 31),
)
```

#### Context Security

Secure agent context with sealed data and access control.

```python
from llmteam.context import (
    SecureAgentContext,
    ContextAccessPolicy,
    SensitivityLevel,
)

# Create secure context
context = SecureAgentContext(
    agent_id="payment_processor",
    agent_name="Payment Processor",
    access_policy=ContextAccessPolicy(
        sensitivity=SensitivityLevel.CONFIDENTIAL,
        sealed_fields={"card_number", "cvv"},
        audit_access=True,
    ),
)

# Store sealed data (only agent can access)
context.set_sealed("card_number", "4111-1111-1111-1111")

# Orchestrator gets filtered view (values NOT included)
visible = context.get_visible_context(
    viewer_id="orchestrator_1",
    viewer_role="pipeline_orch",
)
```

#### Rate Limiting

Protect external APIs with rate limiting and circuit breaker.

```python
from llmteam.ratelimit import (
    RateLimitedExecutor,
    RateLimitConfig,
    CircuitBreakerConfig,
)
from datetime import timedelta

executor = RateLimitedExecutor()

executor.register(
    "external_api",
    RateLimitConfig(requests_per_minute=100, burst_size=10),
    CircuitBreakerConfig(failure_threshold=5, open_timeout=timedelta(seconds=30)),
)

result = await executor.execute("external_api", call_api, param1="value")
```

### v1.8.0 — Orchestration Intelligence

#### Hierarchical Context

Context propagation with parent-child visibility rules.

```python
from llmteam.context import HierarchicalContext, ContextManager, ContextScope

context_manager = ContextManager()

# Create hierarchical context
parent_ctx = HierarchicalContext(
    context_id="pipeline_1",
    scope=ContextScope.PIPELINE,
)

child_ctx = context_manager.create_child(
    parent=parent_ctx,
    context_id="agent_1",
    scope=ContextScope.AGENT,
)

# Child inherits from parent, parent sees child (vertical visibility)
```

#### Pipeline Orchestration

Smart routing with rule-based and LLM-based strategies.

```python
from llmteam.roles import PipelineOrchestrator, RuleBasedStrategy

orchestrator = PipelineOrchestrator(
    pipeline_id="data_pipeline",
    strategy=RuleBasedStrategy(rules=[...]),
)

# Route tasks to appropriate agents
decision = await orchestrator.decide(task_context)
```

#### Process Mining

XES export for process analysis with ProM/Celonis.

```python
from llmteam.roles import ProcessMiningEngine

engine = ProcessMiningEngine()
engine.record_event(process_id, activity, timestamp)

# Export to XES format
xes_data = engine.export_xes()
```

#### Licensing

License-based feature and limit management.

```python
from llmteam.licensing import LicenseManager, LicenseTier

license_manager = LicenseManager()
await license_manager.set_license(tenant_id, LicenseTier.PROFESSIONAL)

# Check limits
can_create = await license_manager.check_limit(tenant_id, "pipelines", current=5)
```

### v1.9.0 — Workflow Runtime

#### External Actions

Execute external API calls and webhooks with retry and timeout handling.

```python
from llmteam.actions import ActionExecutor, ActionRegistry, ActionConfig, ActionType

registry = ActionRegistry()
registry.register(ActionConfig(
    action_id="notify_slack",
    action_type=ActionType.WEBHOOK,
    endpoint="https://hooks.slack.com/...",
    timeout=timedelta(seconds=10),
))

executor = ActionExecutor(registry)
result = await executor.execute("notify_slack", payload={"text": "Pipeline complete"})
```

#### Human Interaction

Human-in-the-loop with approval, chat, and escalation support.

```python
from llmteam.human import HumanInteractionManager, InteractionType, MemoryInteractionStore

store = MemoryInteractionStore()
manager = HumanInteractionManager(store)

# Request approval
request = await manager.request_approval(
    pipeline_id="pipeline_1",
    message="Approve data processing?",
    assignee="admin@company.com",
)

# Wait for response
response = await manager.wait_for_response(request.request_id, timeout=timedelta(hours=24))
```

#### Persistence

Snapshot-based pause/resume for long-running workflows.

```python
from llmteam.persistence import SnapshotManager, MemorySnapshotStore

store = MemorySnapshotStore()
manager = SnapshotManager(store)

# Save pipeline state
snapshot = await manager.create_snapshot(pipeline_id, agents_state, metadata)

# Restore later
result = await manager.restore_snapshot(snapshot.snapshot_id)
```

### v2.0.0 — Canvas Integration

#### Runtime Context

Inject runtime resources (stores, clients, secrets, LLMs) into step execution.

```python
from llmteam.runtime import RuntimeContext, RuntimeContextManager, StepContext

# Create runtime manager with registries
manager = RuntimeContextManager()
manager.register_store("redis", redis_store)
manager.register_client("http", http_client)
manager.register_secrets_provider(vault_provider)

# Create runtime for workflow instance
runtime = manager.create_runtime(
    tenant_id="acme",
    instance_id="workflow_123",
)

# Create step context for step execution
step_ctx = runtime.child_context("process_data")

# Access resources in step handler
store = step_ctx.get_store("redis")
secret = step_ctx.get_secret("api_key")
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
            step_type="http_action",
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

# Validate and serialize
segment.validate()
json_data = segment.to_json()
```

#### Step Catalog

Registry of available step types with metadata.

```python
from llmteam.canvas import StepCatalog, StepCategory

catalog = StepCatalog.get_instance()

# Get all step types
all_types = catalog.list_types()

# Get types by category
ai_types = catalog.list_types(category=StepCategory.AI)

# Get step metadata
llm_agent = catalog.get_type("llm_agent")
print(llm_agent.display_name)  # "LLM Agent"
print(llm_agent.config_schema)  # JSON Schema for config

# Validate step config
is_valid = catalog.validate_config("llm_agent", {"model": "gpt-4"})
```

Built-in step types:
- `llm_agent` — LLM-powered agent (AI)
- `http_action` — HTTP API call (INTEGRATION)
- `human_task` — Human interaction (HUMAN)
- `condition` — Conditional branching (CONTROL)
- `parallel_split` — Parallel execution start (CONTROL)
- `parallel_join` — Parallel execution join (CONTROL)
- `transform` — Data transformation (UTILITY)

#### Segment Runner

Execute workflow segments with handlers.

```python
from llmteam.canvas import SegmentRunner, RunConfig, HumanTaskHandler
from llmteam.human import HumanInteractionManager

# Create runner with handlers
runner = SegmentRunner()

# Register custom handler for human_task
human_handler = HumanTaskHandler(manager=HumanInteractionManager(...))
runner.register_handler("human_task", human_handler)

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

#### Human Task Handler

Built-in handler for human_task step type.

```python
from llmteam.canvas import HumanTaskHandler, create_human_task_handler
from llmteam.human import HumanInteractionManager, MemoryInteractionStore

# Create handler with manager
store = MemoryInteractionStore()
manager = HumanInteractionManager(store)
handler = create_human_task_handler(manager=manager)

# Use in segment step config
step = StepDefinition(
    step_id="approval",
    step_type="human_task",
    config={
        "task_type": "approval",  # approval, choice, input, review
        "title": "Approve Data Processing",
        "description": "Please review and approve the processed data",
        "assignee_ref": "manager@company.com",
        "timeout_hours": 24,
    },
)
```

## Architecture

```
llmteam/
├── tenancy/          # Multi-tenant isolation (v1.7.0)
├── audit/            # Compliance audit trail (v1.7.0)
├── context/          # Context security + hierarchical (v1.7.0, v1.8.0)
├── ratelimit/        # Rate limiting + circuit breaker (v1.7.0)
├── licensing/        # License management (v1.8.0)
├── execution/        # Parallel pipeline execution (v1.8.0)
├── roles/            # Orchestration roles (v1.8.0)
├── actions/          # External API/webhook calls (v1.9.0)
├── human/            # Human-in-the-loop (v1.9.0)
├── persistence/      # Snapshot pause/resume (v1.9.0)
├── runtime/          # Runtime context injection (v2.0.0)
├── events/           # Worktrail events (v2.0.0)
└── canvas/           # Canvas segment execution (v2.0.0)
    ├── models.py     # SegmentDefinition, StepDefinition, EdgeDefinition
    ├── catalog.py    # StepCatalog with 7 built-in types
    ├── runner.py     # SegmentRunner execution engine
    ├── handlers.py   # HumanTaskHandler
    └── exceptions.py # Canvas-specific exceptions
```

## Key Principles

### Security

1. **Horizontal Isolation**: Agents NEVER see each other's contexts
2. **Vertical Visibility**: Orchestrators see only their child agents
3. **Sealed Data**: Only the owning agent can access sealed fields
4. **Tenant Isolation**: Complete data separation between tenants
5. **Instance Namespacing**: Workflow instances isolated within tenant

### Reliability

1. **Rate Limiting**: Protect external APIs from overload
2. **Circuit Breaker**: Prevent cascading failures
3. **Retry with Backoff**: Automatic retry for transient failures
4. **Persistence**: Snapshot-based recovery for long-running workflows

### Canvas Integration

1. **JSON Contract**: Segments defined as portable JSON
2. **Step Catalog**: Extensible registry of step types
3. **Event-Driven**: UI updates via Worktrail events
4. **Resource Injection**: Runtime context provides stores, clients, secrets

## Version History

- **v2.0.0** (Current): Canvas Integration — Runtime Context, Worktrail Events, Segment Runner
- **v1.9.0**: Workflow Runtime — External Actions, Human Interaction, Persistence
- **v1.8.0**: Orchestration Intelligence — Process Mining, Smart Routing, Licensing
- **v1.7.0**: Security Foundation — Multi-tenancy, Audit, Context Security, Rate Limiting

## License

MIT License
