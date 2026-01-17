# llmteam

Enterprise AI Workflow Runtime for building multi-agent LLM pipelines with security, orchestration, and workflow capabilities.

**Current Version: 1.9.0 (Workflow Runtime)**

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
└── persistence/      # Snapshot pause/resume (v1.9.0)
```

## Key Principles

### Security

1. **Horizontal Isolation**: Agents NEVER see each other's contexts
2. **Vertical Visibility**: Orchestrators see only their child agents
3. **Sealed Data**: Only the owning agent can access sealed fields
4. **Tenant Isolation**: Complete data separation between tenants

### Reliability

1. **Rate Limiting**: Protect external APIs from overload
2. **Circuit Breaker**: Prevent cascading failures
3. **Retry with Backoff**: Automatic retry for transient failures
4. **Persistence**: Snapshot-based recovery for long-running workflows

## Version History

- **v1.9.0** (Current): Workflow Runtime — External Actions, Human Interaction, Persistence
- **v1.8.0**: Orchestration Intelligence — Process Mining, Smart Routing, Licensing
- **v1.7.0**: Security Foundation — Multi-tenancy, Audit, Context Security, Rate Limiting

## License

MIT License
