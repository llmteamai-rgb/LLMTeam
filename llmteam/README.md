# llmteam

Enterprise AI Workflow Runtime for building multi-agent LLM pipelines with security, orchestration, and workflow capabilities.

## ⚠️ Migration from llm-pipeline-smtrk

This package was renamed from `llm-pipeline-smtrk` to `llmteam` starting with v1.7.0.

```python
# Old (deprecated)
from llm_pipeline_smtrk import create_pipeline

# New (recommended)
from llmteam import create_pipeline
```

The old import will continue to work for 2 releases with a deprecation warning.

## Installation

```bash
pip install llmteam

# With PostgreSQL support
pip install llmteam[postgres]

# With all optional dependencies
pip install llmteam[all]
```

## Features (v1.7.0 - Security Foundation)

### 🔐 Multi-Tenant Isolation

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

### 📋 Audit Trail

Compliance-ready audit logging with chain integrity verification.

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

### 🛡️ Context Security

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

# Orchestrator gets filtered view
visible = context.get_visible_context(
    viewer_id="orchestrator_1",
    viewer_role="pipeline_orch",
)
# visible["sealed_fields"] = ["card_number", "cvv"]
# But actual values are NOT included
```

### ⚡ Rate Limiting

Protect external APIs with rate limiting and circuit breaker.

```python
from llmteam.ratelimit import (
    RateLimitedExecutor,
    RateLimitConfig,
    CircuitBreakerConfig,
    RateLimitStrategy,
)
from datetime import timedelta

# Create executor
executor = RateLimitedExecutor()

# Register endpoint
executor.register(
    "external_api",
    RateLimitConfig(
        requests_per_minute=100,
        burst_size=10,
        strategy=RateLimitStrategy.QUEUE,
        retry_count=3,
    ),
    CircuitBreakerConfig(
        failure_threshold=5,
        open_timeout=timedelta(seconds=30),
    ),
)

# Execute with protection
result = await executor.execute(
    "external_api",
    call_api,
    param1="value",
)

if result.success:
    print(result.value)
else:
    print(f"Failed: {result.error}")
```

## Architecture

```
llmteam/
├── tenancy/          # Multi-tenant isolation
│   ├── models.py     # TenantConfig, TenantLimits
│   ├── manager.py    # TenantManager
│   ├── context.py    # TenantContext
│   ├── isolation.py  # TenantIsolatedStore
│   └── stores/       # Storage backends
│
├── audit/            # Audit trail
│   ├── models.py     # AuditRecord, AuditQuery
│   ├── trail.py      # AuditTrail
│   └── stores/       # Storage backends
│
├── context/          # Context security
│   ├── visibility.py # VisibilityLevel, SensitivityLevel
│   ├── security.py   # ContextAccessPolicy, SealedData
│   └── secure_context.py  # SecureAgentContext
│
└── ratelimit/        # Rate limiting
    ├── config.py     # RateLimitConfig, CircuitBreakerConfig
    ├── limiter.py    # RateLimiter
    ├── circuit.py    # CircuitBreaker
    └── executor.py   # RateLimitedExecutor
```

## Tier Limits

| Feature | FREE | STARTER | PROFESSIONAL | ENTERPRISE |
|---------|------|---------|--------------|------------|
| Concurrent Pipelines | 1 | 2 | 10 | Unlimited |
| Agents per Pipeline | 5 | 10 | 50 | Unlimited |
| Requests/Minute | 10 | 60 | 300 | Unlimited |
| Storage | 1 GB | 10 GB | 100 GB | Unlimited |
| Runs/Day | 100 | 1,000 | 10,000 | Unlimited |

## Key Principles

### Security

1. **Horizontal Isolation**: Agents NEVER see each other's contexts
2. **Sealed Data**: Only the owning agent can access sealed fields
3. **Audit Everything**: All actions are logged for compliance
4. **Tenant Isolation**: Complete data separation between tenants

### Reliability

1. **Rate Limiting**: Protect external APIs from overload
2. **Circuit Breaker**: Prevent cascading failures
3. **Retry with Backoff**: Automatic retry for transient failures

## Roadmap

- **v1.7.0** (Current): Security Foundation
- **v1.8.0**: Orchestration Intelligence (Process Mining, Smart Routing)
- **v1.9.0**: Workflow Runtime (External Actions, Human Interaction, Persistence)

## Contributing

Contributions are welcome! Please read our contributing guidelines.

## License

MIT License
