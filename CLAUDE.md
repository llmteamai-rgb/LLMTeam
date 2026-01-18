# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**llmteam-ai** — Enterprise AI Workflow Runtime for building multi-agent LLM pipelines with security, orchestration, and workflow capabilities.

- **PyPI package:** `llmteam-ai` (install via `pip install llmteam-ai`)
- **Import as:** `import llmteam`
- **Current version:** 2.2.1
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
llmteam check segment.json   # Comprehensive validation (v2.2.0)
llmteam run segment.json --input data.json
llmteam providers            # List LLM providers (v2.2.0)
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
| v2.0.3 | `providers/` | LLM providers (OpenAI, Anthropic, Azure, Bedrock, Vertex, Ollama, LiteLLM) |
| v2.0.3 | `testing/` | Mock providers, SegmentTestRunner, StepTestHarness |
| v2.0.3 | `events/transports/` | WebSocketTransport, SSETransport |
| v2.0.4 | `middleware/` | Step execution middleware (logging, timing, retry, caching, auth, rate-limit) |
| v2.0.4 | `auth/` | OIDC, JWT, API key authentication + RBAC middleware |
| v2.0.4 | `clients/` | HTTP, GraphQL, gRPC clients with retry and circuit breaker |
| v2.1.0 | `secrets/` | Secrets management (Vault, AWS, Azure, env fallback) |
| v2.2.0 | `canvas/handlers/subworkflow_handler` | Nested workflow execution |
| v2.2.0 | `canvas/handlers/switch_handler` | Multi-way branching (switch/case) |
| v2.2.0 | `events/transports/redis` | Redis Pub/Sub transport |
| v2.2.0 | `events/transports/kafka` | Kafka enterprise streaming |
| v2.2.0 | `api/health` | Health check endpoints |
| v2.2.0 | `docs/` | Sphinx documentation |
| v2.2.0 | `examples/` | Quickstart, FastAPI, Enterprise examples |
| v2.2.1 | `canvas/widget` | Widget Protocol for KorpOS UI (`render()`, `handle_intent()`) |
| v2.2.1 | `canvas/handlers/rag_handler` | RAG Handler (native/proxy modes) |
| v2.2.1 | `context/provider` | Context Provider abstraction (native/proxy) |
| v2.2.1 | `tests/e2e/` | End-to-end workflow execution tests |

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
2. Add imports to `llmteam/__init__.py` (or use lazy import for optional deps)
3. Create tests in `tests/{module}/test_{module}.py`
4. Add module to `TEST_MODULES` in `run_tests.py`

Test directories: `actions`, `api`, `audit`, `auth`, `canvas`, `cli`, `clients`, `context`, `e2e`, `events`, `execution`, `human`, `licensing`, `middleware`, `observability`, `persistence`, `providers`, `ratelimit`, `roles`, `runtime`, `secrets`, `tenancy`, `testing`

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
| `LoopHandler` | `loop` | For-each, while, until, and range loops (v2.0.4) |
| `ErrorHandler` | `error` | Catch, fallback, retry, compensate modes (v2.0.4) |
| `TryCatchHandler` | `try_catch` | Structured try-catch-finally patterns (v2.0.4) |
| `SubworkflowHandler` | `subworkflow` | Execute nested workflow segments (v2.2.0) |
| `SwitchHandler` | `switch` | Multi-way branching based on value matching (v2.2.0) |
| `RAGHandler` | `rag` | Retrieval-augmented generation with native/proxy modes (v2.2.1) |

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

## LLM Providers (v2.0.3+)

Use with `pip install llmteam-ai[providers]` or individual optional deps.

```python
from llmteam.providers import OpenAIProvider, AnthropicProvider

# OpenAI
provider = OpenAIProvider(model="gpt-4o")
response = await provider.complete("Hello!")

# Anthropic
provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")

# LiteLLM (100+ providers via unified API)
from llmteam.providers import LiteLLMProvider
provider = LiteLLMProvider(model="gpt-4")
```

Available: `OpenAIProvider`, `AnthropicProvider`, `AzureOpenAIProvider`, `BedrockProvider`, `VertexAIProvider`, `OllamaProvider`, `LiteLLMProvider`

## Middleware (v2.0.4+)

Composable interceptors for step execution:

```python
from llmteam.middleware import MiddlewareStack, LoggingMiddleware, RetryMiddleware, TimingMiddleware

stack = MiddlewareStack()
stack.use(LoggingMiddleware())
stack.use(TimingMiddleware(slow_threshold_ms=1000))
stack.use(RetryMiddleware(max_retries=3))
```

Built-in: `LoggingMiddleware`, `TimingMiddleware`, `RetryMiddleware`, `CachingMiddleware`, `RateLimitMiddleware`, `AuthMiddleware`, `ValidationMiddleware`

## Secrets Management (v2.1.0)

```python
from llmteam.secrets import SecretsManager, VaultProvider, AWSSecretsProvider

# HashiCorp Vault
vault = VaultProvider(url="https://vault.example.com:8200")
manager = SecretsManager(provider=vault)
api_key = await manager.get_secret("openai/api-key")

# AWS Secrets Manager
aws = AWSSecretsProvider(region_name="us-east-1")
manager = SecretsManager(provider=aws)
```

Providers: `EnvSecretsProvider`, `VaultProvider`, `AWSSecretsProvider`, `AzureKeyVaultProvider`

## Clients (v2.0.4+)

HTTP, GraphQL, and gRPC clients with retry and circuit breaker:

```python
from llmteam.clients import HTTPClient, GraphQLClient, GRPCClient

# HTTP
client = HTTPClient(HTTPClientConfig(base_url="https://api.example.com"))
response = await client.get("/users")

# GraphQL
client = GraphQLClient(endpoint="https://api.example.com/graphql")
result = await client.execute('query { users { name } }')

# gRPC
async with GRPCClient(target="localhost:50051") as client:
    response = await client.unary_call("service.Method", "RPC", {"param": "value"})
```

## Authentication (v2.0.4+)

```python
from llmteam.auth import OIDCProvider, JWTValidator, APIKeyValidator

# OIDC
oidc = OIDCProvider(issuer="https://auth.example.com", client_id="app", client_secret="secret")
token = await oidc.authenticate()

# JWT validation
validator = JWTValidator(issuer="https://auth.example.com")
claims = await validator.validate(token)
```

## Testing Utilities (v2.0.3+)

```python
from llmteam.testing import MockLLMProvider, SegmentTestRunner, StepTestHarness

# Mock LLM with deterministic responses
mock_llm = MockLLMProvider(responses=["Hello!", "How can I help?"])

# Run segment in isolated test mode
runner = SegmentTestRunner()
result = await runner.run(segment, input_data)

# Unit test individual handlers
harness = StepTestHarness()
result = await harness.test_handler(handler, step_config, input_data)
```

Mocks: `MockLLMProvider`, `MockHTTPClient`, `MockStore`, `MockSecretsProvider`, `MockEventEmitter`

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
