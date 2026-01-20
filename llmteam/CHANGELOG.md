# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.0] - 2026-01-19

### Added

- **Typed Agent Architecture** - New `llmteam.agents` package:
  - `AgentType` enum - LLM, RAG, KAG (only 3 types allowed)
  - `AgentMode` enum - NATIVE, PROXY modes for RAG/KAG
  - `AgentConfig`, `LLMAgentConfig`, `RAGAgentConfig`, `KAGAgentConfig` - Configuration dataclasses
  - `AgentState` - Runtime state tracking with lifecycle methods
  - `AgentResult`, `RAGResult`, `KAGResult` - Typed execution results
  - `AgentFactory.create()` - Factory for creating agents from config dicts
  - Preset configs: `create_orchestrator_config`, `create_group_orchestrator_config`, etc.

- **LLMTeam v4.0.0** - Refactored team container:
  - Uses SegmentRunner internally (not custom runtime)
  - `add_agent(config)` - Add agent via dict or AgentConfig
  - `add_llm_agent()`, `add_rag_agent()`, `add_kag_agent()` - Shortcut methods
  - `flow` parameter - DAG definition as string ("a -> b -> c") or dict
  - `ContextMode` - SHARED (one mailbox) vs NOT_SHARED (per-agent mailbox)
  - `pause()`, `resume()`, `cancel()` - Execution control
  - `create_group()` - Create LLMGroup with this team as leader

- **LLMGroup** - Multi-team coordination:
  - Group orchestrator is just an LLMAgent with specialized prompt
  - Automatic routing between teams based on task context
  - `run()` - Execute group with iterative team selection

- **RunResult** - Unified execution result:
  - `success`, `status` - Execution status
  - `output`, `final_output` - Agent outputs
  - `agents_called`, `iterations` - Execution metadata
  - `escalations` - Escalation records
  - `TeamSnapshot` - For pause/resume

### Removed

- **agent.py** - Old Agent base class (use typed agents via LLMTeam)
- **compat/** - Compatibility layer module (no backwards compatibility)
- **orchestration/team_orchestrator.py** - Replaced by LLMTeam internal flow
- **orchestration/group_orchestrator.py** - Replaced by LLMGroup
- **roles/orchestration.py**, **roles/pipeline_orch.py**, **roles/group_orch.py** - Legacy modules

### Changed

- `agents_invoked` renamed to `agents_called` in RunResult and TeamHandler
- `TeamResult` is now an alias for `RunResult`
- Registry uses `BaseAgent` from new agents package
- TeamHandler uses v4.0.0 API with `agents_called` field

---

## [3.0.0] - 2026-01-18

### Added

- **Agent Abstraction** - New `llmteam.agent` module:
  - `Agent` - Abstract base class for all agents with `process()` method
  - `AgentProtocol` - Protocol for duck typing agent implementations
  - `AgentState` - Input state container with data, metadata, history
  - `AgentResult` - Agent execution result with output, escalation support
  - `FunctionAgent` - Simple function-based agent wrapper
  - `LLMAgent` - LLM-powered agent with system prompt and tool support

- **Generic Registries** - New `llmteam.registry` module:
  - `BaseRegistry[T]` - Type-safe generic registry with callbacks
  - `AgentRegistry` - Specialized registry for agents with team back-references
  - `TeamRegistry` - Specialized registry for teams with health/availability tracking

- **LLMTeam Container** - New `llmteam.team` module:
  - `LLMTeam` - Primary team container class (replaces PipelineOrchestrator usage)
  - `TeamConfig` - Team configuration (strict_validation, max_iterations, timeout)
  - `TeamResult` - Team execution result with output, agents_invoked, escalations
  - `team.run(input_data, run_id)` - New execution API
  - `team.register_agent()` / `team.unregister_agent()` - Agent management
  - `team.on_escalation()` - Escalation handler registration
  - Health score and availability tracking

- **Team Orchestrator** - New `llmteam.orchestration.team_orchestrator`:
  - `TeamOrchestrator` - Orchestrates agent execution within a team
  - `OrchestrationMode` - SEQUENTIAL, ROUND_ROBIN, LLM_ROUTING modes
  - `OrchestrationStrategy` - Abstract strategy interface
  - `SequentialStrategy` - Execute agents in order
  - `RoundRobinStrategy` - Distribute work across agents
  - `LLMRoutingStrategy` - LLM-based intelligent routing
  - `OrchestrationContext` - Context for strategy decisions
  - `OrchestrationDecision` - Strategy output with next agent selection

- **Group Orchestrator (v3.0.0)** - New `llmteam.orchestration.group_orchestrator`:
  - `GroupOrchestrator` - Pure coordinator for multiple teams (not a router)
  - `GroupMetrics` - Team health, escalation counts, availability
  - Escalation handling with `handle_team_escalation()`
  - Team registration with `register_team()` / `unregister_team()`

- **Escalation Subsystem** - New `llmteam.escalation` module:
  - `EscalationLevel` - INFO, WARNING, CRITICAL, EMERGENCY severity levels
  - `EscalationAction` - ACKNOWLEDGE, RETRY, REDIRECT, ABORT, HUMAN_REVIEW
  - `Escalation` - Escalation event with source, reason, context
  - `EscalationDecision` - Handler response with action and target
  - `EscalationRecord` - Audit record of escalation handling
  - `EscalationHandler` - Abstract handler interface
  - `DefaultHandler` - Level-based default actions
  - `ThresholdHandler` - Count-based escalation triggering
  - `FunctionHandler` - Custom function wrapper
  - `ChainHandler` - Chain multiple handlers
  - `LevelFilterHandler` - Filter by escalation level

- **RuntimeContext Updates**:
  - `RuntimeContext.teams` - TeamRegistry for team management
  - `RuntimeContext.resolve_team()` / `get_team()` - Team resolution
  - `RuntimeContext.register_team()` - Team registration
  - `RuntimeContextFactory.register_team()` - Factory-level team registration
  - `StepContext.get_team()` - Team access from step context

- **Compatibility Layer** - New `llmteam.compat` module:
  - `LegacyPipelineOrchestrator` - v2.x API wrapper for LLMTeam
  - `LegacyAgentAdapter` - Adapt v2.x agents to v3.0.0 Agent interface
  - `PipelineOrchestratorAdapter` - Wrap v2.x orchestrator for team API
  - `create_team_from_orchestrator()` - Migrate v2.x to v3.0.0
  - `create_orchestrator_from_team()` - Create legacy wrapper
  - `Pipeline` - Deprecated alias for LLMTeam

### Changed

- **TeamHandler** - Updated to use new LLMTeam API:
  - Uses `team.run(input_data, run_id)` instead of `team.orchestrate()`
  - Returns `TeamResult` with metadata (iterations, agents_invoked, escalations)
  - Uses `ctx.get_team()` for team resolution
  - Includes `team_metadata` in handler output

- **Main exports** - Added v3.0.0 classes to `llmteam.__init__.py`:
  - Agent, AgentState, AgentResult, FunctionAgent, LLMAgent
  - BaseRegistry, AgentRegistry, TeamRegistry
  - LLMTeam, TeamConfig, TeamResult
  - TeamOrchestrator, OrchestrationMode, SequentialStrategy, etc.
  - EscalationLevel, EscalationAction, Escalation, handlers

### Deprecated

- `PipelineOrchestrator` - Use `LLMTeam` with `TeamOrchestrator` instead
- `Pipeline` - Deprecated alias, use `LLMTeam`
- `team.orchestrate()` - Use `team.run()` instead

### Migration Guide

**v2.x code:**
```python
from llmteam.roles import PipelineOrchestrator

orchestrator = PipelineOrchestrator(pipeline_id="support")
orchestrator.register_agent("triage", triage_agent)
result = await orchestrator.orchestrate(run_id, input_data)
```

**v3.0.0 code:**
```python
from llmteam import LLMTeam, Agent

class TriageAgent(Agent):
    async def process(self, state):
        return AgentResult(output={"category": "billing"})

team = LLMTeam(team_id="support")
team.register_agent(TriageAgent("triage"))
result = await team.run(input_data, run_id=run_id)
```

**Using compatibility layer:**
```python
from llmteam.compat import LegacyPipelineOrchestrator

# Works like v2.x
orchestrator = LegacyPipelineOrchestrator(pipeline_id="support")
orchestrator.register_agent("triage", triage_agent)
result = await orchestrator.orchestrate(run_id, input_data)
```

## [2.3.0] - 2025-01-18

### Added
- **Team Contracts** - Formal interface definitions in `llmteam.roles`:
  - `TeamContract` - Input/output contract with TypedPort validation
  - `ValidationResult` - Validation feedback with errors list
  - `ContractValidationError` - Exception for validation failures
  - Contract support in `PipelineOrchestrator` with `strict_validation` mode

- **Team Handler** - Execute agent teams as Canvas steps:
  - `TeamHandler` - New step type `team` for invoking PipelineOrchestrator
  - Input/output mapping for flexible data transformation
  - Registered in StepCatalog (now 12 built-in types)

- **Escalation System** - Structured escalation handling in `GroupOrchestrator`:
  - `EscalationLevel` - INFO, WARNING, CRITICAL, EMERGENCY levels
  - `EscalationAction` - ACKNOWLEDGE, RETRY, REDIRECT, ABORT, HUMAN_REVIEW
  - `Escalation` - Escalation event with context and metadata
  - `EscalationDecision` - Handler response with action and target
  - `handle_escalation()` - Process escalations with default or custom handlers
  - `get_escalation_history()` - Query escalations with pipeline/level filters
  - `collect_metrics()` - Health score and escalation statistics

- **Secure Data Bus** - Event-driven communication in `llmteam.transport`:
  - `SecureBus` - Publish/subscribe pattern for events
  - `BusEventType` - RUN_STARTED, RUN_COMPLETED, RUN_FAILED, STEP_*, ESCALATION
  - `BusEvent` - Event model with trace_id, process_run_id, data
  - `DataMode` - REFS_ONLY (redacted) vs FULL_PAYLOAD modes
  - `ControlCommand` - PAUSE, RESUME, CANCEL for run management
  - Event buffering with `get_events()` and `clear_buffer()`
  - Audit logging with `get_audit_log()`
  - Convenience methods: `run_started()`, `run_completed()`, `step_started()`, etc.

- **New Tests** - 65 new tests for v2.3.0 components:
  - `tests/roles/test_contract.py` - TeamContract validation tests
  - `tests/roles/test_escalation.py` - Escalation handling tests
  - `tests/canvas/test_team_handler.py` - TeamHandler execution tests
  - `tests/transport/test_bus.py` - SecureBus messaging tests

### Changed
- `GroupOrchestrator` role changed from Router to Coordinator/Supervisor
- Routing between teams now defined in Canvas, not GroupOrchestrator

### Deprecated
- `GroupOrchestrator.orchestrate()` - Use Canvas with TeamHandler for team routing

## [2.2.1] - 2025-01-18

### Added
- **Widget Protocol** - KorpOS UI integration in `llmteam.canvas.widget`:
  - `Widget` protocol with `render()` and `handle_intent()` methods
  - `WidgetState` - IDLE, LOADING, ERROR, SUCCESS states
  - `WidgetIntent` - User interaction intents

- **RAG Handler** - Retrieval-augmented generation in `llmteam.canvas.handlers`:
  - `RAGHandler` - Native and proxy modes for RAG workflows
  - Integration with vector stores and retrievers

- **Context Provider** - Abstraction for context sources:
  - `ContextProvider` - Native and proxy mode support
  - Pluggable context retrieval for agents

- **End-to-End Tests** - `tests/e2e/` for workflow execution testing

### Changed
- Refactored handlers for better maintainability
- Streamlined event transports (Redis, Kafka)

## [2.2.0] - 2025-01-18

### Added
- **New Step Handlers**:
  - `SubworkflowHandler` - Execute nested workflow segments with input/output mapping
  - `SwitchHandler` - Multi-way branching based on value matching (like switch/case)

- **Event Transports**:
  - `RedisTransport` - Redis Pub/Sub for event streaming
  - `KafkaTransport` - Apache Kafka for enterprise event streaming

- **Transform Enhancements**:
  - JSONPath support via `jsonpath-ng` (expressions starting with `$` or `@`)
  - New functions: `first()`, `last()`, `flatten()`, `unique()`, `sort()`

- **API Health Checks** - `llmteam.api.health`:
  - `HealthChecker` class with component health checks
  - `ComponentHealth` and `HealthStatus` models
  - Pre-built checks for database, Redis, memory, and disk

- **CLI Commands**:
  - `llmteam providers` - List available LLM providers with status
  - `llmteam check` - Comprehensive segment validation with JSON output

- **Examples Directory**:
  - `quickstart/` - Simple workflow example
  - `conditional_flow/` - Branching with switch/condition
  - `custom_handler/` - Creating and registering custom handlers
  - Additional example directories for enterprise patterns

- **Contributing Guide** - CONTRIBUTING.md with development guidelines

- New optional dependencies: `jsonpath`, `kafka`

### Changed
- Updated `version` command to show more installed components
- Extended handlers package with SubworkflowHandler and SwitchHandler

## [2.1.0] - 2025-01-17

### Added
- **Extended LLM Providers** - New providers in `llmteam.providers`:
  - `VertexAIProvider` - Google Vertex AI with Gemini models (gemini-1.5-pro, gemini-1.5-flash)
  - `OllamaProvider` - Local LLMs via Ollama (llama2, mistral, codellama, etc.)
  - `LiteLLMProvider` - Unified API for 100+ LLM backends

- **Secrets Management** - Enterprise secrets handling in `llmteam.secrets`:
  - `SecretsManager` - High-level secrets access with fallback support
  - `CachingSecretsManager` - Secrets caching with TTL
  - `EnvSecretsProvider` - Environment variables backend
  - `VaultProvider` - HashiCorp Vault with AppRole/K8s auth
  - `AWSSecretsProvider` - AWS Secrets Manager with assume role
  - `AzureKeyVaultProvider` - Azure Key Vault with managed identity

- **GraphQL Client** - Full-featured GraphQL support in `llmteam.clients`:
  - `GraphQLClient` - Async client with retry and circuit breaker
  - `GraphQLSubscription` - WebSocket subscription support
  - Query caching with configurable TTL
  - Error handling with structured GraphQL errors

- **gRPC Client** - Enterprise gRPC support in `llmteam.clients`:
  - `GRPCClient` - Async gRPC with channel management
  - Secure (TLS) and insecure connections
  - Retry with exponential backoff
  - Health check protocol support
  - Server streaming support

- New optional dependencies: `vault`, `azure-secrets`, `vertex`, `litellm`, `graphql`, `grpc`

### Changed
- Extended `clients` module to include GraphQL and gRPC clients
- Updated `providers` module to export all 7 providers

## [2.0.4] - 2025-01-17

### Added
- **Middleware System** - Interceptors for step execution in `llmteam.middleware`:
  - `MiddlewareStack` - Composable middleware chain
  - `LoggingMiddleware` - Request/response logging
  - `TimingMiddleware` - Execution timing with slow step detection
  - `RetryMiddleware` - Configurable retry with exponential backoff
  - `CachingMiddleware` - Result caching with TTL
  - `RateLimitMiddleware` - Token bucket rate limiting
  - `AuthMiddleware` - Authentication validation
  - `ValidationMiddleware` - Input/output validation

- **OpenTelemetry Tracing** - Distributed tracing in `llmteam.observability`:
  - `TracingConfig` - Configure OTLP, Jaeger, Zipkin exporters
  - `TracingMiddleware` - Automatic span creation for steps
  - `SpanAttributes` - Standardized attribute names
  - `trace_segment()`, `trace_llm_call()` - Context managers

- **Authentication Module** - Enterprise auth in `llmteam.auth`:
  - `OIDCProvider` - OpenID Connect with PKCE support
  - `JWTValidator` - Token validation with JWKS
  - `APIKeyValidator` - API key management with hashing
  - `AuthenticationMiddleware`, `AuthorizationMiddleware`
  - `RBACConfig` - Role-based access control with inheritance

- **New Step Handlers** - Flow control handlers:
  - `LoopHandler` - For-each, while, until, and range loops
  - `ErrorHandler` - Catch, fallback, retry, compensate modes
  - `TryCatchHandler` - Structured try-catch-finally patterns

- **HTTP Client** - Resilient REST client in `llmteam.clients`:
  - `HTTPClient` - Async HTTP with retry and circuit breaker
  - `RetryConfig` - Configurable retry strategies
  - `ExponentialBackoff`, `LinearBackoff`, `ConstantBackoff`
  - `CircuitBreaker` - Fault tolerance with automatic recovery

- New optional dependencies: `tracing`, `auth`

### Changed
- Updated handlers package to include LoopHandler, ErrorHandler, TryCatchHandler

## [2.0.3] - 2025-01-17

### Added
- **Step Handlers** - Built-in handlers for all step types
  - `LLMAgentHandler` - LLM completion with prompt templating and variable substitution
  - `HTTPActionHandler` - HTTP requests (GET/POST/PUT/PATCH/DELETE) with headers and timeout
  - `TransformHandler` - Data transformation with expressions, field mappings, and functions
  - `ConditionHandler` - Conditional branching with comparison and logical operators
  - `ParallelSplitHandler` - Fan-out to parallel execution branches
  - `ParallelJoinHandler` - Merge parallel results (all/any/first strategies)
- Parallel execution support in SegmentRunner with `asyncio.gather`
- Handler tests (62 new tests for handlers)
- **JSON Schema Export** - `SegmentDefinition.json_schema()` for validation and IDE support
- **LLM Providers** - Ready-to-use providers in `llmteam.providers`:
  - `OpenAIProvider` - GPT-4, GPT-4o, GPT-4o-mini support
  - `AnthropicProvider` - Claude 3.5 Sonnet, Haiku, Opus support
  - `AzureOpenAIProvider` - Azure-hosted OpenAI models
  - `BedrockProvider` - AWS Bedrock (Claude, Llama, Titan)
- **Testing Utilities** - New `llmteam.testing` module:
  - `MockLLMProvider` - Deterministic responses for testing
  - `MockHTTPClient`, `MockStore`, `MockSecretsProvider`
  - `SegmentTestRunner` - Isolated segment execution
  - `StepTestHarness` - Handler unit testing
- **Event Transports** - New `llmteam.events.transports`:
  - `WebSocketTransport` - Bidirectional event streaming
  - `SSETransport` - Server-Sent Events for real-time updates
- **Examples** - New examples directory:
  - `quickstart/` - 5-minute getting started
  - `simple_workflow/` - Conditional branching example
  - `fastapi_server/` - REST API integration
- **Type Hints** - Added `py.typed` marker for PEP 561 compliance
- New optional dependencies: `providers`, `aws`, `websockets`

### Changed
- Moved handlers to `canvas/handlers/` package
- Repository reorganization:
  - Specs moved to `docs/specs/`
  - Testing docs moved to `docs/testing/`
  - Implementation summaries moved to `docs/`
- Updated CLAUDE.md with new structure and handler documentation

### Fixed
- TransformHandler: Returns default value when expression evaluates to None
- ConditionHandler: Logical operators (and/or) now have correct precedence

## [2.0.2] - 2025-01-17

### Fixed
- README.md: Fixed package name `pip install llmteam` → `pip install llmteam-ai`
- Reorganized documentation to highlight v2.0.0 Canvas Integration features
- Moved v1.7.0-v1.9.0 features to "Previous Versions" section

### Added
- CLI usage examples in README
- Step types table in documentation

## [2.0.1] - 2025-01-17

### Changed
- Updated root README.md with v2.0.0 features
- Added badges for PyPI, Python version, License

### Added
- CLI tool with click framework (run, validate, catalog, serve commands)
- Makefile for development workflow
- Docker multi-stage build and docker-compose
- GitHub Actions CI/CD with matrix testing
- JSON Schema validation for canvas segments
- Structured logging with structlog (stdlib fallback)
- REST API module with FastAPI and WebSocket support
- RuntimeContextFactory for easier context creation

## [2.0.0] - 2025-01-17

### Added
- **Canvas Integration** - Visual workflow execution system
  - `canvas/models.py` - Segment JSON contract (SegmentDefinition, StepDefinition, EdgeDefinition, PortDefinition)
  - `canvas/catalog.py` - Step type catalog with 7 built-in types
  - `canvas/runner.py` - SegmentRunner execution engine
  - `canvas/handlers.py` - HumanTaskHandler for human-in-the-loop steps
  - `canvas/exceptions.py` - Canvas-specific exceptions
- **Runtime Context** - Unified resource injection
  - `runtime/context.py` - RuntimeContext and StepContext
  - `runtime/manager.py` - RuntimeContextManager with registries for Store/Client/LLM/Secrets
- **Worktrail Events** - UI event streaming
  - `events/emitter.py` - EventEmitter for step/segment lifecycle
  - `events/models.py` - WorktrailEvent and EventType definitions
  - `events/store.py` - EventStore for persistence

### Changed
- Unified execution model through SegmentRunner
- Step handlers receive StepContext with dependency injection

## [1.9.0] - 2025-01-16

### Added
- **Workflow Runtime** - Pause/resume and external integrations
  - `actions/` - ActionExecutor for external API/webhook calls
  - `human/` - HumanInteractionManager (approval, chat, escalation)
  - `persistence/` - SnapshotManager for pipeline state persistence

## [1.8.0] - 2025-01-15

### Added
- **Orchestration Intelligence** - Pipeline and group orchestration
  - `context/hierarchical.py` - HierarchicalContext with parent-child propagation
  - `licensing/` - LicenseManager with tier-based limits
  - `execution/` - PipelineExecutor with parallel execution
  - `roles/` - PipelineOrchestrator, GroupOrchestrator, ProcessMiningEngine

## [1.7.0] - 2025-01-14

### Added
- **Security Foundation** - Multi-tenant isolation and audit
  - `tenancy/` - TenantManager, TenantContext, TenantIsolatedStore
  - `audit/` - AuditTrail with SHA-256 checksum chain
  - `context/` - SecureAgentContext with SealedData
  - `ratelimit/` - RateLimiter, CircuitBreaker, RateLimitedExecutor

### Changed
- Renamed project from `llm-pipeline-smtrk` to `llmteam`

## [1.6.0] - 2025-01-10

### Added
- Initial public release
- Core pipeline execution framework
- Basic agent context management

[3.0.0]: https://github.com/llmteamai-rgb/LLMTeam/compare/v2.3.0...v3.0.0
[2.3.0]: https://github.com/llmteamai-rgb/LLMTeam/compare/v2.2.1...v2.3.0
[2.2.1]: https://github.com/llmteamai-rgb/LLMTeam/compare/v2.2.0...v2.2.1
[2.2.0]: https://github.com/llmteamai-rgb/LLMTeam/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/llmteamai-rgb/LLMTeam/compare/v2.0.4...v2.1.0
[2.0.4]: https://github.com/llmteamai-rgb/LLMTeam/compare/v2.0.3...v2.0.4
[2.0.3]: https://github.com/llmteamai-rgb/LLMTeam/compare/v2.0.2...v2.0.3
[2.0.2]: https://github.com/llmteamai-rgb/LLMTeam/compare/v2.0.1...v2.0.2
[2.0.1]: https://github.com/llmteamai-rgb/LLMTeam/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/llmteamai-rgb/LLMTeam/compare/v1.9.0...v2.0.0
[1.9.0]: https://github.com/llmteamai-rgb/LLMTeam/compare/v1.8.0...v1.9.0
[1.8.0]: https://github.com/llmteamai-rgb/LLMTeam/compare/v1.7.0...v1.8.0
[1.7.0]: https://github.com/llmteamai-rgb/LLMTeam/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/llmteamai-rgb/LLMTeam/releases/tag/v1.6.0
