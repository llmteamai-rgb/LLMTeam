# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
