# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[2.0.2]: https://github.com/llmteamai-rgb/LLMTeam/compare/v2.0.1...v2.0.2
[2.0.1]: https://github.com/llmteamai-rgb/LLMTeam/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/llmteamai-rgb/LLMTeam/compare/v1.9.0...v2.0.0
[1.9.0]: https://github.com/llmteamai-rgb/LLMTeam/compare/v1.8.0...v1.9.0
[1.8.0]: https://github.com/llmteamai-rgb/LLMTeam/compare/v1.7.0...v1.8.0
[1.7.0]: https://github.com/llmteamai-rgb/LLMTeam/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/llmteamai-rgb/LLMTeam/releases/tag/v1.6.0
