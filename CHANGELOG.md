# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.1] - 2026-01-18

### Added
- **E2E Testing**:
    - `tests/e2e/test_workflow_execution.py`: Comprehensive end-to-end tests for workflow execution.
    - Tests for full functional cycle, parallel execution, human task handling, and error scenarios.
- **API Security**:
    - Rate limiting support with `slowapi` integration in `api/app.py`.
    - Configurable rate limits with `enable_rate_limit` and `rate_limit` parameters.
- **Widget Protocol** (`canvas/widget.py`):
    - `WidgetComponent` protocol with `render()` and `handle_intent()` methods for KorpOS UI integration.
    - Built-in widgets: `TextInputWidget`, `ButtonWidget`, `ApprovalWidget`.
- **Context Provider** (`context/provider.py`):
    - `ContextProvider` abstraction supporting native and proxy modes.
    - `NativeContextProvider` for local context storage.
    - `ProxyContextProvider` for delegating to external RAG services (VCR/KorpOS).
    - `CompositeContextProvider` for combining multiple providers.
- **RAG Handler** (`canvas/handlers/rag_handler.py`):
    - `RAGHandler` step type for retrieval-augmented generation.
    - Supports native mode (local context) and proxy mode (external service).
    - `RAGQueryBuilder` with fluent interface for query construction.
- **Documentation**:
    - MkDocs configuration (`mkdocs.yml`) with Material theme.
    - Documentation structure in `docs/` (index, installation, quickstart).
- **Development Environment**:
    - `docker-compose.dev.yml` for local development with PostgreSQL, Redis, Kafka, MinIO, Vault.

### Security
- **ConditionHandler Input Sanitization**:
    - Added `FORBIDDEN_PATTERNS` regex to block injection attacks.
    - Blocks `eval`, `exec`, `import`, `__builtins__`, `os.`, `sys.`, and other dangerous patterns.
    - Maximum expression length limit (1000 chars).

## [2.2.0] - 2026-01-17

### Added
- **Canvas Handlers**:
    - `SubworkflowHandler`: Support for nested workflows and reusable subflows.
    - `SwitchHandler`: Logic branching support (value match and expressions).
- **Event Transports**:
    - `RedisTransport`: Pub/Sub event streaming using Redis.
    - `KafkaTransport`: Enterprise-grade event streaming with aiokafka.
- **Transformations**:
    - Added JSONPath support to `TransformHandler` (using `jsonpath-ng`).
- **Documentation**:
    - Added `docs/` directory with Sphinx configuration.
    - Added `examples/` directory with Quickstart and advanced examples.
    - Added `CONTRIBUTING.md` guidelines.
- **API & CLI**:
    - Added `HealthChecker` and API endpoints.
    - added `check`, `validate`, `providers`, `version`, `catalog` CLI commands.
    - Added Type Stubs (`.pyi`) for better IDE support.

### Changed
- Updated `CLAUDE.md` and `README.md` to reflect new version and structure.
- Improved validation logic in `SegmentDefinition`.

## [2.1.0] - 2025-12-15
- Initial Open Core release structure.
- Basic Canvas and Runtime implementation.
