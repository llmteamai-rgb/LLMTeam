"""
Observability Module.

This module provides structured logging, tracing, and metrics for LLMTeam.

Components:
- logging: Structured logging configuration with structlog
- tracing: OpenTelemetry distributed tracing
- metrics: Prometheus metrics (future)
"""

from llmteam.observability.logging import (
    configure_logging,
    get_logger,
    LogConfig,
    LogFormat,
)

from llmteam.observability.tracing import (
    TracingConfig,
    setup_tracing,
    get_tracer,
    TracingMiddleware,
    SpanAttributes,
    trace_segment,
    trace_llm_call,
)

__all__ = [
    # Logging
    "configure_logging",
    "get_logger",
    "LogConfig",
    "LogFormat",
    # Tracing
    "TracingConfig",
    "setup_tracing",
    "get_tracer",
    "TracingMiddleware",
    "SpanAttributes",
    "trace_segment",
    "trace_llm_call",
]
