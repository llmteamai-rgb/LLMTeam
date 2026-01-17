"""
Observability Module.

This module provides structured logging and metrics for LLMTeam.

Components:
- logging: Structured logging configuration with structlog
- metrics: Prometheus metrics (future)
"""

from llmteam.observability.logging import (
    configure_logging,
    get_logger,
    LogConfig,
    LogFormat,
)

__all__ = [
    "configure_logging",
    "get_logger",
    "LogConfig",
    "LogFormat",
]
