"""
Structured Logging for LLMTeam.

This module provides structured logging using structlog when available,
with a fallback to standard library logging.

Usage:
    from llmteam.observability import configure_logging, get_logger

    # Configure at application startup
    configure_logging(level="INFO", format="json")

    # Get logger in modules
    logger = get_logger("llmteam.runner")
    logger.info("segment_started", segment_id="seg1", tenant_id="t1")
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TextIO

_configured = False
_use_structlog = False


class LogFormat(Enum):
    """Output format for logs."""

    JSON = "json"  # JSON lines, for production/log aggregators
    CONSOLE = "console"  # Colored, human-readable, for development
    TEXT = "text"  # Plain text without colors


@dataclass
class LogConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: LogFormat = LogFormat.JSON
    output: TextIO = field(default_factory=lambda: sys.stderr)

    # Additional context added to all log entries
    service_name: str = "llmteam"
    service_version: str = ""

    # Filtering
    include_timestamps: bool = True
    include_caller: bool = False  # Add file:line info

    # Performance
    cache_logger_on_first_use: bool = True

    @classmethod
    def from_env(cls) -> "LogConfig":
        """Create config from environment variables."""
        level = os.environ.get("LLMTEAM_LOG_LEVEL", "INFO").upper()
        format_str = os.environ.get("LLMTEAM_LOG_FORMAT", "json").lower()

        try:
            log_format = LogFormat(format_str)
        except ValueError:
            log_format = LogFormat.JSON

        return cls(
            level=level,
            format=log_format,
            service_name=os.environ.get("LLMTEAM_SERVICE_NAME", "llmteam"),
            service_version=os.environ.get("LLMTEAM_VERSION", ""),
        )


def configure_logging(
    level: str = "INFO",
    format: str | LogFormat = "json",
    config: LogConfig | None = None,
) -> None:
    """
    Configure structured logging for LLMTeam.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Output format ("json", "console", "text") or LogFormat enum
        config: Full LogConfig object (overrides level and format)

    Examples:
        # Simple configuration
        configure_logging(level="DEBUG", format="console")

        # From environment variables
        configure_logging(config=LogConfig.from_env())

        # Full control
        config = LogConfig(
            level="INFO",
            format=LogFormat.JSON,
            service_name="my-app",
        )
        configure_logging(config=config)
    """
    global _configured, _use_structlog

    if config is None:
        if isinstance(format, str):
            format = LogFormat(format.lower())
        config = LogConfig(level=level, format=format)

    # Try to use structlog if available
    try:
        _configure_structlog(config)
        _use_structlog = True
    except ImportError:
        _configure_stdlib(config)
        _use_structlog = False

    _configured = True


def _configure_structlog(config: LogConfig) -> None:
    """Configure structlog-based logging."""
    import structlog
    from structlog.stdlib import (
        add_log_level,
        add_logger_name,
        filter_by_level,
        PositionalArgumentsFormatter,
    )
    from structlog.processors import (
        TimeStamper,
        StackInfoRenderer,
        UnicodeDecoder,
        JSONRenderer,
    )

    # Build processor chain
    processors: list[Callable] = [
        filter_by_level,
        add_log_level,
        add_logger_name,
        PositionalArgumentsFormatter(),
    ]

    if config.include_timestamps:
        processors.append(TimeStamper(fmt="iso"))

    if config.include_caller:
        processors.append(StackInfoRenderer())

    processors.append(UnicodeDecoder())

    # Add service context
    if config.service_name or config.service_version:

        def add_service_context(
            logger: Any, method_name: str, event_dict: dict[str, Any]
        ) -> dict[str, Any]:
            if config.service_name:
                event_dict["service"] = config.service_name
            if config.service_version:
                event_dict["version"] = config.service_version
            return event_dict

        processors.append(add_service_context)

    # Add renderer based on format
    if config.format == LogFormat.JSON:
        processors.append(JSONRenderer())
    elif config.format == LogFormat.CONSOLE:
        processors.append(structlog.dev.ConsoleRenderer())
    else:  # TEXT
        processors.append(structlog.dev.ConsoleRenderer(colors=False))

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=config.cache_logger_on_first_use,
    )

    # Configure stdlib logging (structlog uses it underneath)
    logging.basicConfig(
        format="%(message)s",
        stream=config.output,
        level=getattr(logging, config.level),
    )


def _configure_stdlib(config: LogConfig) -> None:
    """Configure standard library logging (fallback)."""
    # Create formatter based on format
    if config.format == LogFormat.JSON:
        formatter = _JsonFormatter(config)
    else:
        formatter = _TextFormatter(config)

    # Configure root logger
    handler = logging.StreamHandler(config.output)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(getattr(logging, config.level))
    root.handlers.clear()
    root.addHandler(handler)


class _JsonFormatter(logging.Formatter):
    """JSON formatter for stdlib logging."""

    def __init__(self, config: LogConfig):
        super().__init__()
        self.config = config

    def format(self, record: logging.LogRecord) -> str:
        import json
        from datetime import datetime, timezone

        data: dict[str, Any] = {
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }

        if self.config.include_timestamps:
            data["timestamp"] = datetime.now(timezone.utc).isoformat()

        if self.config.service_name:
            data["service"] = self.config.service_name

        if self.config.service_version:
            data["version"] = self.config.service_version

        if self.config.include_caller:
            data["caller"] = f"{record.filename}:{record.lineno}"

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in (
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "exc_info",
                "exc_text",
                "thread",
                "threadName",
                "taskName",
                "message",
            ):
                data[key] = value

        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)

        return json.dumps(data, default=str)


class _TextFormatter(logging.Formatter):
    """Text formatter for stdlib logging."""

    def __init__(self, config: LogConfig):
        if config.include_timestamps:
            fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        else:
            fmt = "[%(levelname)s] %(name)s: %(message)s"
        super().__init__(fmt=fmt, datefmt="%Y-%m-%dT%H:%M:%S%z")


def get_logger(name: str | None = None) -> "StructuredLogger":
    """
    Get a structured logger.

    Args:
        name: Logger name (e.g., "llmteam.runner", "llmteam.engine")

    Returns:
        StructuredLogger instance
    """
    global _configured

    if not _configured:
        configure_logging(config=LogConfig.from_env())

    return StructuredLogger(name or "llmteam")


class StructuredLogger:
    """
    Structured logger that works with both structlog and stdlib.

    Provides a consistent API regardless of backend.

    Usage:
        logger = get_logger("llmteam.runner")

        # Simple logging
        logger.info("Starting segment")

        # Structured logging with context
        logger.info("segment_started",
            segment_id="seg1",
            tenant_id="tenant1",
            step_count=5,
        )

        # Binding context for multiple calls
        ctx_logger = logger.bind(segment_id="seg1")
        ctx_logger.info("step_started", step_id="step1")
        ctx_logger.info("step_completed", step_id="step1")
    """

    def __init__(self, name: str, context: dict[str, Any] | None = None):
        self._name = name
        self._context = context or {}
        self._logger: Any = None

    def _get_logger(self) -> Any:
        """Lazy load the underlying logger."""
        if self._logger is None:
            if _use_structlog:
                import structlog

                self._logger = structlog.get_logger(self._name)
                if self._context:
                    self._logger = self._logger.bind(**self._context)
            else:
                self._logger = logging.getLogger(self._name)
        return self._logger

    def bind(self, **kwargs: Any) -> "StructuredLogger":
        """
        Return a new logger with bound context.

        The context will be included in all log entries.
        """
        new_context = {**self._context, **kwargs}
        return StructuredLogger(self._name, new_context)

    def unbind(self, *keys: str) -> "StructuredLogger":
        """Return a new logger with specified keys removed from context."""
        new_context = {k: v for k, v in self._context.items() if k not in keys}
        return StructuredLogger(self._name, new_context)

    def _log(self, level: str, event: str, **kwargs: Any) -> None:
        """Internal log method."""
        logger = self._get_logger()

        if _use_structlog:
            log_method = getattr(logger, level)
            log_method(event, **kwargs)
        else:
            # Stdlib logging
            extra = {**self._context, **kwargs}
            log_method = getattr(logger, level)
            log_method(event, extra=extra)

    def debug(self, event: str, **kwargs: Any) -> None:
        """Log at DEBUG level."""
        self._log("debug", event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        """Log at INFO level."""
        self._log("info", event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        """Log at WARNING level."""
        self._log("warning", event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        """Log at ERROR level."""
        self._log("error", event, **kwargs)

    def critical(self, event: str, **kwargs: Any) -> None:
        """Log at CRITICAL level."""
        self._log("critical", event, **kwargs)

    def exception(self, event: str, **kwargs: Any) -> None:
        """Log at ERROR level with exception info."""
        logger = self._get_logger()

        if _use_structlog:
            logger.exception(event, **kwargs)
        else:
            extra = {**self._context, **kwargs}
            logger.exception(event, extra=extra)
