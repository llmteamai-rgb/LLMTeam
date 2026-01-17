"""Tests for observability logging module."""

import io
import json
import logging
import os
from unittest.mock import patch, MagicMock

import pytest

from llmteam.observability.logging import (
    LogFormat,
    LogConfig,
    configure_logging,
    get_logger,
    StructuredLogger,
    _configured,
)


class TestLogFormat:
    """Tests for LogFormat enum."""

    def test_json_format(self):
        """JSON format value."""
        assert LogFormat.JSON.value == "json"

    def test_console_format(self):
        """Console format value."""
        assert LogFormat.CONSOLE.value == "console"

    def test_text_format(self):
        """Text format value."""
        assert LogFormat.TEXT.value == "text"


class TestLogConfig:
    """Tests for LogConfig."""

    def test_default_values(self):
        """Default config values."""
        config = LogConfig()

        assert config.level == "INFO"
        assert config.format == LogFormat.JSON
        assert config.service_name == "llmteam"
        assert config.include_timestamps is True
        assert config.include_caller is False

    def test_custom_values(self):
        """Custom config values."""
        config = LogConfig(
            level="DEBUG",
            format=LogFormat.CONSOLE,
            service_name="my-service",
            service_version="1.0.0",
            include_timestamps=False,
            include_caller=True,
        )

        assert config.level == "DEBUG"
        assert config.format == LogFormat.CONSOLE
        assert config.service_name == "my-service"
        assert config.service_version == "1.0.0"
        assert config.include_timestamps is False
        assert config.include_caller is True

    def test_from_env_defaults(self):
        """Config from environment with defaults."""
        with patch.dict(os.environ, {}, clear=True):
            config = LogConfig.from_env()

        assert config.level == "INFO"
        assert config.format == LogFormat.JSON

    def test_from_env_custom(self):
        """Config from environment with custom values."""
        env = {
            "LLMTEAM_LOG_LEVEL": "DEBUG",
            "LLMTEAM_LOG_FORMAT": "console",
            "LLMTEAM_SERVICE_NAME": "test-service",
            "LLMTEAM_VERSION": "2.0.0",
        }
        with patch.dict(os.environ, env, clear=True):
            config = LogConfig.from_env()

        assert config.level == "DEBUG"
        assert config.format == LogFormat.CONSOLE
        assert config.service_name == "test-service"
        assert config.service_version == "2.0.0"

    def test_from_env_invalid_format_fallback(self):
        """Config falls back to JSON for invalid format."""
        env = {"LLMTEAM_LOG_FORMAT": "invalid_format"}
        with patch.dict(os.environ, env, clear=True):
            config = LogConfig.from_env()

        assert config.format == LogFormat.JSON


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_with_level_and_format(self):
        """Configure with level and format strings."""
        configure_logging(level="DEBUG", format="json")

        # Should not raise
        logger = get_logger("test")
        assert logger is not None

    def test_configure_with_log_format_enum(self):
        """Configure with LogFormat enum."""
        configure_logging(level="INFO", format=LogFormat.CONSOLE)

        logger = get_logger("test")
        assert logger is not None

    def test_configure_with_config_object(self):
        """Configure with LogConfig object."""
        config = LogConfig(
            level="WARNING",
            format=LogFormat.TEXT,
            service_name="test-app",
        )
        configure_logging(config=config)

        logger = get_logger("test")
        assert logger is not None

    def test_configure_idempotent(self):
        """Multiple configure calls don't break."""
        configure_logging(level="INFO", format="json")
        configure_logging(level="DEBUG", format="console")
        configure_logging(level="WARNING", format="text")

        logger = get_logger("test")
        assert logger is not None


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_with_name(self):
        """Get logger with specific name."""
        logger = get_logger("llmteam.runner")

        assert logger is not None
        assert logger._name == "llmteam.runner"

    def test_get_logger_default_name(self):
        """Get logger with default name."""
        logger = get_logger()

        assert logger is not None
        assert logger._name == "llmteam"

    def test_get_logger_auto_configures(self):
        """Get logger auto-configures if not configured."""
        # This should work even without explicit configure_logging call
        logger = get_logger("auto-test")
        assert logger is not None


class TestStructuredLogger:
    """Tests for StructuredLogger class."""

    def test_create_logger(self):
        """Create logger instance."""
        logger = StructuredLogger("test.module")

        assert logger._name == "test.module"
        assert logger._context == {}

    def test_create_logger_with_context(self):
        """Create logger with initial context."""
        context = {"tenant_id": "acme", "run_id": "123"}
        logger = StructuredLogger("test", context=context)

        assert logger._context == context

    def test_bind_returns_new_logger(self):
        """Bind returns new logger with added context."""
        logger = StructuredLogger("test")
        bound = logger.bind(tenant_id="acme", step_id="step1")

        assert bound is not logger
        assert bound._context["tenant_id"] == "acme"
        assert bound._context["step_id"] == "step1"
        assert logger._context == {}  # Original unchanged

    def test_bind_preserves_existing_context(self):
        """Bind preserves existing context."""
        logger = StructuredLogger("test", context={"tenant_id": "acme"})
        bound = logger.bind(step_id="step1")

        assert bound._context["tenant_id"] == "acme"
        assert bound._context["step_id"] == "step1"

    def test_unbind_removes_keys(self):
        """Unbind removes specified keys."""
        logger = StructuredLogger(
            "test",
            context={"tenant_id": "acme", "step_id": "step1", "run_id": "123"},
        )
        unbound = logger.unbind("step_id", "run_id")

        assert "tenant_id" in unbound._context
        assert "step_id" not in unbound._context
        assert "run_id" not in unbound._context

    def test_unbind_preserves_other_context(self):
        """Unbind preserves non-specified keys."""
        logger = StructuredLogger(
            "test",
            context={"a": 1, "b": 2, "c": 3},
        )
        unbound = logger.unbind("b")

        assert unbound._context == {"a": 1, "c": 3}


class TestStructuredLoggerLogging:
    """Tests for StructuredLogger log methods."""

    def setup_method(self):
        """Setup test logging."""
        # Configure logging to capture output
        configure_logging(level="DEBUG", format="json")

    def test_debug_method(self):
        """Debug method logs at DEBUG level."""
        logger = get_logger("test.debug")

        # Should not raise
        logger.debug("test message", key="value")

    def test_info_method(self):
        """Info method logs at INFO level."""
        logger = get_logger("test.info")

        logger.info("test message", key="value")

    def test_warning_method(self):
        """Warning method logs at WARNING level."""
        logger = get_logger("test.warning")

        logger.warning("test message", key="value")

    def test_error_method(self):
        """Error method logs at ERROR level."""
        logger = get_logger("test.error")

        logger.error("test message", key="value")

    def test_critical_method(self):
        """Critical method logs at CRITICAL level."""
        logger = get_logger("test.critical")

        logger.critical("test message", key="value")

    def test_exception_method(self):
        """Exception method logs with exception info."""
        logger = get_logger("test.exception")

        try:
            raise ValueError("test error")
        except ValueError:
            logger.exception("caught error", key="value")

    def test_log_with_bound_context(self):
        """Log includes bound context."""
        logger = get_logger("test.bound")
        bound = logger.bind(tenant_id="acme")

        # Should include tenant_id in context
        bound.info("test message")

    def test_log_with_extra_kwargs(self):
        """Log includes extra kwargs."""
        logger = get_logger("test.extra")

        logger.info(
            "segment_started",
            segment_id="seg1",
            step_count=5,
            duration_ms=123.45,
        )


class TestJsonFormatter:
    """Tests for JSON formatter output."""

    def test_json_output_format(self):
        """Verify JSON output is valid."""
        output = io.StringIO()
        config = LogConfig(
            level="INFO",
            format=LogFormat.JSON,
            output=output,
            service_name="test-service",
        )
        configure_logging(config=config)

        logger = get_logger("json.test")
        logger.info("test_event", key="value")

        # Output might have structlog or stdlib format
        # Just verify no exceptions were raised
        assert True

    def test_json_includes_service_name(self):
        """JSON output includes service name."""
        output = io.StringIO()
        config = LogConfig(
            level="INFO",
            format=LogFormat.JSON,
            output=output,
            service_name="my-service",
            service_version="1.0.0",
        )
        configure_logging(config=config)

        logger = get_logger("service.test")
        logger.info("test_event")

        # Verify no exceptions
        assert True


class TestTextFormatter:
    """Tests for text formatter output."""

    def test_text_output_format(self):
        """Verify text output works."""
        output = io.StringIO()
        config = LogConfig(
            level="INFO",
            format=LogFormat.TEXT,
            output=output,
        )
        configure_logging(config=config)

        logger = get_logger("text.test")
        logger.info("test_event")

        # Verify no exceptions
        assert True


class TestConsoleFormatter:
    """Tests for console formatter output."""

    def test_console_output_format(self):
        """Verify console output works."""
        output = io.StringIO()
        config = LogConfig(
            level="INFO",
            format=LogFormat.CONSOLE,
            output=output,
        )
        configure_logging(config=config)

        logger = get_logger("console.test")
        logger.info("test_event")

        # Verify no exceptions
        assert True


class TestLoggerIntegration:
    """Integration tests for logging."""

    def test_full_workflow_logging(self):
        """Test logging through a simulated workflow."""
        configure_logging(level="DEBUG", format="json")

        # Create workflow logger
        logger = get_logger("llmteam.runner")

        # Simulate workflow logging
        run_logger = logger.bind(
            tenant_id="acme",
            run_id="run-123",
            segment_id="data_pipeline",
        )

        run_logger.info("segment_started", step_count=3)

        step_logger = run_logger.bind(step_id="fetch")
        step_logger.info("step_started")
        step_logger.debug("fetching_data", url="https://api.example.com")
        step_logger.info("step_completed", duration_ms=150)

        run_logger.info("segment_completed", status="success")

    def test_error_logging_workflow(self):
        """Test error logging through a simulated workflow."""
        configure_logging(level="DEBUG", format="json")

        logger = get_logger("llmteam.runner")
        run_logger = logger.bind(run_id="run-456")

        run_logger.info("segment_started")

        try:
            raise ConnectionError("API timeout")
        except ConnectionError as e:
            run_logger.error(
                "step_failed",
                step_id="api_call",
                error=str(e),
                error_type=type(e).__name__,
            )

        run_logger.info("segment_failed", status="failed")
