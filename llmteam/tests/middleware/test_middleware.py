"""Tests for middleware module."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from llmteam.middleware import (
    Middleware,
    MiddlewareStack,
    MiddlewareContext,
    LoggingMiddleware,
    TimingMiddleware,
    RetryMiddleware,
    CachingMiddleware,
    RateLimitMiddleware,
    AuthMiddleware,
    ValidationMiddleware,
)
from llmteam.middleware.builtin import RetryConfig


class TestMiddlewareContext:
    """Tests for MiddlewareContext."""

    def test_create_context(self):
        """Create middleware context with required fields."""
        ctx = MiddlewareContext(
            step_id="step1",
            step_type="transform",
            segment_id="seg1",
            run_id="run1",
            tenant_id="tenant1",
        )

        assert ctx.step_id == "step1"
        assert ctx.step_type == "transform"
        assert ctx.segment_id == "seg1"
        assert ctx.run_id == "run1"
        assert ctx.tenant_id == "tenant1"

    def test_context_has_metadata(self):
        """Context has metadata dict."""
        ctx = MiddlewareContext(
            step_id="step1",
            step_type="transform",
            segment_id="seg1",
            run_id="run1",
            tenant_id="tenant1",
        )

        ctx.metadata["key"] = "value"
        assert ctx.metadata["key"] == "value"

    def test_context_has_middleware_data(self):
        """Context has middleware_data dict."""
        ctx = MiddlewareContext(
            step_id="step1",
            step_type="transform",
            segment_id="seg1",
            run_id="run1",
            tenant_id="tenant1",
        )

        ctx.middleware_data["timing"] = {"duration_ms": 100}
        assert ctx.middleware_data["timing"]["duration_ms"] == 100


class TestMiddlewareStack:
    """Tests for MiddlewareStack."""

    def test_create_empty_stack(self):
        """Create empty middleware stack."""
        stack = MiddlewareStack()
        assert stack.list_middleware() == []

    def test_use_adds_middleware(self):
        """Use method adds middleware to stack."""
        stack = MiddlewareStack()
        middleware = LoggingMiddleware()

        stack.use(middleware)

        assert "logging" in stack.list_middleware()

    def test_remove_middleware(self):
        """Remove method removes middleware by name."""
        stack = MiddlewareStack()
        stack.use(LoggingMiddleware())
        stack.use(TimingMiddleware())

        result = stack.remove("logging")

        assert result is True
        assert "logging" not in stack.list_middleware()
        assert "timing" in stack.list_middleware()

    def test_remove_nonexistent_returns_false(self):
        """Remove returns False for nonexistent middleware."""
        stack = MiddlewareStack()
        result = stack.remove("nonexistent")
        assert result is False

    def test_clear_removes_all(self):
        """Clear removes all middleware."""
        stack = MiddlewareStack()
        stack.use(LoggingMiddleware())
        stack.use(TimingMiddleware())

        stack.clear()

        assert stack.list_middleware() == []

    def test_middleware_sorted_by_priority(self):
        """Middleware are sorted by priority."""
        stack = MiddlewareStack()

        # Add in non-priority order
        timing = TimingMiddleware()  # priority 20
        logging = LoggingMiddleware()  # priority 10
        retry = RetryMiddleware()  # priority 30

        stack.use(retry)
        stack.use(logging)
        stack.use(timing)

        names = stack.list_middleware()
        assert names == ["logging", "timing", "retry"]

    async def test_execute_calls_handler(self):
        """Execute calls the actual handler."""
        stack = MiddlewareStack()
        handler = AsyncMock(return_value={"result": "ok"})
        ctx = MagicMock()
        middleware_ctx = MiddlewareContext(
            step_id="step1",
            step_type="transform",
            segment_id="seg1",
            run_id="run1",
            tenant_id="tenant1",
        )

        result = await stack.execute(
            handler, ctx, {}, {"input": "data"}, middleware_ctx
        )

        assert result == {"result": "ok"}
        handler.assert_called_once()


class TestLoggingMiddleware:
    """Tests for LoggingMiddleware."""

    def test_default_initialization(self):
        """Initialize with default values."""
        middleware = LoggingMiddleware()

        assert middleware.name == "logging"
        assert middleware.priority == 10
        assert middleware.log_input is False
        assert middleware.log_output is False

    def test_custom_log_func(self):
        """Initialize with custom log function."""
        logs = []
        middleware = LoggingMiddleware(log_func=logs.append)

        middleware.log_func("test message")
        assert "test message" in logs

    async def test_logs_execution(self):
        """Logs step execution."""
        logs = []
        middleware = LoggingMiddleware(log_func=logs.append)
        handler = AsyncMock(return_value={"result": "ok"})
        ctx = MagicMock()
        middleware_ctx = MiddlewareContext(
            step_id="step1",
            step_type="transform",
            segment_id="seg1",
            run_id="run1",
            tenant_id="tenant1",
        )

        await middleware(ctx, {}, {}, handler, middleware_ctx)

        assert any("Starting" in log for log in logs)
        assert any("Completed" in log for log in logs)


class TestTimingMiddleware:
    """Tests for TimingMiddleware."""

    def test_default_initialization(self):
        """Initialize with default values."""
        middleware = TimingMiddleware()

        assert middleware.name == "timing"
        assert middleware.priority == 20
        assert middleware.slow_threshold_ms == 5000

    async def test_records_timing(self):
        """Records timing in middleware context."""
        middleware = TimingMiddleware()
        handler = AsyncMock(return_value={"result": "ok"})
        ctx = MagicMock()
        middleware_ctx = MiddlewareContext(
            step_id="step1",
            step_type="transform",
            segment_id="seg1",
            run_id="run1",
            tenant_id="tenant1",
        )

        await middleware(ctx, {}, {}, handler, middleware_ctx)

        assert "timing" in middleware_ctx.middleware_data
        assert "duration_ms" in middleware_ctx.middleware_data["timing"]


class TestRetryMiddleware:
    """Tests for RetryMiddleware."""

    def test_default_initialization(self):
        """Initialize with default values."""
        middleware = RetryMiddleware()

        assert middleware.name == "retry"
        assert middleware.priority == 30

    async def test_success_on_first_try(self):
        """Succeeds on first try without retry."""
        middleware = RetryMiddleware()
        handler = AsyncMock(return_value={"result": "ok"})
        ctx = MagicMock()
        middleware_ctx = MiddlewareContext(
            step_id="step1",
            step_type="transform",
            segment_id="seg1",
            run_id="run1",
            tenant_id="tenant1",
        )

        result = await middleware(ctx, {}, {}, handler, middleware_ctx)

        assert result == {"result": "ok"}
        assert middleware_ctx.middleware_data["retry"]["attempts"] == 1
        assert middleware_ctx.middleware_data["retry"]["success"] is True

    async def test_retries_on_failure(self):
        """Retries on failure."""
        config = RetryConfig(max_retries=2, initial_delay_ms=1)
        middleware = RetryMiddleware(config=config)

        call_count = 0

        async def failing_handler(ctx, config, input_data):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Simulated failure")
            return {"result": "ok"}

        ctx = MagicMock()
        middleware_ctx = MiddlewareContext(
            step_id="step1",
            step_type="transform",
            segment_id="seg1",
            run_id="run1",
            tenant_id="tenant1",
        )

        result = await middleware(ctx, {}, {}, failing_handler, middleware_ctx)

        assert result == {"result": "ok"}
        assert call_count == 3


class TestCachingMiddleware:
    """Tests for CachingMiddleware."""

    def test_default_initialization(self):
        """Initialize with default values."""
        middleware = CachingMiddleware()

        assert middleware.name == "caching"
        assert middleware.priority == 5
        assert middleware.ttl_seconds == 300

    async def test_cache_hit(self):
        """Returns cached result on cache hit."""
        middleware = CachingMiddleware(ttl_seconds=60)
        handler = AsyncMock(return_value={"result": "ok"})
        ctx = MagicMock()
        middleware_ctx = MiddlewareContext(
            step_id="step1",
            step_type="llm_agent",
            segment_id="seg1",
            run_id="run1",
            tenant_id="tenant1",
        )

        # First call - cache miss
        result1 = await middleware(ctx, {}, {"input": "test"}, handler, middleware_ctx)
        assert middleware_ctx.middleware_data["cache"]["hit"] is False

        # Second call - cache hit
        middleware_ctx2 = MiddlewareContext(
            step_id="step1",
            step_type="llm_agent",
            segment_id="seg1",
            run_id="run1",
            tenant_id="tenant1",
        )
        result2 = await middleware(ctx, {}, {"input": "test"}, handler, middleware_ctx2)

        assert result1 == result2
        assert middleware_ctx2.middleware_data["cache"]["hit"] is True
        assert handler.call_count == 1  # Only called once

    def test_invalidate_clears_cache(self):
        """Invalidate clears cache."""
        middleware = CachingMiddleware()
        middleware._cache["key"] = MagicMock()

        count = middleware.invalidate()

        assert count == 1
        assert len(middleware._cache) == 0


class TestRateLimitMiddleware:
    """Tests for RateLimitMiddleware."""

    def test_default_initialization(self):
        """Initialize with default values."""
        middleware = RateLimitMiddleware()

        assert middleware.name == "rate_limit"
        assert middleware.priority == 15

    async def test_allows_within_limit(self):
        """Allows requests within rate limit."""
        from llmteam.middleware.builtin import RateLimitConfig

        config = RateLimitConfig(requests_per_second=10, burst_size=5)
        middleware = RateLimitMiddleware(config=config)
        handler = AsyncMock(return_value={"result": "ok"})
        ctx = MagicMock()
        middleware_ctx = MiddlewareContext(
            step_id="step1",
            step_type="transform",
            segment_id="seg1",
            run_id="run1",
            tenant_id="tenant1",
        )

        result = await middleware(ctx, {}, {}, handler, middleware_ctx)

        assert result == {"result": "ok"}


class TestAuthMiddleware:
    """Tests for AuthMiddleware."""

    def test_default_initialization(self):
        """Initialize with default values."""
        middleware = AuthMiddleware()

        assert middleware.name == "auth"
        assert middleware.priority == 1

    async def test_passes_without_validators(self):
        """Passes through without validators configured."""
        middleware = AuthMiddleware()
        handler = AsyncMock(return_value={"result": "ok"})
        ctx = MagicMock()
        ctx.metadata = {}
        middleware_ctx = MiddlewareContext(
            step_id="step1",
            step_type="transform",
            segment_id="seg1",
            run_id="run1",
            tenant_id="tenant1",
        )

        result = await middleware(ctx, {}, {}, handler, middleware_ctx)

        assert result == {"result": "ok"}
        # Without validators, auth passes through without setting authenticated
        assert ctx.metadata.get("authenticated", False) is False


class TestValidationMiddleware:
    """Tests for ValidationMiddleware."""

    def test_default_initialization(self):
        """Initialize with default values."""
        middleware = ValidationMiddleware()

        assert middleware.name == "validation"
        assert middleware.priority == 25

    async def test_passes_valid_input(self):
        """Passes through with valid input."""
        def validator(data):
            return "input" in data

        middleware = ValidationMiddleware(input_validator=validator)
        handler = AsyncMock(return_value={"result": "ok"})
        ctx = MagicMock()
        middleware_ctx = MiddlewareContext(
            step_id="step1",
            step_type="transform",
            segment_id="seg1",
            run_id="run1",
            tenant_id="tenant1",
        )

        result = await middleware(ctx, {}, {"input": "test"}, handler, middleware_ctx)

        assert result == {"result": "ok"}

    async def test_fails_invalid_input(self):
        """Raises error for invalid input."""
        def validator(data):
            return "required_field" in data

        middleware = ValidationMiddleware(input_validator=validator)
        handler = AsyncMock(return_value={"result": "ok"})
        ctx = MagicMock()
        middleware_ctx = MiddlewareContext(
            step_id="step1",
            step_type="transform",
            segment_id="seg1",
            run_id="run1",
            tenant_id="tenant1",
        )

        with pytest.raises(ValueError) as exc_info:
            await middleware(ctx, {}, {"other": "data"}, handler, middleware_ctx)

        assert "validation failed" in str(exc_info.value)
