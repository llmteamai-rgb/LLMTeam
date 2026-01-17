"""
Built-in Middleware Implementations.

Provides common middleware for:
- Logging
- Timing
- Retry
- Caching
- Rate limiting
- Authentication
- Validation
"""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from llmteam.runtime import StepContext
from llmteam.middleware.base import (
    Middleware,
    ConditionalMiddleware,
    NextHandler,
    MiddlewareContext,
)


class LoggingMiddleware(Middleware):
    """
    Middleware for logging step execution.

    Logs step start, completion, and errors with configurable detail level.
    """

    name = "logging"
    priority = 10  # Run early

    def __init__(
        self,
        log_input: bool = False,
        log_output: bool = False,
        log_func: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Initialize logging middleware.

        Args:
            log_input: Whether to log input data
            log_output: Whether to log output data
            log_func: Custom logging function (default: print)
        """
        self.log_input = log_input
        self.log_output = log_output
        self.log_func = log_func or print

    async def __call__(
        self,
        ctx: StepContext,
        config: dict,
        input_data: dict,
        next_handler: NextHandler,
        middleware_ctx: MiddlewareContext,
    ) -> Any:
        step_id = middleware_ctx.step_id
        step_type = middleware_ctx.step_type

        # Log start
        self.log_func(f"[{step_id}] Starting {step_type}")
        if self.log_input:
            self.log_func(f"[{step_id}] Input: {json.dumps(input_data, default=str)[:500]}")

        try:
            result = await next_handler(ctx, config, input_data)

            # Log completion
            self.log_func(f"[{step_id}] Completed {step_type}")
            if self.log_output:
                self.log_func(f"[{step_id}] Output: {json.dumps(result, default=str)[:500]}")

            return result

        except Exception as e:
            self.log_func(f"[{step_id}] Failed {step_type}: {type(e).__name__}: {e}")
            raise


class TimingMiddleware(Middleware):
    """
    Middleware for measuring step execution time.

    Adds timing information to middleware context.
    """

    name = "timing"
    priority = 20

    def __init__(
        self,
        slow_threshold_ms: int = 5000,
        on_slow: Optional[Callable[[str, int], None]] = None,
    ) -> None:
        """
        Initialize timing middleware.

        Args:
            slow_threshold_ms: Threshold for slow step warning (ms)
            on_slow: Callback for slow steps (step_id, duration_ms)
        """
        self.slow_threshold_ms = slow_threshold_ms
        self.on_slow = on_slow

    async def __call__(
        self,
        ctx: StepContext,
        config: dict,
        input_data: dict,
        next_handler: NextHandler,
        middleware_ctx: MiddlewareContext,
    ) -> Any:
        start = datetime.now()

        result = await next_handler(ctx, config, input_data)

        duration = datetime.now() - start
        duration_ms = int(duration.total_seconds() * 1000)

        # Store timing in middleware context
        middleware_ctx.middleware_data["timing"] = {
            "started_at": start.isoformat(),
            "duration_ms": duration_ms,
        }

        # Check slow threshold
        if duration_ms > self.slow_threshold_ms and self.on_slow:
            self.on_slow(middleware_ctx.step_id, duration_ms)

        return result


@dataclass
class RetryConfig:
    """Configuration for retry middleware."""

    max_retries: int = 3
    initial_delay_ms: int = 1000
    max_delay_ms: int = 30000
    exponential_base: float = 2.0
    retryable_exceptions: tuple = (Exception,)


class RetryMiddleware(Middleware):
    """
    Middleware for retrying failed steps.

    Implements exponential backoff with configurable retries.
    """

    name = "retry"
    priority = 30

    def __init__(self, config: Optional[RetryConfig] = None) -> None:
        """
        Initialize retry middleware.

        Args:
            config: Retry configuration
        """
        self.config = config or RetryConfig()

    async def __call__(
        self,
        ctx: StepContext,
        config: dict,
        input_data: dict,
        next_handler: NextHandler,
        middleware_ctx: MiddlewareContext,
    ) -> Any:
        last_exception: Optional[Exception] = None
        attempts = 0

        for attempt in range(self.config.max_retries + 1):
            attempts += 1
            try:
                result = await next_handler(ctx, config, input_data)
                middleware_ctx.middleware_data["retry"] = {
                    "attempts": attempts,
                    "success": True,
                }
                return result

            except self.config.retryable_exceptions as e:
                last_exception = e

                if attempt < self.config.max_retries:
                    # Calculate delay with exponential backoff
                    delay_ms = min(
                        self.config.initial_delay_ms * (self.config.exponential_base ** attempt),
                        self.config.max_delay_ms,
                    )
                    await asyncio.sleep(delay_ms / 1000)

        # All retries exhausted
        middleware_ctx.middleware_data["retry"] = {
            "attempts": attempts,
            "success": False,
            "error": str(last_exception),
        }

        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected: no exception after failed retries")


@dataclass
class CacheEntry:
    """Cache entry with expiration."""

    value: Any
    expires_at: datetime


class CachingMiddleware(ConditionalMiddleware):
    """
    Middleware for caching step results.

    Caches results based on step type, config, and input data.
    """

    name = "caching"
    priority = 5  # Run very early (to skip execution if cached)

    def __init__(
        self,
        ttl_seconds: int = 300,
        max_entries: int = 1000,
        cache_step_types: Optional[list[str]] = None,
    ) -> None:
        """
        Initialize caching middleware.

        Args:
            ttl_seconds: Cache entry TTL in seconds
            max_entries: Maximum cache entries
            cache_step_types: Step types to cache (default: llm_agent, http_action)
        """
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.step_types = cache_step_types or ["llm_agent", "http_action"]
        self._cache: dict[str, CacheEntry] = {}

    def _make_key(self, step_type: str, config: dict, input_data: dict) -> str:
        """Generate cache key from step parameters."""
        key_data = {
            "type": step_type,
            "config": config,
            "input": input_data,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _cleanup(self) -> None:
        """Remove expired entries and enforce max size."""
        now = datetime.now()

        # Remove expired
        expired = [k for k, v in self._cache.items() if v.expires_at < now]
        for k in expired:
            del self._cache[k]

        # Enforce max size (remove oldest)
        if len(self._cache) > self.max_entries:
            sorted_keys = sorted(
                self._cache.keys(),
                key=lambda k: self._cache[k].expires_at,
            )
            for k in sorted_keys[: len(self._cache) - self.max_entries]:
                del self._cache[k]

    async def __call__(
        self,
        ctx: StepContext,
        config: dict,
        input_data: dict,
        next_handler: NextHandler,
        middleware_ctx: MiddlewareContext,
    ) -> Any:
        cache_key = self._make_key(middleware_ctx.step_type, config, input_data)

        # Check cache
        entry = self._cache.get(cache_key)
        if entry and entry.expires_at > datetime.now():
            middleware_ctx.middleware_data["cache"] = {"hit": True}
            return entry.value

        # Execute handler
        result = await next_handler(ctx, config, input_data)

        # Store in cache
        self._cache[cache_key] = CacheEntry(
            value=result,
            expires_at=datetime.now() + timedelta(seconds=self.ttl_seconds),
        )
        self._cleanup()

        middleware_ctx.middleware_data["cache"] = {"hit": False}
        return result

    def invalidate(self, step_type: Optional[str] = None) -> int:
        """
        Invalidate cache entries.

        Args:
            step_type: If provided, only invalidate entries for this step type

        Returns:
            Number of entries invalidated
        """
        if step_type is None:
            count = len(self._cache)
            self._cache.clear()
            return count

        # Would need to track step_type in entries for selective invalidation
        # For now, clear all
        count = len(self._cache)
        self._cache.clear()
        return count


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_second: float = 10.0
    burst_size: int = 20
    per_tenant: bool = True


class RateLimitMiddleware(Middleware):
    """
    Middleware for rate limiting step execution.

    Uses token bucket algorithm for smooth rate limiting.
    """

    name = "rate_limit"
    priority = 15

    def __init__(self, config: Optional[RateLimitConfig] = None) -> None:
        """
        Initialize rate limit middleware.

        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()
        self._buckets: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    def _get_bucket_key(self, middleware_ctx: MiddlewareContext) -> str:
        """Get bucket key based on configuration."""
        if self.config.per_tenant:
            return f"tenant:{middleware_ctx.tenant_id}"
        return "global"

    async def _acquire(self, key: str) -> bool:
        """Try to acquire a token from the bucket."""
        async with self._lock:
            now = datetime.now()

            if key not in self._buckets:
                self._buckets[key] = {
                    "tokens": self.config.burst_size,
                    "last_update": now,
                }

            bucket = self._buckets[key]

            # Refill tokens based on time elapsed
            elapsed = (now - bucket["last_update"]).total_seconds()
            refill = elapsed * self.config.requests_per_second
            bucket["tokens"] = min(
                self.config.burst_size,
                bucket["tokens"] + refill,
            )
            bucket["last_update"] = now

            # Try to take a token
            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                return True

            return False

    async def __call__(
        self,
        ctx: StepContext,
        config: dict,
        input_data: dict,
        next_handler: NextHandler,
        middleware_ctx: MiddlewareContext,
    ) -> Any:
        key = self._get_bucket_key(middleware_ctx)

        # Wait for token with backoff
        max_wait_seconds = 30
        wait_interval = 0.1
        total_wait = 0.0

        while not await self._acquire(key):
            if total_wait >= max_wait_seconds:
                raise RuntimeError(
                    f"Rate limit exceeded. Max wait time ({max_wait_seconds}s) reached."
                )
            await asyncio.sleep(wait_interval)
            total_wait += wait_interval

        middleware_ctx.middleware_data["rate_limit"] = {
            "bucket": key,
            "waited_seconds": total_wait,
        }

        return await next_handler(ctx, config, input_data)


class AuthMiddleware(Middleware):
    """
    Middleware for authentication/authorization.

    Validates that the runtime context has required permissions.
    """

    name = "auth"
    priority = 1  # Run first

    def __init__(
        self,
        required_permissions: Optional[list[str]] = None,
        permission_checker: Optional[Callable[[StepContext, list[str]], bool]] = None,
    ) -> None:
        """
        Initialize auth middleware.

        Args:
            required_permissions: List of required permissions
            permission_checker: Custom permission checker function
        """
        self.required_permissions = required_permissions or []
        self.permission_checker = permission_checker

    async def __call__(
        self,
        ctx: StepContext,
        config: dict,
        input_data: dict,
        next_handler: NextHandler,
        middleware_ctx: MiddlewareContext,
    ) -> Any:
        # Check permissions
        if self.permission_checker:
            if not self.permission_checker(ctx, self.required_permissions):
                raise PermissionError(
                    f"Step {middleware_ctx.step_id} requires permissions: "
                    f"{self.required_permissions}"
                )
        elif self.required_permissions:
            # Default: check runtime context metadata
            user_permissions = ctx.metadata.get("permissions", [])
            missing = set(self.required_permissions) - set(user_permissions)
            if missing:
                raise PermissionError(
                    f"Missing permissions for step {middleware_ctx.step_id}: {missing}"
                )

        middleware_ctx.middleware_data["auth"] = {"authorized": True}
        return await next_handler(ctx, config, input_data)


class ValidationMiddleware(Middleware):
    """
    Middleware for validating step input and output.

    Can validate against JSON schemas or custom validators.
    """

    name = "validation"
    priority = 25

    def __init__(
        self,
        input_validator: Optional[Callable[[dict], bool]] = None,
        output_validator: Optional[Callable[[Any], bool]] = None,
        input_schema: Optional[dict] = None,
        output_schema: Optional[dict] = None,
    ) -> None:
        """
        Initialize validation middleware.

        Args:
            input_validator: Custom input validation function
            output_validator: Custom output validation function
            input_schema: JSON Schema for input validation
            output_schema: JSON Schema for output validation
        """
        self.input_validator = input_validator
        self.output_validator = output_validator
        self.input_schema = input_schema
        self.output_schema = output_schema

    def _validate_schema(self, data: Any, schema: dict, name: str) -> None:
        """Validate data against JSON Schema."""
        try:
            import jsonschema
            jsonschema.validate(data, schema)
        except ImportError:
            # jsonschema not installed, skip validation
            pass
        except jsonschema.ValidationError as e:
            raise ValueError(f"{name} validation failed: {e.message}")

    async def __call__(
        self,
        ctx: StepContext,
        config: dict,
        input_data: dict,
        next_handler: NextHandler,
        middleware_ctx: MiddlewareContext,
    ) -> Any:
        # Validate input
        if self.input_validator and not self.input_validator(input_data):
            raise ValueError(f"Input validation failed for step {middleware_ctx.step_id}")

        if self.input_schema:
            self._validate_schema(input_data, self.input_schema, "Input")

        # Execute handler
        result = await next_handler(ctx, config, input_data)

        # Validate output
        if self.output_validator and not self.output_validator(result):
            raise ValueError(f"Output validation failed for step {middleware_ctx.step_id}")

        if self.output_schema:
            self._validate_schema(result, self.output_schema, "Output")

        middleware_ctx.middleware_data["validation"] = {"passed": True}
        return result
