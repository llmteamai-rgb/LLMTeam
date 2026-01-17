"""
Middleware Base Classes.

Provides the core middleware abstraction for step execution interception.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, TypeVar, Awaitable
from datetime import datetime

from llmteam.runtime import StepContext


# Type alias for the next handler in the chain
NextHandler = Callable[[StepContext, dict, dict], Awaitable[Any]]


@dataclass
class MiddlewareContext:
    """
    Extended context passed to middleware.

    Contains additional metadata about the execution.
    """

    step_id: str
    step_type: str
    segment_id: str
    run_id: str
    tenant_id: str

    # Timing
    started_at: datetime = field(default_factory=datetime.now)

    # Metadata (user-defined)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Previous middleware results
    middleware_data: dict[str, Any] = field(default_factory=dict)


class Middleware(ABC):
    """
    Base class for all middleware.

    Middleware intercepts step execution and can:
    - Modify input data before execution
    - Modify output data after execution
    - Skip execution entirely (e.g., for caching)
    - Add timing, logging, or tracing
    - Handle errors and implement retry logic

    Usage:
        class MyMiddleware(Middleware):
            async def __call__(
                self,
                ctx: StepContext,
                config: dict,
                input_data: dict,
                next_handler: NextHandler,
                middleware_ctx: MiddlewareContext,
            ) -> Any:
                # Pre-processing
                modified_input = self.preprocess(input_data)

                # Call next handler
                result = await next_handler(ctx, config, modified_input)

                # Post-processing
                return self.postprocess(result)
    """

    # Middleware name (for logging/debugging)
    name: str = "middleware"

    # Priority (lower = runs first)
    priority: int = 100

    # Whether this middleware is enabled
    enabled: bool = True

    @abstractmethod
    async def __call__(
        self,
        ctx: StepContext,
        config: dict,
        input_data: dict,
        next_handler: NextHandler,
        middleware_ctx: MiddlewareContext,
    ) -> Any:
        """
        Execute middleware logic.

        Args:
            ctx: Step execution context
            config: Step configuration
            input_data: Input data for the step
            next_handler: The next handler in the chain (call to continue)
            middleware_ctx: Extended middleware context

        Returns:
            Step output (possibly modified)
        """
        pass

    def should_run(
        self,
        step_type: str,
        step_id: str,
        middleware_ctx: MiddlewareContext,
    ) -> bool:
        """
        Determine if this middleware should run for the given step.

        Override to implement conditional execution.

        Args:
            step_type: Type of step (e.g., "llm_agent", "transform")
            step_id: Step ID
            middleware_ctx: Middleware context

        Returns:
            True if middleware should run, False to skip
        """
        return self.enabled


class MiddlewareStack:
    """
    Stack of middleware for composing execution chains.

    Middleware are executed in priority order (lower priority first).

    Usage:
        stack = MiddlewareStack()
        stack.use(LoggingMiddleware())
        stack.use(TimingMiddleware())
        stack.use(RetryMiddleware(max_retries=3))

        # Execute with middleware
        result = await stack.execute(handler, ctx, config, input_data, middleware_ctx)
    """

    def __init__(self) -> None:
        self._middleware: list[Middleware] = []
        self._sorted = False

    def use(self, middleware: Middleware) -> "MiddlewareStack":
        """
        Add middleware to the stack.

        Args:
            middleware: Middleware instance to add

        Returns:
            Self for chaining
        """
        self._middleware.append(middleware)
        self._sorted = False
        return self

    def remove(self, middleware_name: str) -> bool:
        """
        Remove middleware by name.

        Args:
            middleware_name: Name of middleware to remove

        Returns:
            True if removed, False if not found
        """
        for i, m in enumerate(self._middleware):
            if m.name == middleware_name:
                self._middleware.pop(i)
                return True
        return False

    def clear(self) -> None:
        """Remove all middleware."""
        self._middleware.clear()
        self._sorted = False

    def list_middleware(self) -> list[str]:
        """Get list of middleware names in execution order."""
        self._ensure_sorted()
        return [m.name for m in self._middleware]

    def _ensure_sorted(self) -> None:
        """Sort middleware by priority if needed."""
        if not self._sorted:
            self._middleware.sort(key=lambda m: m.priority)
            self._sorted = True

    async def execute(
        self,
        handler: Callable,
        ctx: StepContext,
        config: dict,
        input_data: dict,
        middleware_ctx: MiddlewareContext,
    ) -> Any:
        """
        Execute handler with middleware chain.

        Args:
            handler: The actual step handler
            ctx: Step context
            config: Step configuration
            input_data: Input data
            middleware_ctx: Middleware context

        Returns:
            Handler output (possibly modified by middleware)
        """
        self._ensure_sorted()

        # Build the chain from bottom up
        async def final_handler(c: StepContext, cfg: dict, inp: dict) -> Any:
            return await handler(c, cfg, inp)

        chain: NextHandler = final_handler

        # Wrap each middleware around the chain (in reverse order)
        for middleware in reversed(self._middleware):
            if middleware.should_run(
                middleware_ctx.step_type,
                middleware_ctx.step_id,
                middleware_ctx,
            ):
                chain = self._wrap_middleware(middleware, chain, middleware_ctx)

        return await chain(ctx, config, input_data)

    def _wrap_middleware(
        self,
        middleware: Middleware,
        next_handler: NextHandler,
        middleware_ctx: MiddlewareContext,
    ) -> NextHandler:
        """Wrap a middleware around a handler."""
        async def wrapped(ctx: StepContext, config: dict, input_data: dict) -> Any:
            return await middleware(ctx, config, input_data, next_handler, middleware_ctx)
        return wrapped


class ConditionalMiddleware(Middleware):
    """
    Middleware that only runs for specific step types.

    Usage:
        class LLMOnlyMiddleware(ConditionalMiddleware):
            step_types = ["llm_agent"]

            async def __call__(self, ctx, config, input_data, next_handler, middleware_ctx):
                # Only runs for llm_agent steps
                ...
    """

    # Step types this middleware applies to (empty = all)
    step_types: list[str] = []

    # Step IDs to exclude
    exclude_step_ids: list[str] = []

    def should_run(
        self,
        step_type: str,
        step_id: str,
        middleware_ctx: MiddlewareContext,
    ) -> bool:
        """Check if middleware should run based on step type and ID."""
        if not self.enabled:
            return False

        if step_id in self.exclude_step_ids:
            return False

        if self.step_types and step_type not in self.step_types:
            return False

        return True


class CompositeMiddleware(Middleware):
    """
    Middleware that combines multiple middleware into one.

    Useful for grouping related middleware.

    Usage:
        observability = CompositeMiddleware(
            name="observability",
            middleware=[
                LoggingMiddleware(),
                TimingMiddleware(),
                TracingMiddleware(),
            ]
        )
        stack.use(observability)
    """

    def __init__(
        self,
        name: str = "composite",
        middleware: Optional[list[Middleware]] = None,
        priority: int = 100,
    ) -> None:
        self.name = name
        self.priority = priority
        self._stack = MiddlewareStack()

        for m in middleware or []:
            self._stack.use(m)

    def add(self, middleware: Middleware) -> "CompositeMiddleware":
        """Add middleware to the composite."""
        self._stack.use(middleware)
        return self

    async def __call__(
        self,
        ctx: StepContext,
        config: dict,
        input_data: dict,
        next_handler: NextHandler,
        middleware_ctx: MiddlewareContext,
    ) -> Any:
        """Execute all contained middleware."""
        # Create a handler that calls next_handler
        async def final_handler(c: StepContext, cfg: dict, inp: dict) -> Any:
            return await next_handler(c, cfg, inp)

        return await self._stack.execute(
            final_handler, ctx, config, input_data, middleware_ctx
        )
