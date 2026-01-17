"""
Middleware System for LLMTeam.

Provides interceptors for step execution, enabling:
- Logging and tracing
- Authentication and authorization
- Rate limiting
- Caching
- Error handling
- Metrics collection

Usage:
    from llmteam.middleware import Middleware, MiddlewareStack

    class LoggingMiddleware(Middleware):
        async def __call__(self, ctx, config, input_data, next_handler):
            print(f"Starting step: {ctx.step_id}")
            result = await next_handler(ctx, config, input_data)
            print(f"Completed step: {ctx.step_id}")
            return result

    stack = MiddlewareStack()
    stack.use(LoggingMiddleware())
"""

from llmteam.middleware.base import (
    Middleware,
    MiddlewareStack,
    NextHandler,
    MiddlewareContext,
)

from llmteam.middleware.builtin import (
    LoggingMiddleware,
    TimingMiddleware,
    RetryMiddleware,
    CachingMiddleware,
    RateLimitMiddleware,
    AuthMiddleware,
    ValidationMiddleware,
)

__all__ = [
    # Base
    "Middleware",
    "MiddlewareStack",
    "NextHandler",
    "MiddlewareContext",
    # Built-in
    "LoggingMiddleware",
    "TimingMiddleware",
    "RetryMiddleware",
    "CachingMiddleware",
    "RateLimitMiddleware",
    "AuthMiddleware",
    "ValidationMiddleware",
]
