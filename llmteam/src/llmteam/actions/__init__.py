"""
External Actions Module (v1.9.0).

Provides external action execution with:
- Webhook/REST API calls
- Python function calls
- Action registry and executor
- Integration with RateLimiter and AuditTrail

Usage:
    from llmteam.actions import (
        ActionRegistry,
        ActionExecutor,
        ActionContext,
        ActionResult,
        ActionStatus,
    )

    # Create registry
    registry = ActionRegistry()
    registry.register_webhook("notify", "https://api.example.com/notify")
    registry.register_function("process", my_function)

    # Create executor
    executor = ActionExecutor(registry, rate_limiter, audit_trail)

    # Execute action
    context = ActionContext(
        action_name="notify",
        run_id="run_123",
        agent_name="agent_1",
        tenant_id="tenant_1",
        input_data={"message": "Hello"},
    )
    result = await executor.execute("notify", context)
"""

from .models import (
    ActionType,
    ActionStatus,
    ActionConfig,
    ActionContext,
    ActionResult,
    ActionHandler,
)
from .registry import ActionRegistry
from .executor import ActionExecutor
from .handlers import WebhookActionHandler, FunctionActionHandler

__all__ = [
    # Enums
    "ActionType",
    "ActionStatus",
    # Models
    "ActionConfig",
    "ActionContext",
    "ActionResult",
    "ActionHandler",
    # Registry & Executor
    "ActionRegistry",
    "ActionExecutor",
    # Handlers
    "WebhookActionHandler",
    "FunctionActionHandler",
]
