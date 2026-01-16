"""
Human Interaction Module (v1.9.0).

Provides human-in-the-loop capabilities:
- Approval requests
- Choice selections
- Data input requests
- Notification channels (Slack, Webhook, etc.)
- Escalation handling

Usage:
    from llmteam.human import (
        HumanInteractionManager,
        InteractionRequest,
        InteractionResponse,
        InteractionType,
        InteractionStatus,
    )

    # Create manager
    store = MemoryInteractionStore()
    manager = HumanInteractionManager(store, notification_config, audit_trail)

    # Request approval
    request = await manager.request_approval(
        title="Approve deployment",
        description="Deploy version 1.2.3 to production?",
        run_id="run_123",
        pipeline_id="deploy_pipeline",
        agent_name="deploy_agent",
        assignee="alice@example.com",
    )

    # Wait for response
    response = await manager.wait_for_response(request.request_id)
"""

from .models import (
    InteractionType,
    InteractionStatus,
    InteractionPriority,
    InteractionRequest,
    InteractionResponse,
    NotificationConfig,
)
from .manager import HumanInteractionManager
from .store import InteractionStore, MemoryInteractionStore
from .notifications.base import NotificationChannel

__all__ = [
    # Enums
    "InteractionType",
    "InteractionStatus",
    "InteractionPriority",
    # Models
    "InteractionRequest",
    "InteractionResponse",
    "NotificationConfig",
    # Manager & Store
    "HumanInteractionManager",
    "InteractionStore",
    "MemoryInteractionStore",
    # Notifications
    "NotificationChannel",
]
