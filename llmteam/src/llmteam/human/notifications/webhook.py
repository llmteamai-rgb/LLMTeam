"""
Generic Webhook Notification Channel.
"""

from typing import List

try:
    import aiohttp
except ImportError:
    aiohttp = None

from .base import NotificationChannel
from ..models import InteractionRequest


class WebhookNotificationChannel(NotificationChannel):
    """Generic webhook notification channel."""

    def __init__(self, webhook_url: str):
        if aiohttp is None:
            raise ImportError("aiohttp is required for webhook notifications")

        self.webhook_url = webhook_url

    async def send(self, request: InteractionRequest, recipients: List[str]) -> bool:
        """Send notification via webhook."""
        payload = {
            "request_id": request.request_id,
            "type": request.interaction_type.value,
            "title": request.title,
            "description": request.description,
            "priority": request.priority.value,
            "assignees": recipients,
            "deadline": request.deadline.isoformat() if request.deadline else None,
            "pipeline_id": request.pipeline_id,
            "run_id": request.run_id,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    return resp.ok
        except Exception:
            return False
