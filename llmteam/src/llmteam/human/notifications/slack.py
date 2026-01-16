"""
Slack Notification Channel.
"""

from typing import List, Dict

try:
    import aiohttp
except ImportError:
    aiohttp = None

from .base import NotificationChannel
from ..models import InteractionRequest, InteractionType, InteractionPriority


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel using webhooks."""

    def __init__(self, webhook_url: str, channel: str = ""):
        if aiohttp is None:
            raise ImportError("aiohttp is required for Slack notifications")

        self.webhook_url = webhook_url
        self.channel = channel

    async def send(self, request: InteractionRequest, recipients: List[str]) -> bool:
        """Send notification to Slack."""
        payload = {
            "channel": self.channel,
            "text": f"*{request.title}*\n{request.description}",
            "attachments": [
                {
                    "color": self._get_priority_color(request.priority),
                    "fields": [
                        {"title": "Pipeline", "value": request.pipeline_id, "short": True},
                        {"title": "Priority", "value": request.priority.value, "short": True},
                        {
                            "title": "Deadline",
                            "value": str(request.deadline) if request.deadline else "N/A",
                            "short": True,
                        },
                        {"title": "Assignee", "value": request.assignee_id or "Unassigned", "short": True},
                    ],
                    "actions": self._build_actions(request),
                }
            ],
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as resp:
                    return resp.ok
        except Exception:
            return False

    def _get_priority_color(self, priority: InteractionPriority) -> str:
        """Get color code for priority."""
        colors = {
            InteractionPriority.LOW: "#36a64f",  # Green
            InteractionPriority.NORMAL: "#2196f3",  # Blue
            InteractionPriority.HIGH: "#ff9800",  # Orange
            InteractionPriority.CRITICAL: "#f44336",  # Red
        }
        return colors.get(priority, "#2196f3")

    def _build_actions(self, request: InteractionRequest) -> List[Dict]:
        """Build action buttons for request type."""
        if request.interaction_type == InteractionType.APPROVAL:
            return [
                {"type": "button", "text": "Approve", "value": "approve", "style": "primary"},
                {"type": "button", "text": "Reject", "value": "reject", "style": "danger"},
            ]
        elif request.interaction_type == InteractionType.CHOICE and request.options:
            return [
                {"type": "button", "text": opt.get("label", "Option"), "value": opt.get("value", "")}
                for opt in request.options[:5]  # Limit to 5 buttons
            ]
        return []
