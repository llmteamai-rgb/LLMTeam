"""
Notification Channels for Human Interaction.

Provides notification channels for alerting humans:
- Base NotificationChannel
- Slack notifications
- Webhook notifications
"""

from .base import NotificationChannel

__all__ = ["NotificationChannel"]
