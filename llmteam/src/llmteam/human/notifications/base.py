"""
Base Notification Channel.
"""

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import InteractionRequest


class NotificationChannel:
    """Base class for notification channels."""

    async def send(self, request: "InteractionRequest", recipients: List[str]) -> bool:
        """
        Send notification.

        Args:
            request: Interaction request
            recipients: List of recipient IDs

        Returns:
            True if sent successfully
        """
        raise NotImplementedError
