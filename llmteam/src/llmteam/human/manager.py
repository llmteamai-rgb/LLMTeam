"""
Human Interaction Manager.

Manages human-in-the-loop interactions with:
- Approval requests
- Choice requests
- Input requests
- Notification channels
- Escalation handling
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, TYPE_CHECKING
import uuid

from .models import (
    InteractionRequest,
    InteractionResponse,
    InteractionType,
    InteractionStatus,
    InteractionPriority,
    NotificationConfig,
)
from .store import InteractionStore

if TYPE_CHECKING:
    from ..audit import AuditTrail, AuditEventType
    from .notifications.base import NotificationChannel


def generate_uuid() -> str:
    """Generate UUID4 string."""
    return str(uuid.uuid4())


from llmteam.licensing import professional_only


@professional_only
class HumanInteractionManager:
    """
    Manager for human interactions.

    Integrates with:
    - AuditTrail from v1.7.0
    - TenantContext from v1.7.0
    """

    def __init__(
        self,
        store: InteractionStore,
        notification_config: Optional[NotificationConfig] = None,
        audit_trail: Optional["AuditTrail"] = None,
    ):
        self.store = store
        self.notification_config = notification_config or NotificationConfig()
        self.audit_trail = audit_trail

        self._channels: Dict[str, "NotificationChannel"] = {}
        self._pending_requests: Dict[str, asyncio.Event] = {}

        self._setup_channels()

    def _setup_channels(self) -> None:
        """Setup notification channels."""
        # Slack channel
        if self.notification_config.slack_enabled:
            try:
                from .notifications.slack import SlackNotificationChannel

                self._channels["slack"] = SlackNotificationChannel(
                    self.notification_config.slack_webhook_url,
                    self.notification_config.slack_channel,
                )
            except ImportError:
                pass

        # Webhook channel
        if self.notification_config.webhook_enabled:
            try:
                from .notifications.webhook import WebhookNotificationChannel

                self._channels["webhook"] = WebhookNotificationChannel(
                    self.notification_config.custom_webhook_url
                )
            except ImportError:
                pass

    async def request_approval(
        self,
        title: str,
        description: str,
        *,
        run_id: str,
        pipeline_id: str,
        agent_name: str,
        assignee: Optional[str] = None,
        priority: InteractionPriority = InteractionPriority.NORMAL,
        timeout: timedelta = timedelta(hours=24),
        context_data: Optional[dict] = None,
    ) -> InteractionRequest:
        """Request approval from human."""
        from ..tenancy import current_tenant

        request = InteractionRequest(
            request_id=generate_uuid(),
            interaction_type=InteractionType.APPROVAL,
            run_id=run_id,
            pipeline_id=pipeline_id,
            agent_name=agent_name,
            step_name="approval",
            tenant_id=current_tenant.get(),
            title=title,
            description=description,
            context_data=context_data or {},
            assignee_id=assignee,
            priority=priority,
            timeout=timeout,
            deadline=datetime.now() + timeout,
        )

        await self.store.save(request)
        await self._notify(request)

        # Audit logging
        if self.audit_trail:
            from ..audit import AuditEventType

            await self.audit_trail.log(
                AuditEventType.APPROVAL_REQUESTED,
                actor_id=agent_name,
                resource_type="approval_request",
                resource_id=request.request_id,
                metadata={"title": title, "assignee": assignee},
            )

        return request

    async def request_choice(
        self,
        title: str,
        description: str,
        options: List[Dict[str, any]],
        **kwargs,
    ) -> InteractionRequest:
        """Request choice from options."""
        from ..tenancy import current_tenant

        request = InteractionRequest(
            request_id=generate_uuid(),
            interaction_type=InteractionType.CHOICE,
            title=title,
            description=description,
            options=options,
            tenant_id=current_tenant.get(),
            **kwargs,
        )

        await self.store.save(request)
        await self._notify(request)
        return request

    async def request_input(
        self,
        title: str,
        description: str,
        input_schema: dict,
        **kwargs,
    ) -> InteractionRequest:
        """Request data input."""
        from ..tenancy import current_tenant

        request = InteractionRequest(
            request_id=generate_uuid(),
            interaction_type=InteractionType.INPUT,
            title=title,
            description=description,
            input_schema=input_schema,
            tenant_id=current_tenant.get(),
            **kwargs,
        )

        await self.store.save(request)
        await self._notify(request)
        return request

    async def respond(
        self,
        request_id: str,
        responder_id: str,
        **kwargs,
    ) -> InteractionResponse:
        """Submit response to interaction request."""
        request = await self.store.get(request_id)
        if not request:
            raise ValueError(f"Request {request_id} not found")

        response = InteractionResponse(
            request_id=request_id,
            responder_id=responder_id,
            **kwargs,
        )

        # Update request status
        request.status = InteractionStatus.COMPLETED
        request.updated_at = datetime.now()
        await self.store.update(request)

        # Save response
        await self.store.save_response(response)

        # Wake up any waiting coroutines
        if request_id in self._pending_requests:
            self._pending_requests[request_id].set()

        # Audit logging
        if self.audit_trail:
            from ..audit import AuditEventType

            await self.audit_trail.log(
                AuditEventType.APPROVAL_COMPLETED,
                actor_id=responder_id,
                resource_type="approval_response",
                resource_id=request_id,
                metadata={"approved": response.approved},
            )

        return response

    async def wait_for_response(
        self,
        request_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[InteractionResponse]:
        """Wait for response to interaction request."""
        request = await self.store.get(request_id)
        if not request:
            return None

        timeout_seconds = timeout or request.timeout.total_seconds()

        # Create event for waiting
        event = asyncio.Event()
        self._pending_requests[request_id] = event

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_seconds)
            return await self.store.get_response(request_id)

        except asyncio.TimeoutError:
            # Handle timeout
            request.status = InteractionStatus.TIMEOUT
            await self.store.update(request)

            # Try escalation if configured
            await self._escalate(request)

            return None
        finally:
            self._pending_requests.pop(request_id, None)

    async def _notify(self, request: InteractionRequest) -> None:
        """Send notifications for interaction request."""
        recipients = []
        if request.assignee_id:
            recipients.append(request.assignee_id)

        # Send to all configured channels
        for channel in self._channels.values():
            try:
                await channel.send(request, recipients)
            except Exception as e:
                # Log but don't fail
                print(f"Failed to send notification: {e}")

    async def _escalate(self, request: InteractionRequest) -> None:
        """Escalate request to next level."""
        if not request.escalation_chain:
            return

        if request.current_escalation_level >= len(request.escalation_chain):
            # No more escalation levels
            return

        # Move to next level
        request.current_escalation_level += 1
        request.assignee_id = request.escalation_chain[request.current_escalation_level]
        request.status = InteractionStatus.ESCALATED
        request.updated_at = datetime.now()

        await self.store.update(request)
        await self._notify(request)
