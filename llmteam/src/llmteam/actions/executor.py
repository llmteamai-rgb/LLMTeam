"""
Action Executor.

Executes external actions with integration to:
- RateLimiter (v1.7.0) for rate limiting
- AuditTrail (v1.7.0) for audit logging
- TenantContext (v1.7.0) for multi-tenancy
"""

from typing import Optional, TYPE_CHECKING

from .models import ActionContext, ActionResult, ActionStatus
from .registry import ActionRegistry

if TYPE_CHECKING:
    from ..ratelimit import RateLimitedExecutor
    from ..audit import AuditTrail, AuditEventType

from llmteam.licensing import professional_only


@professional_only
class ActionExecutor:
    """
    Executor for external actions.

    Integrates with:
    - RateLimiter from v1.7.0
    - AuditTrail from v1.7.0
    - TenantContext from v1.7.0
    """

    def __init__(
        self,
        registry: ActionRegistry,
        rate_limiter: Optional["RateLimitedExecutor"] = None,
        audit_trail: Optional["AuditTrail"] = None,
    ):
        self.registry = registry
        self.rate_limiter = rate_limiter
        self.audit_trail = audit_trail

    async def execute(
        self,
        action_name: str,
        context: ActionContext,
    ) -> ActionResult:
        """
        Execute an action.

        Args:
            action_name: Name of the action to execute
            context: Execution context

        Returns:
            ActionResult with execution outcome
        """
        # Check if action exists
        handler = self.registry.get_handler(action_name)
        if not handler:
            return ActionResult(
                action_name=action_name,
                status=ActionStatus.FAILED,
                error_message=f"Action '{action_name}' not found",
                error_type="ActionNotFound",
            )

        # Execute with rate limiting if configured
        if self.rate_limiter:
            try:
                result = await self.rate_limiter.execute(
                    action_name,
                    handler.execute,
                    context,
                )
            except Exception as e:
                # Rate limiter can throw errors (circuit open, etc.)
                result = ActionResult(
                    action_name=action_name,
                    status=ActionStatus.FAILED,
                    error_message=str(e),
                    error_type=type(e).__name__,
                )
        else:
            result = await handler.execute(context)

        # Log to audit trail if configured
        if self.audit_trail:
            await self._log_to_audit(context, result)

        return result

    async def _log_to_audit(
        self,
        context: ActionContext,
        result: ActionResult,
    ) -> None:
        """Log action execution to audit trail."""
        from ..audit import AuditEventType

        # Determine event type based on result status
        if result.status == ActionStatus.COMPLETED:
            event_type = AuditEventType.ACTION_COMPLETED
        elif result.status == ActionStatus.FAILED:
            event_type = AuditEventType.ACTION_FAILED
        elif result.status == ActionStatus.TIMEOUT:
            event_type = AuditEventType.ACTION_FAILED
        else:
            event_type = AuditEventType.ACTION_COMPLETED

        await self.audit_trail.log(
            event_type,
            actor_id=context.agent_name,
            resource_type="external_action",
            resource_id=context.action_name,
            success=(result.status == ActionStatus.COMPLETED),
            metadata={
                "status": result.status.value,
                "duration_ms": result.duration_ms,
                "response_code": result.response_code,
                "error_message": result.error_message if result.error_message else None,
                "correlation_id": context.correlation_id,
                "run_id": context.run_id,
            },
        )

    async def execute_batch(
        self,
        contexts: list[ActionContext],
    ) -> list[ActionResult]:
        """
        Execute multiple actions in parallel.

        Args:
            contexts: List of action contexts

        Returns:
            List of action results
        """
        import asyncio

        tasks = [
            self.execute(ctx.action_name, ctx)
            for ctx in contexts
        ]

        return await asyncio.gather(*tasks, return_exceptions=False)
