"""
Function Action Handler.

Handles Python function calls with:
- Sync and async function support
- Error handling
- Timing tracking
"""

import asyncio
from datetime import datetime
from typing import Callable

from ..models import ActionConfig, ActionContext, ActionResult, ActionStatus, ActionHandler


class FunctionActionHandler(ActionHandler):
    """Handler for Python function calls."""

    def __init__(self, config: ActionConfig, func: Callable):
        self.config = config
        self.func = func

    async def execute(self, context: ActionContext) -> ActionResult:
        """Execute Python function."""
        started_at = datetime.now()

        try:
            # Check if function is async
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(context.input_data, context.pipeline_state)
            else:
                # Run sync function in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self.func, context.input_data, context.pipeline_state
                )

            completed_at = datetime.now()
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)

            return ActionResult(
                action_name=self.config.name,
                status=ActionStatus.COMPLETED,
                response_data=result,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
            )

        except Exception as e:
            completed_at = datetime.now()
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)

            return ActionResult(
                action_name=self.config.name,
                status=ActionStatus.FAILED,
                error_message=str(e),
                error_type=type(e).__name__,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
            )
