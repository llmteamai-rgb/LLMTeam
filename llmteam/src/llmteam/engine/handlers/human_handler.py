"""
Human Task Handler.

Handles human-in-the-loop interactions.
Re-exports from original handlers.py for backwards compatibility.
"""

from typing import Any, Optional
import asyncio

from llmteam.runtime import StepContext
from llmteam.human import (
    HumanInteractionManager,
    InteractionType,
    InteractionStatus,
    MemoryInteractionStore,
)


class HumanTaskHandler:
    """
    Handler for human_task step type.

    Integrates with HumanInteractionManager to create and wait for
    human interactions.
    """

    def __init__(
        self,
        manager: Optional[HumanInteractionManager] = None,
        timeout_seconds: float = 86400,  # 24 hours default
    ) -> None:
        """
        Initialize handler.

        Args:
            manager: HumanInteractionManager instance. If None, creates one with MemoryStore.
            timeout_seconds: Default timeout for human tasks.
        """
        if manager is None:
            store = MemoryInteractionStore()
            manager = HumanInteractionManager(store)
        self.manager = manager
        self.timeout_seconds = timeout_seconds

    async def __call__(
        self,
        ctx: StepContext,
        config: dict[str, Any],
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute human task.

        Args:
            ctx: Step context
            config: Step configuration with:
                - task_type: "approval", "input", "review", "choice"
                - title: Task title
                - description: Task description
                - assignee_ref: Reference to assignee
                - timeout_hours: Timeout in hours
                - choices: List of choices (for choice type)
                - escalation_chain: Escalation chain
            input_data: Data to present to human

        Returns:
            Dict with output port data:
                - approved: Output if approved
                - rejected: Output if rejected
                - modified: Output if modified
        """
        task_type = config.get("task_type", "approval")
        title = config.get("title", "Human Task Required")
        description = config.get("description", "")
        assignee = config.get("assignee_ref", "")
        timeout_hours = config.get("timeout_hours", 24)
        choices = config.get("choices", [])

        # Map task_type to InteractionType
        type_mapping = {
            "approval": InteractionType.APPROVAL,
            "input": InteractionType.INPUT,
            "review": InteractionType.REVIEW,
            "choice": InteractionType.CHOICE,
        }
        interaction_type = type_mapping.get(task_type, InteractionType.APPROVAL)

        # Create request
        if interaction_type == InteractionType.APPROVAL:
            request = await self.manager.request_approval(
                title=title,
                description=description,
                run_id=ctx.run_id,
                pipeline_id=ctx.segment_id,
                agent_name=ctx.step_id,
                assignee=assignee,
                context_data=input_data,
            )
        elif interaction_type == InteractionType.CHOICE:
            # Convert simple choices list to options format
            options = [{"id": str(i), "label": c} for i, c in enumerate(choices)]
            request = await self.manager.request_choice(
                title=title,
                description=description,
                options=options,
                run_id=ctx.run_id,
                pipeline_id=ctx.segment_id,
                agent_name=ctx.step_id,
                step_name="choice",
            )
        elif interaction_type == InteractionType.INPUT:
            request = await self.manager.request_input(
                title=title,
                description=description,
                input_schema=config.get("input_schema", {}),
                run_id=ctx.run_id,
                pipeline_id=ctx.segment_id,
                agent_name=ctx.step_id,
                step_name="input",
            )
        else:
            # Default to approval for review
            request = await self.manager.request_approval(
                title=title,
                description=description,
                run_id=ctx.run_id,
                pipeline_id=ctx.segment_id,
                agent_name=ctx.step_id,
                assignee=assignee,
                context_data=input_data,
            )

        # Wait for response with timeout
        timeout = timeout_hours * 3600
        try:
            response = await asyncio.wait_for(
                self.manager.wait_for_response(request.request_id),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            # Timeout - escalate or fail
            raise TimeoutError(f"Human task timed out after {timeout_hours} hours")

        # Route to appropriate output port based on response
        if response.approved is True:
            return {
                "approved": {
                    "data": input_data,
                    "response_data": response.input_data,
                    "responder": response.responder_id,
                    "comment": response.comment,
                }
            }
        elif response.approved is False:
            return {
                "rejected": {
                    "data": input_data,
                    "response_data": response.input_data,
                    "responder": response.responder_id,
                    "reason": response.reason,
                    "comment": response.comment,
                }
            }
        else:
            # Modified or other (review changes, input data)
            return {
                "modified": {
                    "data": response.review_changes or response.input_data or input_data,
                    "original_data": input_data,
                    "responder": response.responder_id,
                    "comment": response.comment,
                }
            }


def create_human_task_handler(
    manager: Optional[HumanInteractionManager] = None,
) -> HumanTaskHandler:
    """
    Create a human task handler.

    Args:
        manager: Optional HumanInteractionManager instance.

    Returns:
        Configured HumanTaskHandler.
    """
    return HumanTaskHandler(manager)
