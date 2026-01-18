"""
Conditional Flow Example

Demonstrates branching workflows with condition and switch handlers.
"""

import asyncio
from llmteam.canvas import SegmentDefinition, StepDefinition, EdgeDefinition, SegmentRunner
from llmteam.runtime import RuntimeContextFactory


async def main():
    factory = RuntimeContextFactory()
    runtime = factory.create_runtime(
        tenant_id="example",
        instance_id="conditional-1",
    )

    # Segment with conditional branching
    segment = SegmentDefinition(
        segment_id="conditional-flow",
        name="Conditional Flow Example",
        description="Routes requests based on action type",
        entrypoint="check_action",
        steps=[
            # Check the action type
            StepDefinition(
                step_id="check_action",
                step_type="switch",
                name="Route by Action",
                config={
                    "expression": "action",
                    "cases": {
                        "create": "handle_create",
                        "update": "handle_update",
                        "delete": "handle_delete",
                    },
                    "default": "handle_unknown",
                },
            ),
            # Handle create action
            StepDefinition(
                step_id="handle_create",
                step_type="transform",
                name="Handle Create",
                config={
                    "mapping": {
                        "result": "data",
                        "action_taken": "'created'",
                    }
                },
            ),
            # Handle update action
            StepDefinition(
                step_id="handle_update",
                step_type="transform",
                name="Handle Update",
                config={
                    "mapping": {
                        "result": "data",
                        "action_taken": "'updated'",
                    }
                },
            ),
            # Handle delete action
            StepDefinition(
                step_id="handle_delete",
                step_type="transform",
                name="Handle Delete",
                config={
                    "mapping": {
                        "id": "id",
                        "action_taken": "'deleted'",
                    }
                },
            ),
            # Handle unknown action
            StepDefinition(
                step_id="handle_unknown",
                step_type="transform",
                name="Handle Unknown",
                config={
                    "mapping": {
                        "error": "'Unknown action'",
                        "action_taken": "'none'",
                    }
                },
            ),
        ],
        edges=[
            EdgeDefinition(from_step="check_action", from_port="handle_create", to_step="handle_create"),
            EdgeDefinition(from_step="check_action", from_port="handle_update", to_step="handle_update"),
            EdgeDefinition(from_step="check_action", from_port="handle_delete", to_step="handle_delete"),
            EdgeDefinition(from_step="check_action", from_port="handle_unknown", to_step="handle_unknown"),
        ],
    )

    # Test with different actions
    test_cases = [
        {"action": "create", "data": {"name": "New Item"}},
        {"action": "update", "data": {"name": "Updated Item"}, "id": 123},
        {"action": "delete", "id": 456},
        {"action": "unknown"},
    ]

    runner = SegmentRunner()

    for test_input in test_cases:
        print(f"\nInput: {test_input}")
        result = await runner.run(
            segment=segment,
            input_data=test_input,
            runtime=runtime,
        )
        print(f"Output: {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
