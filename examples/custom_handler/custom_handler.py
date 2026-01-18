"""
Custom Handler Example

Shows how to create and register custom step handlers.
"""

import asyncio
from typing import Any
from llmteam.canvas import SegmentDefinition, StepDefinition, EdgeDefinition, SegmentRunner
from llmteam.runtime import RuntimeContextFactory, StepContext


# Custom handler function
async def email_sender_handler(
    ctx: StepContext,
    config: dict[str, Any],
    input_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Custom handler that simulates sending an email.

    Config:
        template: Email template name
        subject_field: Input field containing subject
        body_field: Input field containing body

    Input:
        to: Recipient email address
        subject: Email subject (or field from subject_field)
        body: Email body (or field from body_field)

    Output:
        sent: Boolean indicating success
        message_id: Simulated message ID
    """
    template = config.get("template", "default")
    to = input_data.get("to", "")
    subject = input_data.get(config.get("subject_field", "subject"), "No Subject")
    body = input_data.get(config.get("body_field", "body"), "")

    # Simulate sending email
    print(f"[EmailHandler] Sending email to {to}")
    print(f"  Template: {template}")
    print(f"  Subject: {subject}")
    print(f"  Body: {body[:50]}...")

    # Simulate async operation
    await asyncio.sleep(0.1)

    return {
        "sent": True,
        "message_id": f"msg-{hash(to + subject) % 10000:04d}",
        "recipient": to,
    }


# Custom handler class
class DataValidatorHandler:
    """
    Custom handler that validates input data against a schema.

    Config:
        required_fields: List of required field names
        field_types: Dict mapping field names to expected types

    Input:
        Any data to validate

    Output:
        valid: Boolean indicating if validation passed
        errors: List of validation errors
    """

    STEP_TYPE = "data_validator"
    DISPLAY_NAME = "Data Validator"
    DESCRIPTION = "Validates input data against schema"
    CATEGORY = "validation"

    async def __call__(
        self,
        ctx: StepContext,
        config: dict[str, Any],
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        errors = []
        required_fields = config.get("required_fields", [])
        field_types = config.get("field_types", {})

        # Check required fields
        for field in required_fields:
            if field not in input_data:
                errors.append(f"Missing required field: {field}")
            elif input_data[field] is None:
                errors.append(f"Field '{field}' cannot be null")

        # Check field types
        type_map = {
            "string": str,
            "int": int,
            "float": (int, float),
            "bool": bool,
            "list": list,
            "dict": dict,
        }

        for field, expected_type in field_types.items():
            if field in input_data:
                value = input_data[field]
                if value is not None:
                    expected = type_map.get(expected_type, str)
                    if not isinstance(value, expected):
                        errors.append(
                            f"Field '{field}' expected {expected_type}, got {type(value).__name__}"
                        )

        return {
            "valid": len(errors) == 0,
            "errors": errors if errors else None,
            "data": input_data if not errors else None,
        }


async def main():
    factory = RuntimeContextFactory()
    runtime = factory.create_runtime(
        tenant_id="example",
        instance_id="custom-handler-1",
    )

    # Create segment with custom handlers
    segment = SegmentDefinition(
        segment_id="custom-handlers",
        name="Custom Handlers Example",
        entrypoint="validate",
        steps=[
            StepDefinition(
                step_id="validate",
                step_type="data_validator",
                name="Validate Input",
                config={
                    "required_fields": ["to", "subject", "body"],
                    "field_types": {
                        "to": "string",
                        "subject": "string",
                        "body": "string",
                    },
                },
            ),
            StepDefinition(
                step_id="check_valid",
                step_type="condition",
                name="Check Validation",
                config={
                    "expression": "valid == True",
                },
            ),
            StepDefinition(
                step_id="send_email",
                step_type="email_sender",
                name="Send Email",
                config={
                    "template": "notification",
                },
            ),
            StepDefinition(
                step_id="handle_error",
                step_type="transform",
                name="Handle Error",
                config={
                    "mapping": {
                        "success": "false",
                        "errors": "errors",
                    },
                },
            ),
        ],
        edges=[
            EdgeDefinition(from_step="validate", to_step="check_valid"),
            EdgeDefinition(from_step="check_valid", from_port="true", to_step="send_email"),
            EdgeDefinition(from_step="check_valid", from_port="false", to_step="handle_error"),
        ],
    )

    # Create runner and register custom handlers
    runner = SegmentRunner()
    runner.register_handler("email_sender", email_sender_handler)
    runner.register_handler("data_validator", DataValidatorHandler())

    # Test with valid input
    print("=== Test with valid input ===")
    result = await runner.run(
        segment=segment,
        input_data={
            "to": "user@example.com",
            "subject": "Hello from LLMTeam",
            "body": "This is a test email sent using a custom handler.",
        },
        runtime=runtime,
    )
    print(f"Result: {result.output}")

    # Test with invalid input
    print("\n=== Test with invalid input ===")
    result = await runner.run(
        segment=segment,
        input_data={
            "to": "user@example.com",
            # Missing subject and body
        },
        runtime=runtime,
    )
    print(f"Result: {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
