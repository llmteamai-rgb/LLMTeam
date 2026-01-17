"""
Simple Workflow Example.

Demonstrates conditional branching and multiple step types.

Usage:
    # With real LLM
    export OPENAI_API_KEY=sk-your-key
    python run.py

    # With mocks (no API key needed)
    python run.py --mock
"""

import asyncio
import argparse
import json
from pathlib import Path

from llmteam.canvas import (
    SegmentDefinition,
    StepDefinition,
    EdgeDefinition,
    SegmentRunner,
    validate_segment,
)
from llmteam.runtime import RuntimeContextFactory


def create_segment() -> SegmentDefinition:
    """Create the workflow segment."""
    return SegmentDefinition(
        segment_id="ticket_classifier",
        name="Support Ticket Classifier",
        description="Classifies support tickets and routes to appropriate handler",
        entrypoint="classify",
        steps=[
            # Step 1: Classify the ticket
            StepDefinition(
                step_id="classify",
                type="llm_agent",
                name="Classify Ticket",
                config={
                    "llm_ref": "default",
                    "prompt": """Classify this support ticket as 'urgent' or 'normal'.

Ticket: {ticket_text}

Respond with just one word: 'urgent' or 'normal'.""",
                },
            ),
            # Step 2: Check classification
            StepDefinition(
                step_id="check_priority",
                type="condition",
                name="Check Priority",
                config={
                    "field": "response",
                    "operator": "contains",
                    "value": "urgent",
                    "true_branch": "urgent_handler",
                    "false_branch": "normal_handler",
                },
            ),
            # Step 3a: Handle urgent tickets
            StepDefinition(
                step_id="urgent_handler",
                type="transform",
                name="Urgent Handler",
                config={
                    "mappings": {
                        "priority": "'HIGH'",
                        "escalate": "True",
                        "response_time": "'1 hour'",
                        "original_ticket": "ticket_text",
                    },
                },
            ),
            # Step 3b: Handle normal tickets
            StepDefinition(
                step_id="normal_handler",
                type="transform",
                name="Normal Handler",
                config={
                    "mappings": {
                        "priority": "'NORMAL'",
                        "escalate": "False",
                        "response_time": "'24 hours'",
                        "original_ticket": "ticket_text",
                    },
                },
            ),
            # Step 4: Format final output
            StepDefinition(
                step_id="format_output",
                type="transform",
                name="Format Output",
                config={
                    "expression": "input",
                },
            ),
        ],
        edges=[
            EdgeDefinition(from_step="classify", to_step="check_priority"),
            EdgeDefinition(from_step="check_priority", to_step="urgent_handler"),
            EdgeDefinition(from_step="check_priority", to_step="normal_handler"),
            EdgeDefinition(from_step="urgent_handler", to_step="format_output"),
            EdgeDefinition(from_step="normal_handler", to_step="format_output"),
        ],
    )


async def run_with_real_llm():
    """Run with real OpenAI LLM."""
    from llmteam.providers import OpenAIProvider

    # Create provider
    provider = OpenAIProvider(model="gpt-4o-mini")

    # Create runtime
    factory = RuntimeContextFactory()
    factory.register_llm("default", provider)

    runtime = factory.create_runtime(
        tenant_id="example",
        instance_id="workflow-001",
    )

    # Create and validate segment
    segment = create_segment()
    validation = validate_segment(segment)
    if not validation.is_valid:
        print("Validation errors:")
        for msg in validation.errors:
            print(f"  - {msg.message}")
        return

    # Run segment
    runner = SegmentRunner()
    result = await runner.run(
        segment=segment,
        input_data={
            "ticket_text": "URGENT: Production server is down! All customers affected!"
        },
        runtime=runtime,
    )

    print(f"\nStatus: {result.status}")
    print(f"Output: {json.dumps(result.output, indent=2)}")

    await provider.close()


async def run_with_mocks():
    """Run with mock providers (no API key needed)."""
    from llmteam.testing import MockLLMProvider, SegmentTestRunner, TestRunConfig

    print("Running with mock LLM provider...")

    # Create test runner with mock responses
    runner = SegmentTestRunner()
    runner.configure(
        TestRunConfig(
            mock_llm_responses=["urgent"],  # Mock classification response
        )
    )

    # Create segment
    segment = create_segment()

    # Run test
    result = await runner.run(
        segment=segment,
        input_data={
            "ticket_text": "URGENT: Production server is down! All customers affected!"
        },
    )

    print(f"\nStatus: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Output: {json.dumps(result.output, indent=2)}")
    print(f"Duration: {result.duration_ms:.2f}ms")

    if result.error:
        print(f"Error: {result.error}")


def main():
    parser = argparse.ArgumentParser(description="Run simple workflow example")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock providers instead of real LLM",
    )
    args = parser.parse_args()

    if args.mock:
        asyncio.run(run_with_mocks())
    else:
        asyncio.run(run_with_real_llm())


if __name__ == "__main__":
    main()
