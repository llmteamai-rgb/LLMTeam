"""
LLMTeam Quickstart Example.

A minimal example demonstrating segment definition and execution.

Usage:
    export OPENAI_API_KEY=sk-your-key
    python simple.py
"""

import asyncio
from llmteam.canvas import (
    SegmentDefinition,
    StepDefinition,
    EdgeDefinition,
    SegmentRunner,
)
from llmteam.runtime import RuntimeContextFactory
from llmteam.providers import OpenAIProvider


async def main():
    # 1. Create LLM provider
    provider = OpenAIProvider(model="gpt-4o-mini")

    # 2. Create runtime context
    factory = RuntimeContextFactory()
    factory.register_llm("gpt4", provider)

    runtime = factory.create_runtime(
        tenant_id="quickstart",
        instance_id="example-001",
    )

    # 3. Define segment (workflow)
    segment = SegmentDefinition(
        segment_id="hello_world",
        name="Hello World Workflow",
        entrypoint="ask",
        steps=[
            StepDefinition(
                step_id="ask",
                type="llm_agent",
                name="Ask Question",
                config={
                    "llm_ref": "gpt4",
                    "prompt": "Answer this question briefly: {query}",
                },
            ),
            StepDefinition(
                step_id="format",
                type="transform",
                name="Format Output",
                config={
                    "expression": "response",
                },
            ),
        ],
        edges=[
            EdgeDefinition(from_step="ask", to_step="format"),
        ],
    )

    # 4. Run segment
    runner = SegmentRunner()
    result = await runner.run(
        segment=segment,
        input_data={"query": "What is artificial intelligence?"},
        runtime=runtime,
    )

    # 5. Print results
    print(f"Status: {result.status}")
    print(f"Output: {result.output}")

    # Clean up
    await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
