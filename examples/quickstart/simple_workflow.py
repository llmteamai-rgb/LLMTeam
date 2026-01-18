#!/usr/bin/env python3
"""
Quickstart: Run your first workflow in 5 minutes.

Prerequisites:
    pip install llmteam-ai[providers]
    export OPENAI_API_KEY=sk-...
"""

import asyncio
import os
from dotenv import load_dotenv
from llmteam.canvas import SegmentDefinition, SegmentRunner
from llmteam.runtime import RuntimeContext
from llmteam.providers import OpenAIProvider

# Load environment variables
load_dotenv()


# 1. Define workflow as JSON
WORKFLOW = {
    "segment_id": "quickstart",
    "name": "My First Workflow",
    "version": "1.0",
    "entrypoint": "ask_ai",
    "steps": [
        {
            "step_id": "ask_ai",
            "type": "llm_agent",
            "name": "Ask AI",
            "config": {
                "llm_ref": "openai",
                "prompt": "Answer this question concisely: {question}"
            }
        }
    ]
}


async def main():
    # 2. Parse workflow
    segment = SegmentDefinition.from_dict(WORKFLOW)
    
    # 3. Setup runtime with LLM provider
    runtime = RuntimeContext(
        tenant_id="demo",
        instance_id="quickstart",
        run_id="run_001",
        segment_id=segment.segment_id,
    )
    
    # Use config from env or default
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Using dummy provider mode if supported.")
        
    runtime.llms.register("openai", OpenAIProvider(model="gpt-4o-mini", api_key=api_key))
    
    # 4. Run workflow
    runner = SegmentRunner()
    print("Running workflow...")
    result = await runner.run(
        segment=segment,
        runtime=runtime,
        input_data={"question": "What is the capital of France?"}
    )
    
    # 5. Print result
    print(f"Status: {result.status.value}")
    if result.status.value == "completed":
        print(f"Answer: {result.output.get('output', 'No output')}")
    else:
        print(f"Error: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
