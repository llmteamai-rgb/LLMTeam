# Quick Start

This guide walks you through creating and running your first LLMTeam workflow.

## 1. Create a Simple Workflow

```python
from llmteam.canvas import SegmentDefinition, StepDefinition, EdgeDefinition, SegmentRunner
from llmteam.runtime import RuntimeContextFactory

# Create runtime context
factory = RuntimeContextFactory()
runtime = factory.create_runtime(
    tenant_id="my-tenant",
    instance_id="workflow-001",
)

# Define a simple 3-step workflow
segment = SegmentDefinition(
    segment_id="data-pipeline",
    name="Data Processing Pipeline",
    entrypoint="input",
    steps=[
        StepDefinition(
            step_id="input",
            type="transform",
            config={"expression": "input"},
        ),
        StepDefinition(
            step_id="process",
            type="llm_agent",
            config={
                "prompt": "Analyze the following data: {input}",
                "llm_ref": "gpt4",
            },
        ),
        StepDefinition(
            step_id="output",
            type="transform",
            config={"expression": "append_processed"},
        ),
    ],
    edges=[
        EdgeDefinition(from_step="input", to_step="process"),
        EdgeDefinition(from_step="process", to_step="output"),
    ],
)

# Run the workflow
runner = SegmentRunner()
result = await runner.run(
    segment=segment,
    input_data={"data": "Sales increased by 15% in Q4"},
    runtime=runtime,
)

print(f"Status: {result.status}")
print(f"Output: {result.output}")
```

## 2. Using JSON Definition

You can also define workflows as JSON:

```json
{
  "segment_id": "json-workflow",
  "name": "JSON Defined Workflow",
  "entrypoint": "start",
  "steps": [
    {"step_id": "start", "type": "transform", "config": {}},
    {"step_id": "analyze", "type": "llm_agent", "config": {"llm_ref": "gpt4"}}
  ],
  "edges": [
    {"from_step": "start", "to_step": "analyze"}
  ]
}
```

Load and run:

```python
import json
from llmteam.canvas import SegmentDefinition

with open("workflow.json") as f:
    data = json.load(f)

segment = SegmentDefinition.from_dict(data)
result = await runner.run(segment=segment, input_data={}, runtime=runtime)
```

## 3. Using the CLI

```bash
# Validate a workflow
llmteam validate workflow.json

# Run a workflow
llmteam run workflow.json --input '{"query": "Hello"}'

# Start API server
llmteam serve --port 8000
```

## Next Steps

- [Configuration](configuration.md) - Configure LLM providers and settings
- [Step Types](../steps/overview.md) - Learn about all available step types
- [API Reference](../api/rest.md) - REST API documentation
