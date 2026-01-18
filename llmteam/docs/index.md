# LLMTeam Documentation

**Enterprise AI Workflow Runtime** for building multi-agent LLM pipelines with security, orchestration, and workflow capabilities.

## Features

- **Canvas Workflow Engine** - Declarative JSON-based workflow definitions
- **12+ Step Types** - LLM, Transform, Condition, Loop, Parallel, Human Task, and more
- **7 LLM Providers** - OpenAI, Anthropic, Azure, Bedrock, Vertex, Ollama, LiteLLM
- **Enterprise Security** - Multi-tenancy, RBAC, Audit Trail, Secrets Management
- **RAG/KAG Integration** - Native and Proxy modes for context retrieval
- **Real-time Events** - WebSocket, SSE, Redis, Kafka transports

## Quick Install

```bash
pip install llmteam-ai
```

With optional dependencies:

```bash
pip install llmteam-ai[api]        # FastAPI server
pip install llmteam-ai[providers]  # All LLM providers
pip install llmteam-ai[all]        # Everything
```

## Quick Example

```python
from llmteam.canvas import SegmentDefinition, StepDefinition, EdgeDefinition, SegmentRunner
from llmteam.runtime import RuntimeContextFactory

# Create runtime
factory = RuntimeContextFactory()
runtime = factory.create_runtime(tenant_id="acme", instance_id="run-1")

# Define workflow
segment = SegmentDefinition(
    segment_id="greeting",
    name="Greeting Workflow",
    entrypoint="start",
    steps=[
        StepDefinition(step_id="start", type="transform", config={"expression": "input"}),
        StepDefinition(step_id="greet", type="llm_agent", config={"llm_ref": "gpt4"}),
    ],
    edges=[
        EdgeDefinition(from_step="start", to_step="greet"),
    ],
)

# Execute
runner = SegmentRunner()
result = await runner.run(segment=segment, input_data={"query": "Hello!"}, runtime=runtime)
print(result.output)
```

## Version

Current version: **2.2.1**

## License

Apache 2.0
