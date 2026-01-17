# LLMTeam Quickstart

Get started with LLMTeam in 5 minutes.

## Installation

```bash
pip install llmteam-ai[providers]
```

## Set up API Key

```bash
# For OpenAI
export OPENAI_API_KEY=sk-your-key-here

# Or for Anthropic
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

## Run the Example

```bash
python simple.py
```

## What's Happening

1. **Define a Segment** - A workflow with steps and edges
2. **Create Runtime** - Context with LLM provider
3. **Run Segment** - Execute the workflow
4. **Get Results** - Output from the final step

## Files

- `simple.py` - Minimal Python example
- `workflow.json` - Same workflow as JSON

## Next Steps

- Check out `../simple_workflow/` for a more complete example
- See `../fastapi_server/` for REST API integration
