# Simple Workflow Example

A more complete example demonstrating multiple step types and branching.

## Features

- Multiple step types (LLM, Transform, Condition)
- Conditional branching
- Mock providers for testing
- JSON Schema validation

## Installation

```bash
pip install llmteam-ai[providers]
```

## Run with Real LLM

```bash
export OPENAI_API_KEY=sk-your-key
python run.py
```

## Run with Mocks (no API key needed)

```bash
python run.py --mock
```

## Workflow Structure

```
[input] -> [classify] -> [condition]
                            |
              +-------------+-------------+
              |                           |
           [urgent]                   [normal]
              |                           |
              +-----------+---------------+
                          |
                      [format]
                          |
                       [output]
```

## Files

- `run.py` - Main execution script
- `workflow.json` - Workflow definition
