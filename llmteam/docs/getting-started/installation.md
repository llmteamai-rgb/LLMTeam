# Installation

## Requirements

- Python 3.10 or higher
- pip or poetry

## Basic Installation

```bash
pip install llmteam-ai
```

## Optional Dependencies

### API Server (FastAPI)

```bash
pip install llmteam-ai[api]
```

### LLM Providers

```bash
pip install llmteam-ai[providers]
```

Individual providers:

```bash
pip install llmteam-ai[openai]     # OpenAI
pip install llmteam-ai[anthropic]  # Anthropic
pip install llmteam-ai[azure]      # Azure OpenAI
pip install llmteam-ai[bedrock]    # AWS Bedrock
pip install llmteam-ai[vertex]     # Google Vertex AI
```

### Database Support

```bash
pip install llmteam-ai[postgres]   # PostgreSQL stores
pip install llmteam-ai[redis]      # Redis transport
```

### All Features

```bash
pip install llmteam-ai[all]
```

## Development Installation

```bash
git clone https://github.com/llmteamai-rgb/LLMTeam.git
cd LLMTeam/llmteam
pip install -e ".[dev]"
```

## Verify Installation

```python
import llmteam
print(llmteam.__version__)
```

Or via CLI:

```bash
llmteam --version
```
