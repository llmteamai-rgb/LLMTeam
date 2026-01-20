# llmteam-ai

Enterprise AI Workflow Runtime for building multi-agent LLM pipelines.

[![PyPI version](https://badge.fury.io/py/llmteam-ai.svg)](https://pypi.org/project/llmteam-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Current Version: v4.0.0 — Typed Agent Architecture

### Key Features

- **Three Agent Types** — LLM, RAG, KAG (config-driven, no custom agent classes)
- **Simple API** — Create agents via dict, no boilerplate
- **SegmentRunner Integration** — LLMTeam uses Canvas runtime internally
- **LLMGroup** — Multi-team coordination with automatic routing
- **Presets** — Ready-to-use orchestrator, summarizer, reviewer configs

## Installation

```bash
pip install llmteam-ai

# With optional dependencies
pip install llmteam-ai[api]       # FastAPI server
pip install llmteam-ai[postgres]  # PostgreSQL stores
pip install llmteam-ai[all]       # Everything
```

## Quick Start

### Create a Team with Agents

```python
from llmteam import LLMTeam

# Simple: dict-based config
team = LLMTeam(
    team_id="content",
    agents=[
        {"type": "rag", "role": "retriever", "collection": "docs", "top_k": 5},
        {"type": "llm", "role": "writer", "prompt": "Based on context, write about: {query}"},
    ]
)

# Run
result = await team.run({"query": "AI trends in 2026"})
print(result.output)
```

### Add Agents Dynamically

```python
team = LLMTeam(team_id="support")

# Method 1: Dict
team.add_agent({
    "type": "llm",
    "role": "triage",
    "prompt": "Classify this query: {query}",
    "model": "gpt-4o-mini",
})

# Method 2: Shortcut
team.add_llm_agent(
    role="resolver",
    prompt="Resolve the issue: {issue}",
    temperature=0.3,
)

# Method 3: RAG/KAG
team.add_rag_agent(role="knowledge", collection="faq", top_k=3)
team.add_kag_agent(role="graph", max_hops=2)
```

### Use Presets

```python
from llmteam.agents import create_orchestrator_config, create_summarizer_config

# Orchestrator for adaptive flow
team.add_agent(create_orchestrator_config(
    available_agents=["writer", "editor", "reviewer"],
    model="gpt-4o-mini",
))

# Summarizer preset
team.add_agent(create_summarizer_config(role="summarizer"))
```

### Multi-Team Groups

```python
from llmteam import LLMTeam

research_team = LLMTeam(team_id="research", agents=[...])
writing_team = LLMTeam(team_id="writing", agents=[...])

# Create group with leader
group = research_team.create_group(
    group_id="content_pipeline",
    teams=[writing_team],
)

result = await group.run({"topic": "Quantum Computing"})
```

### Execution Control

```python
# Start
result = await team.run({"query": "..."})

# Pause and resume
snapshot = await team.pause()
# ... later ...
result = await team.resume(snapshot)

# Cancel
await team.cancel()
```

## Agent Types

| Type | Purpose | Key Config |
|------|---------|------------|
| `llm` | Text generation | `prompt`, `model`, `temperature`, `max_tokens` |
| `rag` | Vector retrieval | `collection`, `top_k`, `score_threshold` |
| `kag` | Knowledge graph | `max_hops`, `max_entities` |

### LLM Agent Config

```python
{
    "type": "llm",
    "role": "writer",              # Required: unique ID
    "prompt": "Write: {topic}",    # Required: prompt template
    "model": "gpt-4o-mini",        # Default: gpt-4o-mini
    "temperature": 0.7,            # Default: 0.7
    "max_tokens": 1000,            # Default: 1000
    "system_prompt": "You are...", # Optional
    "use_context": True,           # Use RAG/KAG context
    "output_format": "text",       # "text" | "json"
}
```

### RAG Agent Config

```python
{
    "type": "rag",
    "role": "retriever",
    "collection": "documents",     # Vector store collection
    "top_k": 5,                    # Number of results
    "score_threshold": 0.7,        # Minimum similarity
    "mode": "native",              # "native" | "proxy"
}
```

### KAG Agent Config

```python
{
    "type": "kag",
    "role": "knowledge",
    "max_hops": 2,                 # Graph traversal depth
    "max_entities": 10,            # Max entities to return
    "include_relations": True,     # Include relationships
}
```

## Flow Definition

```python
# Sequential (default)
team = LLMTeam(team_id="seq", flow="sequential")

# String syntax
team = LLMTeam(team_id="pipe", flow="retriever -> writer -> editor")

# Parallel
team = LLMTeam(team_id="par", flow="a, b -> c")  # a and b run parallel, then c

# DAG with conditions
team = LLMTeam(team_id="dag", flow={
    "edges": [
        {"from": "retriever", "to": "writer"},
        {"from": "writer", "to": "reviewer"},
        {"from": "reviewer", "to": "writer", "condition": "rejected"},
        {"from": "reviewer", "to": "publisher", "condition": "approved"},
    ]
})

# Adaptive (with orchestrator)
team = LLMTeam(team_id="adaptive", orchestration=True)
```

## Context Modes

```python
from llmteam import LLMTeam, ContextMode

# Shared context (default) - all agents see all results
team = LLMTeam(team_id="shared", context_mode=ContextMode.SHARED)

# Not shared - each agent gets only explicitly delivered context
team = LLMTeam(team_id="isolated", context_mode=ContextMode.NOT_SHARED)
```

## License Tiers

| Feature | Community | Professional | Enterprise |
|---------|-----------|--------------|------------|
| LLM/RAG/KAG agents | ✅ | ✅ | ✅ |
| Memory stores | ✅ | ✅ | ✅ |
| Canvas runner | ✅ | ✅ | ✅ |
| Process mining | ❌ | ✅ | ✅ |
| PostgreSQL stores | ❌ | ✅ | ✅ |
| Human-in-the-loop | ❌ | ✅ | ✅ |
| Multi-tenant | ❌ | ❌ | ✅ |
| Audit trail | ❌ | ❌ | ✅ |
| SSO/SAML | ❌ | ❌ | ✅ |

## Migration from v3.x

v4.0.0 is a **breaking change**. Key differences:

| v3.x | v4.x |
|------|------|
| `class Agent` with `process()` | Dict config |
| `team.register_agent(agent)` | `team.add_agent(config)` |
| `TeamOrchestrator` class | `flow` parameter or `orchestration=True` |
| Custom agent classes | External logic pattern |
| `result.agents_invoked` | `result.agents_called` |

### Migration Example

```python
# ═══════════════════════════════════════════
# v3.x (old) - Custom agent class
# ═══════════════════════════════════════════
from llmteam import Agent, AgentState, AgentResult

class WriterAgent(Agent):
    async def process(self, state: AgentState) -> AgentResult:
        query = state.data.get("query", "")
        # Custom logic here
        return AgentResult(output={"text": f"Article about {query}"})

team = LLMTeam(team_id="content")
team.register_agent(WriterAgent("writer"))

# ═══════════════════════════════════════════
# v4.x (new) - Dict config
# ═══════════════════════════════════════════
from llmteam import LLMTeam

team = LLMTeam(
    team_id="content",
    agents=[
        {"type": "llm", "role": "writer", "prompt": "Write article about: {query}"}
    ]
)

# For custom logic, use external pattern:
result = await team.run({"query": "AI"})
processed = my_custom_function(result.output)
```

## API Reference

### LLMTeam

```python
class LLMTeam:
    def __init__(
        self,
        team_id: str,
        agents: List[Dict] = None,
        flow: Union[str, Dict] = "sequential",
        model: str = "gpt-4o-mini",
        context_mode: ContextMode = ContextMode.SHARED,
        orchestration: bool = False,
        timeout: int = None,
    ): ...

    def add_agent(self, config: Dict) -> BaseAgent: ...
    def add_llm_agent(self, role: str, prompt: str, **kwargs) -> BaseAgent: ...
    def add_rag_agent(self, role: str = "rag", **kwargs) -> BaseAgent: ...
    def add_kag_agent(self, role: str = "kag", **kwargs) -> BaseAgent: ...
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]: ...
    def list_agents(self) -> List[BaseAgent]: ...

    async def run(self, input_data: Dict, run_id: str = None) -> RunResult: ...
    async def pause(self) -> TeamSnapshot: ...
    async def resume(self, snapshot: TeamSnapshot) -> RunResult: ...
    async def cancel(self) -> bool: ...

    def create_group(self, group_id: str, teams: List[LLMTeam]) -> LLMGroup: ...
    def to_config(self) -> Dict: ...
    @classmethod
    def from_config(cls, config: Dict) -> LLMTeam: ...
```

### RunResult

```python
@dataclass
class RunResult:
    success: bool
    status: RunStatus  # COMPLETED, FAILED, PAUSED, CANCELLED, TIMEOUT
    output: Dict[str, Any]
    final_output: Any
    agents_called: List[str]
    iterations: int
    duration_ms: int
    error: Optional[str]
    started_at: datetime
    completed_at: datetime
```

## Documentation

- [Full Documentation](https://docs.llmteam.ai)
- [API Reference](https://docs.llmteam.ai/api)
- [Examples](https://github.com/llmteamai/llmteam/tree/main/examples)
- [Changelog](https://github.com/llmteamai/llmteam/blob/main/CHANGELOG.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
