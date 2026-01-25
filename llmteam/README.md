# llmteam-ai

Enterprise AI Workflow Runtime for building multi-agent LLM pipelines.

[![PyPI version](https://badge.fury.io/py/llmteam-ai.svg)](https://pypi.org/project/llmteam-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Current Version: v5.5.0

### Key Features

- **Three Agent Types** — LLM, RAG, KAG (config-driven, no custom agent classes)
- **Simple API** — Create agents via dict, no boilerplate
- **Quality Slider** — Single 0-100 parameter controls quality/cost tradeoff everywhere
- **SegmentRunner Integration** — LLMTeam uses Canvas runtime internally
- **LLMGroup** — Multi-team coordination with automatic routing
- **Presets** — Ready-to-use orchestrator, summarizer, reviewer configs
- **Retry & Circuit Breaker** — Per-agent retry policies with configurable backoff
- **Cost Tracking** — Built-in token usage tracking and budget management
- **Streaming** — Async generator-based event streaming for real-time progress
- **Tool/Function Calling** — `@tool` decorator, ToolExecutor, OpenAI-compatible schemas
- **Lifecycle Management** — Opt-in team state machine (configure → ready → run)
- **Dynamic Team Builder** — LLM analyzes task and creates optimal team automatically

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

## Retry & Circuit Breaker (v5.3.0)

```python
from llmteam import LLMTeam

team = LLMTeam(team_id="resilient", orchestration=True)

# Per-agent retry with exponential backoff
team.add_agent({
    "type": "llm",
    "role": "analyst",
    "prompt": "Analyze: {query}",
    "retry_policy": {
        "max_retries": 3,
        "backoff": "exponential",
        "base_delay": 1.0,
        "max_delay": 30.0,
    },
    "circuit_breaker": {
        "failure_threshold": 5,
        "recovery_timeout_seconds": 60.0,
    },
})
```

## Cost Tracking & Budgets (v5.3.0)

```python
from llmteam import LLMTeam

# Team with budget enforcement
team = LLMTeam(team_id="production", orchestration=True, max_cost_per_run=5.0)

result = await team.run({"query": "..."})
print(result.summary["cost"])  # RunCost with total, per-agent breakdown

# Custom pricing
from llmteam import PricingRegistry, ModelPricing
registry = PricingRegistry()
registry.register("my-model", ModelPricing(input_per_1k=0.005, output_per_1k=0.01))
```

## Streaming (v5.3.0)

```python
from llmteam import LLMTeam, StreamEventType

team = LLMTeam(team_id="stream", orchestration=True)
team.add_agent({"type": "llm", "role": "writer", "prompt": "..."})

async for event in team.stream({"query": "Hello"}):
    if event.type == StreamEventType.AGENT_COMPLETED:
        print(f"Agent {event.agent_id}: {event.data['output']}")
    elif event.type == StreamEventType.COST_UPDATE:
        print(f"Cost: ${event.data['current_cost']:.4f}")
    elif event.type == StreamEventType.RUN_COMPLETED:
        print(f"Done: {event.data['output']}")
```

## Tool/Function Calling (v5.3.0)

```python
from llmteam import tool, LLMTeam, ToolExecutor

@tool(description="Get weather for a city")
def get_weather(city: str, units: str = "celsius") -> str:
    return f"Weather in {city}: 22°{units[0].upper()}"

# Per-agent tools
team = LLMTeam(team_id="tools", orchestration=True)
team.add_agent({
    "type": "llm",
    "role": "assistant",
    "prompt": "Help the user",
    "tools": [get_weather.tool_definition],
})

# Get OpenAI-compatible schemas
agent = team.get_agent("assistant")
schemas = agent.tool_executor.get_schemas()
```

## Lifecycle Management (v5.3.0)

```python
from llmteam import LLMTeam, ConfigurationProposal

# Opt-in lifecycle enforcement
team = LLMTeam(team_id="prod", orchestration=True, enforce_lifecycle=True)
team.add_agent({"type": "llm", "role": "agent1", "prompt": "..."})

# Configure → Ready → Run
team.mark_configuring()
team.lifecycle.add_proposal(ConfigurationProposal(
    proposal_id="p-1", changes={"model": "gpt-4o"}, reason="Better accuracy"
))
team.lifecycle.approve_all()
team.mark_ready()

result = await team.run({"query": "test"})
assert team.state == "completed"

# Re-run
team.mark_ready()
result2 = await team.run({"query": "test2"})
```

## Quality Slider (v5.5.0)

Single 0-100 parameter controls quality/cost tradeoff for ALL LLM calls:

```python
from llmteam import LLMTeam, QualityManager

# Create team with quality setting
team = LLMTeam(
    team_id="content",
    agents=[{"type": "llm", "role": "writer", "prompt": "Write: {query}"}],
    quality=70,              # 0-100 (higher = better quality, higher cost)
    max_cost_per_run=1.00,   # Optional budget limit
)

# Or use preset names
team = LLMTeam(team_id="fast", quality="draft")       # quality=20
team = LLMTeam(team_id="prod", quality="production")  # quality=75

# Get cost estimate BEFORE running
estimate = await team.estimate_cost(complexity="medium")
print(f"Estimated: ${estimate.min_cost:.4f} - ${estimate.max_cost:.4f}")

# Run with quality override
result = await team.run({"query": "..."}, quality=90)      # Override for this run
result = await team.run({"query": "..."}, importance="high")  # +20 quality boost

# Stream with quality
async for event in team.stream({"query": "..."}, quality=80):
    print(event)
```

### Quality Levels

| Quality | Preset | Model | Temperature | Max Tokens | Use Case |
|---------|--------|-------|-------------|------------|----------|
| 0-30 | draft, economy | gpt-4o-mini | 0.3 | 500 | Quick iteration, testing |
| 30-70 | balanced | gpt-4o | 0.5 | 1000 | Standard production |
| 70-100 | production, best | gpt-4-turbo | 0.7 | 2000 | High-quality output |

### Quality Integration (v5.5.0)

Quality now affects ALL LLM calls across the system:

```python
# ConfigurationSession - quality-aware task analysis
session = await team.configure(task="Generate LinkedIn posts")
session.set_quality(80)  # High quality for configuration LLM calls

# TeamOrchestrator - quality-aware routing decisions
team = LLMTeam(team_id="router", orchestration=True, quality=70)
# Orchestrator's decide_next_agent() uses quality-appropriate model

# GroupOrchestrator - quality-aware coordination
from llmteam.orchestration import GroupOrchestrator
group = GroupOrchestrator(group_id="multi", quality=75)

# DynamicTeamBuilder - quality-aware blueprint generation
from llmteam.builder import DynamicTeamBuilder
builder = DynamicTeamBuilder(quality=80)
blueprint = await builder.analyze_task("Research AI trends")
```

### Pre-flight Budget Check (v5.5.0)

```python
team = LLMTeam(
    team_id="budget",
    quality=90,              # High quality = higher estimated cost
    max_cost_per_run=0.10,   # Low budget
)

# run() estimates cost BEFORE execution
result = await team.run({"query": "..."})
# If estimated cost > budget: fails immediately with
# "Pre-flight budget check failed: estimated cost $X would exceed budget"
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

## Migration from v3.x / v4.x

v5.x builds on v4.0.0 with non-breaking additions. Key v3.x → v4.x+ differences:

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
        quality: Union[int, str] = 50,         # v5.5.0: Quality slider 0-100 or preset
        max_cost_per_run: float = None,        # v5.3.0: Budget limit
        enforce_lifecycle: bool = False,       # v5.3.0: Opt-in lifecycle
    ): ...

    def add_agent(self, config: Dict) -> BaseAgent: ...
    def add_llm_agent(self, role: str, prompt: str, **kwargs) -> BaseAgent: ...
    def add_rag_agent(self, role: str = "rag", **kwargs) -> BaseAgent: ...
    def add_kag_agent(self, role: str = "kag", **kwargs) -> BaseAgent: ...
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]: ...
    def list_agents(self) -> List[BaseAgent]: ...

    # v5.5.0: Quality-aware execution
    async def run(
        self,
        input_data: Dict,
        run_id: str = None,
        quality: Union[int, str] = None,       # Override quality for this run
        importance: str = None,                # "high"|"medium"|"low" adjusts quality
    ) -> RunResult: ...
    async def stream(
        self,
        input_data: Dict,
        run_id: str = None,
        quality: Union[int, str] = None,
        importance: str = None,
    ) -> AsyncIterator[StreamEvent]: ...
    async def estimate_cost(self, complexity: str = "medium") -> CostEstimate: ...  # v5.5.0
    async def pause(self) -> TeamSnapshot: ...
    async def resume(self, snapshot: TeamSnapshot) -> RunResult: ...
    async def cancel(self) -> bool: ...

    # v5.5.0: Quality management
    @property
    def quality(self) -> int: ...
    @quality.setter
    def quality(self, value: Union[int, str]) -> None: ...
    def get_quality_manager(self) -> QualityManager: ...

    # v5.3.0: Lifecycle
    def mark_configuring(self) -> None: ...
    def mark_ready(self) -> None: ...
    @property
    def state(self) -> Optional[str]: ...
    @property
    def lifecycle(self) -> Optional[TeamLifecycle]: ...
    @property
    def cost_tracker(self) -> CostTracker: ...
    @property
    def budget_manager(self) -> Optional[BudgetManager]: ...

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
