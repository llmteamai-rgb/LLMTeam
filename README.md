# llmteam-ai

Enterprise AI Workflow Runtime for building multi-agent LLM pipelines with security, orchestration, and workflow capabilities.

[![PyPI version](https://badge.fury.io/py/llmteam-ai.svg)](https://pypi.org/project/llmteam-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Current Version: v5.4.0 — Agentic Execution

### New in v5.4.0

- **RFC-021: DynamicTeamBuilder** — LLM analyzes your task, designs a team of agents with tools, and executes it
- **RFC-015: Provider Function Calling** — `complete_with_tools()`, `LLMResponse`, `ToolCall`, `ToolMessage`
- **RFC-016: Agent Tool Loop** — LLMAgent executes multi-round tool calling (configurable `max_tool_rounds`)
- **RFC-017: Tool Stream Events** — `TOOL_CALL`, `TOOL_RESULT`, `AGENT_THINKING` events during execution
- **RFC-018: Built-in Tools** — `web_search`, `http_fetch`, `json_extract`, `text_summarize`, `code_eval`
- **RFC-019: Period Budgets** — `PeriodBudgetManager` for hour/day/month cost limits
- **RFC-020: Retry-After** — Automatic retry-after handling from LLM providers

---

## What LLMTeam Is

LLMTeam is an **AI agent orchestration library** that enables:

- Creating teams of typed AI agents (LLM, RAG, KAG) with defined roles
- Dynamic team building from natural language task descriptions
- Agentic tool-use loops with streaming events
- Orchestrating agent interactions with ROUTER/PASSIVE/FULL modes
- Enterprise-grade security, audit, cost tracking, and multi-tenancy

---

## Installation

```bash
pip install llmteam-ai

# With LLM providers (OpenAI, Anthropic)
pip install llmteam-ai[providers]

# With all optional dependencies
pip install llmteam-ai[all]
```

---

## Quick Start

### 1. Create a Team with Typed Agents

```python
from llmteam import LLMTeam

# Create team with typed agents (LLM, RAG, KAG)
team = LLMTeam(
    team_id="support",
    agents=[
        {"type": "llm", "role": "triage", "prompt": "Categorize this ticket: {query}"},
        {"type": "llm", "role": "resolver", "prompt": "Resolve this issue: {query}"},
    ],
    flow="triage -> resolver",  # DAG flow
    quality=70,                 # Quality slider 0-100
)

result = await team.run({"query": "I can't login to my account"})
print(result.output)
```

### 2. Agents with Tools (Agentic Execution)

```python
from llmteam import LLMTeam
from llmteam.tools.builtin import web_search, code_eval

team = LLMTeam(
    team_id="research",
    agents=[
        {
            "type": "llm",
            "role": "researcher",
            "prompt": "Research this topic: {query}",
            "tools": [web_search.tool_definition],
            "max_tool_rounds": 5,
        },
        {
            "type": "llm",
            "role": "analyst",
            "prompt": "Analyze these findings: {query}",
            "tools": [code_eval.tool_definition],
        },
    ],
    orchestration=True,  # ROUTER mode: orchestrator picks the right agent
)

# Stream events during execution
async for event in team.stream({"query": "AI market size 2025"}):
    if event.type.value == "tool_call":
        print(f"  Tool: {event.data['tool_name']}({event.data['arguments']})")
    elif event.type.value == "agent_completed":
        print(f"  Agent {event.agent_id}: {event.data['output'][:100]}")
```

### 3. Dynamic Team Builder (RFC-021)

```python
from llmteam.builder import DynamicTeamBuilder

# LLM designs a team from your task description
builder = DynamicTeamBuilder(model="gpt-4o-mini")

# Analyze task → get blueprint
blueprint = await builder.analyze_task(
    "Research AI trends, calculate statistics, and summarize findings"
)
# Blueprint: 3 agents (researcher, calculator, summarizer) with appropriate tools

# Build and execute
team = builder.build_team(blueprint)
await builder.execute(team, {"query": "Latest LLM breakthroughs"})

# Or run the full interactive CLI
await builder.run_interactive()
```

### 4. ROUTER Mode (Orchestrator Picks Agents)

```python
from llmteam import LLMTeam

team = LLMTeam(
    team_id="helpdesk",
    agents=[
        {"type": "llm", "role": "billing", "prompt": "Handle billing: {query}"},
        {"type": "llm", "role": "technical", "prompt": "Handle tech: {query}"},
        {"type": "llm", "role": "general", "prompt": "Handle general: {query}"},
    ],
    orchestration=True,  # Orchestrator LLM routes to the right agent
)

# Orchestrator automatically routes to "billing" agent
result = await team.run({"query": "Why was I charged twice?"})
```

### 5. Cost Tracking & Budgets

```python
from llmteam import LLMTeam

team = LLMTeam(
    team_id="content",
    agents=[{"type": "llm", "role": "writer", "prompt": "Write about: {query}"}],
    quality=50,              # Controls model selection & cost
    max_cost_per_run=0.50,   # Hard cost limit per execution
)

# Get cost estimate before running
estimate = await team.estimate_cost(complexity="medium")
print(f"Estimated: ${estimate.min_cost:.2f} - ${estimate.max_cost:.2f}")

# Period budgets (hourly/daily/monthly limits)
from llmteam.cost import PeriodBudgetManager, BudgetPeriod

budget = PeriodBudgetManager(max_cost=100.0, period=BudgetPeriod.DAILY)
```

---

## CLI Usage

```bash
llmteam --version              # Show version
llmteam catalog                # List step types
llmteam validate segment.json  # Validate workflow
llmteam run segment.json       # Run workflow
llmteam providers              # List LLM providers
llmteam serve --port 8000      # Start API server
```

### Dynamic Builder Demo

```bash
export OPENAI_API_KEY="sk-..."
python examples/dynamic_demo.py
```

---

## Features by Version

### v5.4.0 — Agentic Execution (Current)

| Feature | Description |
|---------|-------------|
| **DynamicTeamBuilder** | LLM-powered automatic team creation from task descriptions |
| **Provider Function Calling** | `complete_with_tools()` on all providers |
| **Agent Tool Loop** | Multi-round tool execution with `max_tool_rounds` |
| **Built-in Tools** | 5 ready-to-use tools (web_search, http_fetch, etc.) |
| **Tool Stream Events** | Real-time TOOL_CALL/TOOL_RESULT events |
| **Period Budgets** | Hour/day/month cost management |
| **Retry-After** | Automatic provider rate-limit handling |

### v5.3.0 — Enterprise Features

| Feature | Description |
|---------|-------------|
| **Per-agent Retry** | RetryPolicy & CircuitBreakerPolicy per agent |
| **Cost Tracking** | CostTracker, BudgetManager, PricingRegistry |
| **Streaming** | StreamEvent, StreamEventType for real-time events |
| **Team Lifecycle** | TeamState, ConfigurationProposal, TeamLifecycle |

### v5.1.0 — Quality Slider

- Quality slider (0-100) controls model selection and cost
- Presets: draft, economy, balanced, production, best
- CostEstimator for pre-run cost prediction

### v4.1.0 — TeamOrchestrator

- TeamOrchestrator as separate supervisor entity (not an agent)
- OrchestratorMode: PASSIVE, ACTIVE (ROUTER), FULL
- AgentReport for automatic agent-to-orchestrator reporting

### v4.0.0 — Typed Agents

- Only 3 agent types: LLM, RAG, KAG (no custom Agent classes)
- AgentFactory with config-driven creation
- DAG flow support (string or dict)

### v2.x — Canvas & Runtime

- SegmentRunner workflow execution
- RuntimeContext resource injection
- Worktrail events, 12 built-in step handlers
- LLM providers (OpenAI, Anthropic, Azure, Bedrock, Vertex, Ollama)

### v1.x — Security Foundation

- Multi-tenant isolation
- Audit trail with SHA-256 chain
- Context security (sealed data, visibility levels)
- Rate limiting, circuit breakers

---

## Architecture

```
DynamicTeamBuilder              — Task analysis → TeamBlueprint → LLMTeam
       │
       ▼
LLMTeam + TeamOrchestrator      — Agent container + supervisor (ROUTER/PASSIVE/FULL)
       │
       ├─ LLMAgent (tools)      — Tool execution loop (max_tool_rounds)
       ├─ RAGAgent              — Retrieval-augmented generation
       └─ KAGAgent              — Knowledge graph queries
       │
       ▼
StreamEvent                     — Real-time events (TOOL_CALL, AGENT_COMPLETED, etc.)
       │
       ▼
Canvas (ExecutionEngine)        — Workflow routing (edges, conditions, DAG)
```

### Module Structure

```
llmteam/
├── builder/          # RFC-021: DynamicTeamBuilder, TeamBlueprint
├── agents/           # Typed agents (LLM/RAG/KAG), orchestrator, factory
├── team/             # LLMTeam, LLMGroup, RunResult, lifecycle
├── tools/            # ToolDefinition, ToolExecutor, built-in tools
├── providers/        # LLM providers (OpenAI, Anthropic, Azure, etc.)
├── cost/             # CostTracker, BudgetManager, PeriodBudgetManager
├── quality/          # QualityManager, CostEstimator
├── events/           # StreamEvent, Worktrail events, transports
├── engine/           # ExecutionEngine (workflow runner)
├── runtime/          # RuntimeContext, StepContext
├── escalation/       # EscalationLevel, handlers
├── tenancy/          # Multi-tenant isolation
├── audit/            # Compliance audit trail
└── auth/             # OIDC, JWT, API key + RBAC
```

---

## Built-in Tools

| Tool | Description |
|------|-------------|
| `web_search` | Search the web for information |
| `http_fetch` | Fetch content from a URL |
| `json_extract` | Extract values from JSON via dot-notation path |
| `text_summarize` | Summarize text by extracting key sentences |
| `code_eval` | Safely evaluate Python expressions |

```python
from llmteam.tools.builtin import web_search, http_fetch, code_eval

team.add_agent({
    "type": "llm",
    "role": "researcher",
    "prompt": "Research: {query}",
    "tools": [web_search.tool_definition, http_fetch.tool_definition],
    "max_tool_rounds": 5,
})
```

---

## Key Principles

### Security

1. **Horizontal Isolation** — Agents never see each other's contexts
2. **Vertical Visibility** — Orchestrators see only their child agents
3. **Sealed Data** — Only the owner agent can access sealed fields
4. **Tenant Isolation** — Complete data separation between tenants

### Orchestration

1. **Typed Agents** — Only LLM, RAG, KAG (no custom Agent classes)
2. **Separate Orchestrator** — TeamOrchestrator is NOT an agent
3. **Config-Driven** — Agents defined by config dicts, not subclasses
4. **Tool Loop** — Agents autonomously call tools in multi-round loops

### Cost & Quality

1. **Quality Slider** — Single 0-100 parameter controls quality/cost tradeoff
2. **Budget Enforcement** — Per-run and per-period cost limits
3. **Cost Estimation** — Pre-run cost prediction
4. **Streaming** — Real-time events for progress monitoring

---

## Links

- [PyPI Package](https://pypi.org/project/llmteam-ai/)
- [GitHub Repository](https://github.com/llmteamai-rgb/LLMTeam)

## License

Apache 2.0 License
