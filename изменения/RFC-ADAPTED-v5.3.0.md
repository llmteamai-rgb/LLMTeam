# RFC Adaptation Analysis for llmteam v5.3.0

**Дата анализа:** 2026-01-24
**Версия библиотеки:** 5.3.0
**Автор:** @Architect

---

## Резюме

Все 5 RFC из roadmap v5.2.0 реализованы в v5.3.0:

| RFC | Статус | Модуль | Тесты |
|-----|--------|--------|-------|
| **RFC-008** (Group Architecture) | ✅ **DONE** (v5.2.0) | `orchestration/` | Есть |
| **RFC-010** (Cost Tracking) | ✅ **DONE** (v5.3.0) | `cost/` | 37 |
| **RFC-011** (Streaming Output) | ✅ **DONE** (v5.3.0) | `events/streaming.py` | 21 |
| **RFC-012** (Retry & Circuit Breaker) | ✅ **DONE** (v5.3.0) | `agents/retry.py` | 39 |
| **RFC-013** (Tool/Function Calling) | ✅ **DONE** (v5.3.0) | `tools/` | 60 |
| **RFC-014** (Enhanced Configurator) | ✅ **DONE** (v5.3.0) | `team/lifecycle.py` | 35 |

**Всего новых тестов:** 192 (все проходят)

---

## 1. RFC-012: Retry Policies & Circuit Breaker

### Статус: ✅ ПОЛНОСТЬЮ РЕАЛИЗОВАН

**Модуль:** `llmteam/agents/retry.py`

```python
from llmteam import RetryPolicy, CircuitBreakerPolicy, RetryMetrics, AgentRetryExecutor

# Per-agent retry policy
policy = RetryPolicy(
    max_retries=3,
    backoff="exponential",  # "exponential", "linear", "constant"
    base_delay=1.0,
    max_delay=30.0,
    multiplier=2.0,
    jitter=0.1,
    retryable_exceptions=(Exception,),
    on_retry=lambda attempt, error, delay: print(f"Retry {attempt}: {error}"),
)

# Per-agent circuit breaker
cb = CircuitBreakerPolicy(
    failure_threshold=5,
    recovery_timeout_seconds=30.0,
    success_threshold=2,
    failure_window_seconds=60.0,
    on_state_change=lambda old, new: print(f"CB: {old} → {new}"),
)

# Usage in team
team = LLMTeam(team_id="my-team")
team.add_agent({
    "type": "llm",
    "role": "analyst",
    "prompt": "...",
    "retry_policy": {"max_retries": 5, "backoff": "exponential"},
    "circuit_breaker": {"failure_threshold": 3, "recovery_timeout_seconds": 60},
})
```

**Интеграция:**
- `AgentConfig.retry_policy` / `AgentConfig.circuit_breaker` — per-agent policies
- `BaseAgent._retry_executor` — автоматически оборачивает `_execute()`
- `RetryMetrics` — прикрепляется к `AgentResult.context_payload["retry_metrics"]`
- При открытии CircuitBreaker — `result.should_escalate = True`

---

## 2. RFC-010: Cost Tracking & Budget Management

### Статус: ✅ ПОЛНОСТЬЮ РЕАЛИЗОВАН

**Модуль:** `llmteam/cost/` (pricing.py, tracker.py, budget.py)

```python
from llmteam import (
    ModelPricing, PricingRegistry, TokenUsage, RunCost,
    CostTracker, Budget, BudgetPeriod, BudgetStatus,
    BudgetManager, BudgetExceededError,
)

# Pricing registry with built-in models
registry = PricingRegistry()
cost = registry.calculate_cost("gpt-4o-mini", input_tokens=1000, output_tokens=500)

# Register custom model pricing
registry.register("my-model", ModelPricing(input_per_1k=0.005, output_per_1k=0.01))

# Cost tracker per run
tracker = CostTracker()
tracker.start_run("run-1", "team-1")
usage = tracker.record_usage("gpt-4o-mini", input_tokens=500, output_tokens=200)
run_cost = tracker.end_run()

# Budget management
manager = BudgetManager(Budget(max_cost=5.0, alert_threshold=0.8, hard_limit=True))
manager.on_alert(lambda cost, max_c: print(f"Alert: ${cost:.2f}/{max_c:.2f}"))
status = manager.check(current_cost=4.5)  # BudgetStatus.ALERT

# LLMTeam integration
team = LLMTeam(team_id="test", max_cost_per_run=10.0)
assert team.cost_tracker is not None
assert team.budget_manager is not None
result = await team.run({"query": "test"})
# result.summary["cost"] contains RunCost data
```

**Built-in модели:** gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo, claude-3-5-sonnet, claude-3-haiku, claude-3-opus, o1, o1-mini, o1-preview (с prefix matching)

---

## 3. RFC-011: Streaming Output

### Статус: ✅ ПОЛНОСТЬЮ РЕАЛИЗОВАН

**Модуль:** `llmteam/events/streaming.py` + `LLMTeam.stream()`

```python
from llmteam import LLMTeam, StreamEventType, StreamEvent

team = LLMTeam(team_id="test", orchestration=True)
team.add_agent({"type": "llm", "role": "writer", "prompt": "..."})

# Stream execution events
async for event in team.stream({"query": "Hello"}):
    if event.type == StreamEventType.RUN_STARTED:
        print(f"Run started: {event.data['agents']}")
    elif event.type == StreamEventType.AGENT_STARTED:
        print(f"Agent {event.agent_id} started")
    elif event.type == StreamEventType.AGENT_COMPLETED:
        print(f"Agent {event.agent_id} done: {event.data['output']}")
    elif event.type == StreamEventType.COST_UPDATE:
        print(f"Cost: ${event.data['current_cost']:.4f}")
    elif event.type == StreamEventType.RUN_COMPLETED:
        print(f"Done! Output: {event.data['output']}")
    elif event.type == StreamEventType.RUN_FAILED:
        print(f"Error: {event.data['error']}")

# SSE serialization
event = StreamEvent(type=StreamEventType.TOKEN, data={"token": "Hello"})
sse_string = event.to_sse()  # "event: token\ndata: {...}\n\n"
dict_data = event.to_dict()  # {"type": "token", "data": {...}, ...}
```

**StreamEventType values:**
- Run lifecycle: `RUN_STARTED`, `RUN_COMPLETED`, `RUN_FAILED`
- Agent lifecycle: `AGENT_STARTED`, `AGENT_COMPLETED`, `AGENT_FAILED`
- Token streaming: `TOKEN`, `CHUNK`
- Progress: `PROGRESS`
- Cost: `COST_UPDATE`

**Ограничения:** `stream()` работает только в ROUTER mode (orchestration=True).

---

## 4. RFC-013: Tool/Function Calling

### Статус: ✅ ПОЛНОСТЬЮ РЕАЛИЗОВАН

**Модуль:** `llmteam/tools/` (definition.py, decorator.py, executor.py)

```python
from llmteam import (
    ParamType, ToolParameter, ToolDefinition, ToolResult,
    tool, ToolExecutor,
)

# Decorator approach
@tool(description="Get weather for a city")
def get_weather(city: str, units: str = "celsius") -> str:
    """Get current weather."""
    return f"Weather in {city}: 22°{units[0].upper()}"

# Access the ToolDefinition
print(get_weather.tool_definition.name)  # "get_weather"
schema = get_weather.tool_definition.to_schema()  # OpenAI-compatible

# Manual definition
tool_def = ToolDefinition(
    name="search",
    description="Search the web",
    parameters=[
        ToolParameter(name="query", type=ParamType.STRING, required=True),
        ToolParameter(name="limit", type=ParamType.INTEGER, required=False, default=10),
    ],
    handler=lambda query, limit=10: f"Results for {query}",
)

# Executor
executor = ToolExecutor(tools=[get_weather.tool_definition, tool_def], timeout=30.0)
result = await executor.execute("get_weather", {"city": "London"})
assert result.success
assert result.output == "Weather in London: 22°C"

# Per-agent integration
team = LLMTeam(team_id="test")
team.add_agent({
    "type": "llm",
    "role": "assistant",
    "prompt": "...",
    "tools": [get_weather.tool_definition],  # or as dicts
})
agent = team.get_agent("assistant")
schemas = agent.tool_executor.get_schemas()  # OpenAI function calling format
```

**Поддерживаемые типы:** `str`, `int`, `float`, `bool`, `List[T]`, `Dict[str, T]`, `Optional[T]`
**Type coercion:** Автоматическая конвертация (e.g., `"42"` → `42` для INTEGER)
**Async support:** `ToolExecutor` поддерживает как sync, так и async handlers

---

## 5. RFC-014: Enhanced Configurator Mode (Opt-in Lifecycle)

### Статус: ✅ ПОЛНОСТЬЮ РЕАЛИЗОВАН

**Модуль:** `llmteam/team/lifecycle.py`

```python
from llmteam import (
    LLMTeam, TeamState, ProposalStatus,
    ConfigurationProposal, TeamLifecycle, LifecycleError,
)

# Opt-in lifecycle enforcement
team = LLMTeam(team_id="prod", enforce_lifecycle=True)
team.add_agent({"type": "llm", "role": "agent1", "prompt": "..."})

assert team.state == "unconfigured"

# Configure the team
team.mark_configuring()
assert team.state == "configuring"

# Add configuration proposals
proposal = ConfigurationProposal(
    proposal_id="p-1",
    changes={"model": "gpt-4o"},
    reason="Better accuracy needed",
)
team.lifecycle.add_proposal(proposal)
team.lifecycle.approve_all()

# Mark ready
team.mark_ready()
assert team.state == "ready"

# Now run is allowed
result = await team.run({"query": "test"})
assert team.state == "completed"  # or "failed"

# Re-run: COMPLETED → READY
team.mark_ready()
result2 = await team.run({"query": "test2"})

# Without lifecycle enforcement — works as before
team2 = LLMTeam(team_id="dev")
await team2.run({"query": "test"})  # No lifecycle checks
```

**State transitions:**
```
UNCONFIGURED → CONFIGURING → READY → RUNNING → COMPLETED
                                              → FAILED
                                    → PAUSED → RUNNING
COMPLETED → READY (re-run)
COMPLETED → CONFIGURING (re-configure)
FAILED → READY / CONFIGURING
```

**Opt-in design:** Lifecycle НЕ активируется по умолчанию. Только `enforce_lifecycle=True` включает enforcement.

---

## Архитектурные решения v5.3.0

### 1. Per-agent vs Team-level (RFC-012)
Retry и Circuit Breaker реализованы **per-agent** (через `AgentConfig`), а не на уровне team. Это позволяет:
- Разные retry policies для разных агентов
- Изоляция circuit breaker по агентам
- Более гранулярный контроль

### 2. Provider callback vs Agent result (RFC-010)
Стоимость рассчитывается из `AgentResult.tokens_used` с эвристикой 60/40 (input/output split). При интеграции реальных провайдеров — можно переключить на точные данные.

### 3. Async generator vs Queue (RFC-011)
`team.stream()` реализован как **async generator** (yield), а не через asyncio.Queue. Это проще, не требует background tasks, и работает в рамках существующего execution loop.

### 4. Basic types only (RFC-013)
Типы ограничены базовыми: `str`, `int`, `float`, `bool`, `List`, `Dict`. Без nested objects, unions, или custom classes. Это достаточно для большинства use cases и упрощает validation.

### 5. Opt-in lifecycle (RFC-014)
Lifecycle enforcement по умолчанию **выключен**. Это сохраняет backward compatibility — существующий код не ломается. Включается явно через `enforce_lifecycle=True`.

---

## Новые модули v5.3.0

| Модуль | Файлы | Назначение |
|--------|-------|------------|
| `agents/retry.py` | 1 файл | RetryPolicy, CircuitBreakerPolicy, AgentRetryExecutor |
| `cost/` | 4 файла | ModelPricing, CostTracker, BudgetManager |
| `events/streaming.py` | 1 файл | StreamEventType, StreamEvent |
| `tools/` | 4 файла | ToolDefinition, @tool, ToolExecutor |
| `team/lifecycle.py` | 1 файл | TeamState, TeamLifecycle, ConfigurationProposal |

**Всего:** 11 новых файлов, ~1500 строк кода, 192 теста

---

## Что дальше (v5.4.0+)

Все запланированные RFC реализованы. Возможные направления развития:

| Направление | Описание | Приоритет |
|-------------|----------|-----------|
| Tool execution loop в LLMAgent | Автоматический цикл tool_call → execute → respond | P1 |
| Streaming token-by-token | Интеграция с провайдерами для посимвольного стриминга | P1 |
| Built-in tools library | Набор готовых tools (web_search, code_exec, file_read) | P2 |
| Budget per-period | Бюджеты за час/день/месяц (не только per-run) | P2 |
| Configuration chat mode | Интерактивный чат с CONFIGURATOR через proposals | P3 |
| Retry-After from providers | Учёт Retry-After headers от LLM провайдеров | P3 |

---

## Definition of Done — Все ✅

### RFC-012 (Retry & Circuit Breaker)
- [x] RetryPolicy dataclass с backoff strategies
- [x] CircuitBreakerPolicy dataclass
- [x] AgentRetryExecutor wrapping _execute()
- [x] Integration в BaseAgent.__init__
- [x] Dict-to-dataclass conversion в AgentFactory
- [x] RetryMetrics в AgentResult
- [x] CircuitBreakerOpen → should_escalate
- [x] Tests: 39 passed

### RFC-010 (Cost Tracking)
- [x] PricingRegistry с 10+ моделями и prefix matching
- [x] CostTracker с start_run/record_usage/end_run
- [x] TokenUsage и RunCost dataclasses
- [x] BudgetManager с check/check_or_raise/on_alert
- [x] BudgetExceededError для hard limits
- [x] Integration в LLMTeam.run() (start/end cost tracking)
- [x] Budget enforcement в ROUTER mode
- [x] Tests: 37 passed

### RFC-011 (Streaming)
- [x] StreamEventType enum (10 event types)
- [x] StreamEvent с to_dict()/to_sse()
- [x] LLMTeam.stream() async generator
- [x] RUN_STARTED/COMPLETED/FAILED events
- [x] AGENT_STARTED/COMPLETED/FAILED events
- [x] COST_UPDATE event integration
- [x] Exception handling с RUN_FAILED
- [x] Tests: 21 passed

### RFC-013 (Tool Calling)
- [x] ParamType enum (6 types)
- [x] ToolParameter с validate() и coercion
- [x] ToolDefinition с to_schema() (OpenAI format)
- [x] @tool decorator с type inference
- [x] ToolExecutor с execute(), timeout, history
- [x] ToolResult dataclass
- [x] AgentConfig.tools field
- [x] Per-agent ToolExecutor в BaseAgent
- [x] Dict-to-ToolDefinition в AgentFactory
- [x] Tests: 60 passed

### RFC-014 (Enhanced Configurator)
- [x] TeamState enum (7 states)
- [x] VALID_TRANSITIONS map
- [x] TeamLifecycle state machine
- [x] ConfigurationProposal с approve/reject/apply
- [x] ProposalStatus enum
- [x] LifecycleError exception
- [x] LLMTeam(enforce_lifecycle=True)
- [x] mark_configuring() / mark_ready()
- [x] Lifecycle enforcement в run()
- [x] State transitions: RUNNING → COMPLETED/FAILED
- [x] Tests: 35 passed
