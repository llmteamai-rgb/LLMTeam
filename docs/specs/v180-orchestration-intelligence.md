# 🧠 Пакет изменений v1.8.0: Orchestration Intelligence

## 🎯 Цель релиза

Умная оркестрация с Process Mining и параллельным выполнением:
- Иерархический контекст между агентами
- Роли оркестраторов с Process Mining
- Параллельное выполнение с лицензированием

**Зависимость:** Требует v1.7.0 (Security Foundation)

---

## 📋 Состав пакета

| # | RFC | Файл | Effort |
|---|-----|------|--------|
| 1 | Hierarchical Context | `rfc-hierarchical-context-visibility.md` | 1.5 нед |
| 2 | Pipeline Orchestrator Roles | `rfc-pipeline-orchestrator-roles.md` | 2 нед |
| 3 | Group Orchestrator Roles | `rfc-group-orchestrator-roles.md` | 1.5 нед |
| 4 | Parallel Execution | `rfc-parallel-execution.md` | 2 нед |
| 4.1 | Licensing | `rfc-licensing-concurrency.md` | 0.5 нед |

**Общий effort: 7.5 недель**

---

## 🔗 Зависимости

```
v1.7.0 (Security Foundation)
         │
         ▼
┌─────────────────────┐
│ Hierarchical Context│
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌─────────┐
│Pipeline │ │ Group   │
│Orch     │ │ Orch    │
│Roles    │ │ Roles   │
└─────────┘ └────┬────┘
                 │
┌────────────────┴────────────────┐
│                                 │
▼                                 ▼
┌─────────────────┐    ┌─────────────────┐
│ Parallel        │───►│ Licensing       │
│ Execution       │    │                 │
└─────────────────┘    └─────────────────┘
```

---

## 📑 RFC #1: Hierarchical Context Visibility

### Назначение

Иерархическая модель передачи контекста между уровнями оркестрации.

### Модель иерархии

```
┌─────────────────────────────────────────────────────────┐
│                   GroupOrchestrator                     │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Pipeline A  │  │ Pipeline B  │  │ Pipeline C  │     │
│  │             │  │             │  │             │     │
│  │ ┌─┐ ┌─┐ ┌─┐│  │ ┌─┐ ┌─┐    │  │ ┌─┐ ┌─┐ ┌─┐│     │
│  │ │1│ │2│ │3││  │ │4│ │5│    │  │ │6│ │7│ │8││     │
│  │ └─┘ └─┘ └─┘│  │ └─┘ └─┘    │  │ └─┘ └─┘ └─┘│     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘

Правила видимости:
• Agent 1 видит: только себя
• Agent 2 видит: только себя (НЕ видит Agent 1)
• Pipeline A Orch видит: Agent 1, 2, 3
• Pipeline B Orch видит: Agent 4, 5 (НЕ видит 1, 2, 3)
• Group Orch видит: Pipeline A, B, C контексты
```

### Модель данных

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ContextScope(Enum):
    """Область видимости контекста."""
    AGENT = "agent"
    PIPELINE = "pipeline"
    GROUP = "group"


@dataclass
class HierarchicalContext:
    """
    Контекст с поддержкой иерархии.
    
    Интегрируется с SecureAgentContext из v1.7.0.
    """
    
    scope: ContextScope
    owner_id: str
    parent_id: Optional[str] = None
    
    # Данные контекста
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Дочерние контексты (для pipeline/group)
    children: Dict[str, "HierarchicalContext"] = field(default_factory=dict)
    
    # Метаданные для Process Mining
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
    def get_child_context(self, child_id: str) -> Optional["HierarchicalContext"]:
        """Получить контекст дочернего элемента."""
        return self.children.get(child_id)
    
    def set_child_context(self, child_id: str, context: "HierarchicalContext") -> None:
        """Установить контекст дочернего элемента."""
        context.parent_id = self.owner_id
        self.children[child_id] = context
        self.updated_at = datetime.now()
    
    def get_visible_children(self, viewer_id: str, viewer_role: str) -> Dict[str, dict]:
        """
        Получить видимые контексты дочерних элементов.
        
        Использует ContextAccessPolicy из v1.7.0.
        """
        result = {}
        for child_id, child_ctx in self.children.items():
            # Проверяем политику доступа
            if hasattr(child_ctx, 'access_policy'):
                allowed, _ = child_ctx.access_policy.can_access(viewer_id, viewer_role)
                if not allowed:
                    result[child_id] = {"access": "denied"}
                    continue
            
            result[child_id] = child_ctx.get_summary()
        
        return result
    
    def get_summary(self) -> dict:
        """Получить сводку контекста (без детальных данных)."""
        return {
            "scope": self.scope.value,
            "owner_id": self.owner_id,
            "children_count": len(self.children),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass 
class ContextPropagationConfig:
    """Конфигурация передачи контекста."""
    
    # Что передавать вверх (agent → pipeline → group)
    propagate_up: List[str] = field(default_factory=lambda: [
        "status", "confidence", "error_count"
    ])
    
    # Что передавать вниз (group → pipeline → agent)
    propagate_down: List[str] = field(default_factory=lambda: [
        "global_config", "shared_state"
    ])
    
    # Агрегация при передаче вверх
    aggregation_rules: Dict[str, str] = field(default_factory=lambda: {
        "confidence": "avg",
        "error_count": "sum",
        "status": "worst",  # idle < running < error < failed
    })


class ContextManager:
    """Управление иерархическим контекстом."""
    
    def __init__(self, propagation_config: ContextPropagationConfig = None):
        self.config = propagation_config or ContextPropagationConfig()
        self._contexts: Dict[str, HierarchicalContext] = {}
    
    def create_context(
        self, 
        scope: ContextScope, 
        owner_id: str,
        parent_id: str = None,
    ) -> HierarchicalContext:
        """Создать контекст."""
        ctx = HierarchicalContext(
            scope=scope,
            owner_id=owner_id,
            parent_id=parent_id,
        )
        self._contexts[owner_id] = ctx
        
        # Привязать к родителю
        if parent_id and parent_id in self._contexts:
            self._contexts[parent_id].set_child_context(owner_id, ctx)
        
        return ctx
    
    def propagate_up(self, child_id: str) -> None:
        """Передать данные вверх по иерархии."""
        child = self._contexts.get(child_id)
        if not child or not child.parent_id:
            return
        
        parent = self._contexts.get(child.parent_id)
        if not parent:
            return
        
        for field in self.config.propagate_up:
            if field in child.data:
                self._aggregate_to_parent(parent, field, child.data[field])
    
    def propagate_down(self, parent_id: str) -> None:
        """Передать данные вниз по иерархии."""
        parent = self._contexts.get(parent_id)
        if not parent:
            return
        
        for child in parent.children.values():
            for field in self.config.propagate_down:
                if field in parent.data:
                    child.data[field] = parent.data[field]
    
    def _aggregate_to_parent(self, parent: HierarchicalContext, field: str, value: Any) -> None:
        """Агрегировать значение в родительском контексте."""
        rule = self.config.aggregation_rules.get(field, "last")
        
        if rule == "sum":
            parent.data[field] = parent.data.get(field, 0) + value
        elif rule == "avg":
            # Simplified: just store last for now
            parent.data[field] = value
        elif rule == "worst":
            # Status ordering
            status_order = {"idle": 0, "running": 1, "error": 2, "failed": 3}
            current = parent.data.get(field, "idle")
            if status_order.get(value, 0) > status_order.get(current, 0):
                parent.data[field] = value
        else:
            parent.data[field] = value
```

### Использование

```python
from llmteam.context import ContextManager, ContextScope

ctx_manager = ContextManager()

# Создаём иерархию
group_ctx = ctx_manager.create_context(ContextScope.GROUP, "group_1")
pipeline_ctx = ctx_manager.create_context(ContextScope.PIPELINE, "pipeline_a", parent_id="group_1")
agent_ctx = ctx_manager.create_context(ContextScope.AGENT, "agent_1", parent_id="pipeline_a")

# Агент обновляет свой контекст
agent_ctx.data["confidence"] = 0.95
agent_ctx.data["status"] = "completed"

# Данные передаются вверх
ctx_manager.propagate_up("agent_1")

# Pipeline видит агрегированные данные
print(pipeline_ctx.data)  # {"confidence": 0.95, "status": "completed"}
```

---

## 📑 RFC #2: Pipeline Orchestrator Roles

### Назначение

Роли для pipeline orchestrator: Orchestration + Process Mining.

### Модель данных

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum
from datetime import datetime, timedelta


class OrchestratorRole(Enum):
    """Роли оркестратора."""
    ORCHESTRATION = "orchestration"
    PROCESS_MINING = "process_mining"


# === ORCHESTRATION ROLE ===

@dataclass
class OrchestrationDecision:
    """Решение оркестратора."""
    
    decision_type: str  # "route", "retry", "escalate", "skip", "parallel"
    target_agents: List[str]
    reason: str
    confidence: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class OrchestrationContext:
    """Контекст для принятия решений."""
    
    current_step: str
    available_agents: List[str]
    agent_states: Dict[str, dict]
    execution_history: List[dict]
    global_state: dict
    
    # Метрики для решений
    step_duration: timedelta = timedelta()
    retry_count: int = 0
    error_rate: float = 0.0


class OrchestrationStrategy:
    """Стратегия оркестрации."""
    
    async def decide(self, context: OrchestrationContext) -> OrchestrationDecision:
        """Принять решение о следующем шаге."""
        raise NotImplementedError


class RuleBasedStrategy(OrchestrationStrategy):
    """Стратегия на основе правил."""
    
    def __init__(self):
        self.rules: List[Callable[[OrchestrationContext], Optional[OrchestrationDecision]]] = []
    
    def add_rule(self, rule: Callable) -> "RuleBasedStrategy":
        self.rules.append(rule)
        return self
    
    async def decide(self, context: OrchestrationContext) -> OrchestrationDecision:
        for rule in self.rules:
            decision = rule(context)
            if decision:
                return decision
        
        # Default: next agent in sequence
        return OrchestrationDecision(
            decision_type="route",
            target_agents=[context.available_agents[0]] if context.available_agents else [],
            reason="default_sequence",
            confidence=1.0,
        )


class LLMBasedStrategy(OrchestrationStrategy):
    """Стратегия на основе LLM."""
    
    def __init__(self, llm, prompt_template: str = None):
        self.llm = llm
        self.prompt_template = prompt_template or self._default_prompt()
    
    async def decide(self, context: OrchestrationContext) -> OrchestrationDecision:
        prompt = self.prompt_template.format(
            current_step=context.current_step,
            agents=context.available_agents,
            states=context.agent_states,
            history=context.execution_history[-5:],  # Last 5 steps
        )
        
        response = await self.llm.generate(prompt)
        return self._parse_response(response)
    
    def _default_prompt(self) -> str:
        return """
        You are a pipeline orchestrator. Based on the current state, decide the next action.
        
        Current step: {current_step}
        Available agents: {agents}
        Agent states: {states}
        Recent history: {history}
        
        Respond with JSON: {{"decision": "route|retry|escalate|skip", "targets": [...], "reason": "..."}}
        """


# === PROCESS MINING ROLE ===

@dataclass
class ProcessEvent:
    """Событие для Process Mining."""
    
    event_id: str
    timestamp: datetime
    activity: str
    resource: str  # agent name
    
    # Стандартные атрибуты XES
    case_id: str  # run_id
    lifecycle: str = "complete"  # start, complete, suspend, resume
    
    # Дополнительные атрибуты
    duration_ms: int = 0
    cost: float = 0.0
    attributes: Dict = field(default_factory=dict)


@dataclass
class ProcessMetrics:
    """Метрики процесса."""
    
    # Время
    avg_duration: timedelta
    min_duration: timedelta
    max_duration: timedelta
    
    # Throughput
    cases_per_hour: float
    completion_rate: float
    
    # Quality
    error_rate: float
    retry_rate: float
    
    # Bottlenecks
    bottleneck_activities: List[str]
    waiting_time_by_activity: Dict[str, timedelta]


@dataclass
class ProcessModel:
    """Модель процесса (discovered)."""
    
    activities: List[str]
    transitions: Dict[str, List[str]]  # activity -> possible next activities
    frequencies: Dict[str, int]  # transition frequencies
    
    # Conformance
    fitness: float = 1.0
    precision: float = 1.0


class ProcessMiningEngine:
    """Движок Process Mining."""
    
    def __init__(self):
        self._events: List[ProcessEvent] = []
        self._cases: Dict[str, List[ProcessEvent]] = {}
    
    def record_event(self, event: ProcessEvent) -> None:
        """Записать событие."""
        self._events.append(event)
        
        if event.case_id not in self._cases:
            self._cases[event.case_id] = []
        self._cases[event.case_id].append(event)
    
    def discover_model(self) -> ProcessModel:
        """Обнаружить модель процесса (Alpha Miner simplified)."""
        activities = set()
        transitions = {}
        frequencies = {}
        
        for case_events in self._cases.values():
            sorted_events = sorted(case_events, key=lambda e: e.timestamp)
            
            for i, event in enumerate(sorted_events):
                activities.add(event.activity)
                
                if i > 0:
                    prev = sorted_events[i-1].activity
                    curr = event.activity
                    
                    if prev not in transitions:
                        transitions[prev] = []
                    if curr not in transitions[prev]:
                        transitions[prev].append(curr)
                    
                    key = f"{prev}->{curr}"
                    frequencies[key] = frequencies.get(key, 0) + 1
        
        return ProcessModel(
            activities=list(activities),
            transitions=transitions,
            frequencies=frequencies,
        )
    
    def calculate_metrics(self) -> ProcessMetrics:
        """Рассчитать метрики процесса."""
        durations = []
        errors = 0
        retries = 0
        
        for case_events in self._cases.values():
            if len(case_events) >= 2:
                start = min(e.timestamp for e in case_events)
                end = max(e.timestamp for e in case_events)
                durations.append(end - start)
            
            for event in case_events:
                if "error" in event.activity.lower():
                    errors += 1
                if "retry" in event.activity.lower():
                    retries += 1
        
        total_cases = len(self._cases)
        
        return ProcessMetrics(
            avg_duration=sum(durations, timedelta()) / len(durations) if durations else timedelta(),
            min_duration=min(durations) if durations else timedelta(),
            max_duration=max(durations) if durations else timedelta(),
            cases_per_hour=total_cases / max(1, (datetime.now() - self._events[0].timestamp).total_seconds() / 3600) if self._events else 0,
            completion_rate=sum(1 for c in self._cases.values() if any(e.lifecycle == "complete" for e in c)) / max(1, total_cases),
            error_rate=errors / max(1, len(self._events)),
            retry_rate=retries / max(1, len(self._events)),
            bottleneck_activities=self._find_bottlenecks(),
            waiting_time_by_activity={},
        )
    
    def _find_bottlenecks(self) -> List[str]:
        """Найти bottleneck activities."""
        activity_durations = {}
        
        for event in self._events:
            if event.activity not in activity_durations:
                activity_durations[event.activity] = []
            activity_durations[event.activity].append(event.duration_ms)
        
        # Sort by average duration
        avg_durations = {
            a: sum(d) / len(d) 
            for a, d in activity_durations.items()
        }
        
        sorted_activities = sorted(avg_durations.items(), key=lambda x: x[1], reverse=True)
        return [a for a, _ in sorted_activities[:3]]
    
    def export_xes(self) -> str:
        """Экспорт в XES формат для ProM/Celonis."""
        # Simplified XES export
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append('<log>')
        
        for case_id, events in self._cases.items():
            lines.append(f'  <trace>')
            lines.append(f'    <string key="concept:name" value="{case_id}"/>')
            
            for event in sorted(events, key=lambda e: e.timestamp):
                lines.append(f'    <event>')
                lines.append(f'      <string key="concept:name" value="{event.activity}"/>')
                lines.append(f'      <string key="org:resource" value="{event.resource}"/>')
                lines.append(f'      <date key="time:timestamp" value="{event.timestamp.isoformat()}"/>')
                lines.append(f'      <string key="lifecycle:transition" value="{event.lifecycle}"/>')
                lines.append(f'    </event>')
            
            lines.append(f'  </trace>')
        
        lines.append('</log>')
        return '\n'.join(lines)


# === PIPELINE ORCHESTRATOR ===

class PipelineOrchestrator:
    """Pipeline Orchestrator с двумя ролями."""
    
    def __init__(
        self,
        pipeline_id: str,
        strategy: OrchestrationStrategy = None,
        enable_process_mining: bool = True,
    ):
        self.pipeline_id = pipeline_id
        self.strategy = strategy or RuleBasedStrategy()
        
        # Process Mining
        self.process_mining = ProcessMiningEngine() if enable_process_mining else None
        
        # State
        self._agents: Dict[str, Any] = {}
        self._execution_history: List[dict] = []
    
    def register_agent(self, name: str, agent: Any) -> None:
        """Зарегистрировать агента."""
        self._agents[name] = agent
    
    async def orchestrate(self, run_id: str, input_data: dict) -> dict:
        """Выполнить оркестрацию."""
        current_step = "start"
        state = input_data.copy()
        
        while current_step != "end":
            # Собираем контекст
            context = OrchestrationContext(
                current_step=current_step,
                available_agents=list(self._agents.keys()),
                agent_states={n: a.get_state() for n, a in self._agents.items()},
                execution_history=self._execution_history,
                global_state=state,
            )
            
            # Принимаем решение
            decision = await self.strategy.decide(context)
            
            # Записываем для Process Mining
            if self.process_mining:
                self.process_mining.record_event(ProcessEvent(
                    event_id=generate_uuid(),
                    timestamp=datetime.now(),
                    activity=f"decision:{decision.decision_type}",
                    resource="orchestrator",
                    case_id=run_id,
                    lifecycle="complete",
                ))
            
            # Выполняем решение
            if decision.decision_type == "route":
                for agent_name in decision.target_agents:
                    agent = self._agents[agent_name]
                    
                    # Start event
                    if self.process_mining:
                        start_time = datetime.now()
                        self.process_mining.record_event(ProcessEvent(
                            event_id=generate_uuid(),
                            timestamp=start_time,
                            activity=agent_name,
                            resource=agent_name,
                            case_id=run_id,
                            lifecycle="start",
                        ))
                    
                    # Execute
                    result = await agent.process(state)
                    state.update(result)
                    
                    # Complete event
                    if self.process_mining:
                        self.process_mining.record_event(ProcessEvent(
                            event_id=generate_uuid(),
                            timestamp=datetime.now(),
                            activity=agent_name,
                            resource=agent_name,
                            case_id=run_id,
                            lifecycle="complete",
                            duration_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                        ))
                    
                    current_step = agent_name
            
            elif decision.decision_type == "end" or not decision.target_agents:
                current_step = "end"
            
            # Save to history
            self._execution_history.append({
                "step": current_step,
                "decision": decision,
                "timestamp": datetime.now(),
            })
        
        return state
    
    def get_process_metrics(self) -> Optional[ProcessMetrics]:
        """Получить метрики процесса."""
        if self.process_mining:
            return self.process_mining.calculate_metrics()
        return None
    
    def export_process_model(self) -> Optional[str]:
        """Экспортировать модель процесса в XES."""
        if self.process_mining:
            return self.process_mining.export_xes()
        return None
```

### Использование

```python
from llmteam.roles import (
    PipelineOrchestrator, 
    RuleBasedStrategy, 
    LLMBasedStrategy,
)

# Rule-based orchestration
strategy = RuleBasedStrategy()
strategy.add_rule(lambda ctx: 
    OrchestrationDecision("retry", [ctx.current_step], "high_error_rate", 0.9)
    if ctx.error_rate > 0.5 else None
)

orchestrator = PipelineOrchestrator(
    pipeline_id="loan_approval",
    strategy=strategy,
    enable_process_mining=True,
)

# Execute
result = await orchestrator.orchestrate(run_id="run_123", input_data=data)

# Get insights
metrics = orchestrator.get_process_metrics()
print(f"Avg duration: {metrics.avg_duration}")
print(f"Bottlenecks: {metrics.bottleneck_activities}")

# Export for external tools
xes = orchestrator.export_process_model()
```

---

## 📑 RFC #3: Group Orchestrator Roles

### Назначение

Роли для group orchestrator: управление несколькими pipeline.

### Модель данных

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class GroupDecisionType(Enum):
    """Типы решений group orchestrator."""
    ROUTE_TO_PIPELINE = "route_to_pipeline"
    PARALLEL_PIPELINES = "parallel_pipelines"
    AGGREGATE_RESULTS = "aggregate_results"
    ESCALATE = "escalate"


@dataclass
class GroupOrchestrationDecision:
    """Решение group orchestrator."""
    
    decision_type: GroupDecisionType
    target_pipelines: List[str]
    aggregation_strategy: str = "merge"  # merge, vote, first, all
    reason: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class PipelineStatus:
    """Статус pipeline для group orchestrator."""
    
    pipeline_id: str
    status: str  # idle, running, completed, failed
    progress: float  # 0.0 - 1.0
    current_step: str
    error_count: int
    last_update: datetime


class GroupOrchestrationStrategy:
    """Стратегия оркестрации группы."""
    
    async def decide(
        self, 
        pipelines: Dict[str, PipelineStatus],
        input_data: dict,
    ) -> GroupOrchestrationDecision:
        raise NotImplementedError


class LoadBalancingStrategy(GroupOrchestrationStrategy):
    """Стратегия балансировки нагрузки."""
    
    async def decide(self, pipelines: Dict[str, PipelineStatus], input_data: dict):
        # Find least loaded pipeline
        idle_pipelines = [
            p for p in pipelines.values() 
            if p.status == "idle"
        ]
        
        if idle_pipelines:
            return GroupOrchestrationDecision(
                decision_type=GroupDecisionType.ROUTE_TO_PIPELINE,
                target_pipelines=[idle_pipelines[0].pipeline_id],
                reason="load_balancing",
            )
        
        # All busy - queue or reject
        return GroupOrchestrationDecision(
            decision_type=GroupDecisionType.ROUTE_TO_PIPELINE,
            target_pipelines=[min(pipelines.values(), key=lambda p: p.progress).pipeline_id],
            reason="least_progress",
        )


class ContentBasedRoutingStrategy(GroupOrchestrationStrategy):
    """Стратегия маршрутизации по содержимому."""
    
    def __init__(self, routing_rules: Dict[str, List[str]]):
        # keyword -> pipeline_ids
        self.routing_rules = routing_rules
    
    async def decide(self, pipelines: Dict[str, PipelineStatus], input_data: dict):
        input_text = str(input_data).lower()
        
        for keyword, pipeline_ids in self.routing_rules.items():
            if keyword in input_text:
                return GroupOrchestrationDecision(
                    decision_type=GroupDecisionType.ROUTE_TO_PIPELINE,
                    target_pipelines=pipeline_ids,
                    reason=f"keyword_match:{keyword}",
                )
        
        # Default: first available
        return GroupOrchestrationDecision(
            decision_type=GroupDecisionType.ROUTE_TO_PIPELINE,
            target_pipelines=[list(pipelines.keys())[0]],
            reason="default",
        )


class ParallelFanOutStrategy(GroupOrchestrationStrategy):
    """Стратегия параллельного выполнения."""
    
    def __init__(self, aggregation: str = "merge"):
        self.aggregation = aggregation
    
    async def decide(self, pipelines: Dict[str, PipelineStatus], input_data: dict):
        return GroupOrchestrationDecision(
            decision_type=GroupDecisionType.PARALLEL_PIPELINES,
            target_pipelines=list(pipelines.keys()),
            aggregation_strategy=self.aggregation,
            reason="fan_out",
        )


class GroupOrchestrator:
    """Group Orchestrator."""
    
    def __init__(
        self,
        group_id: str,
        strategy: GroupOrchestrationStrategy = None,
    ):
        self.group_id = group_id
        self.strategy = strategy or LoadBalancingStrategy()
        
        self._pipelines: Dict[str, PipelineOrchestrator] = {}
        self._statuses: Dict[str, PipelineStatus] = {}
    
    def register_pipeline(self, pipeline: PipelineOrchestrator) -> None:
        """Зарегистрировать pipeline."""
        self._pipelines[pipeline.pipeline_id] = pipeline
        self._statuses[pipeline.pipeline_id] = PipelineStatus(
            pipeline_id=pipeline.pipeline_id,
            status="idle",
            progress=0.0,
            current_step="",
            error_count=0,
            last_update=datetime.now(),
        )
    
    async def orchestrate(self, run_id: str, input_data: dict) -> dict:
        """Выполнить оркестрацию группы."""
        decision = await self.strategy.decide(self._statuses, input_data)
        
        if decision.decision_type == GroupDecisionType.ROUTE_TO_PIPELINE:
            # Single pipeline execution
            pipeline_id = decision.target_pipelines[0]
            pipeline = self._pipelines[pipeline_id]
            return await pipeline.orchestrate(run_id, input_data)
        
        elif decision.decision_type == GroupDecisionType.PARALLEL_PIPELINES:
            # Parallel execution
            tasks = []
            for pipeline_id in decision.target_pipelines:
                pipeline = self._pipelines[pipeline_id]
                tasks.append(pipeline.orchestrate(f"{run_id}_{pipeline_id}", input_data))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate
            return self._aggregate_results(
                results, 
                decision.aggregation_strategy,
            )
        
        return {}
    
    def _aggregate_results(self, results: List, strategy: str) -> dict:
        """Агрегировать результаты."""
        valid_results = [r for r in results if not isinstance(r, Exception)]
        
        if strategy == "first":
            return valid_results[0] if valid_results else {}
        
        elif strategy == "vote":
            # Majority vote on specific field
            # Simplified: return most common result
            return valid_results[0] if valid_results else {}
        
        elif strategy == "merge":
            merged = {}
            for r in valid_results:
                merged.update(r)
            return merged
        
        elif strategy == "all":
            return {"results": valid_results}
        
        return {}
    
    def get_pipeline_statuses(self) -> Dict[str, PipelineStatus]:
        """Получить статусы всех pipeline."""
        return self._statuses.copy()
    
    def get_group_metrics(self) -> dict:
        """Получить агрегированные метрики группы."""
        all_metrics = []
        for pipeline in self._pipelines.values():
            m = pipeline.get_process_metrics()
            if m:
                all_metrics.append(m)
        
        if not all_metrics:
            return {}
        
        return {
            "total_pipelines": len(self._pipelines),
            "avg_completion_rate": sum(m.completion_rate for m in all_metrics) / len(all_metrics),
            "avg_error_rate": sum(m.error_rate for m in all_metrics) / len(all_metrics),
            "bottlenecks": list(set(
                b for m in all_metrics for b in m.bottleneck_activities
            )),
        }
```

### Использование

```python
from llmteam.roles import GroupOrchestrator, ContentBasedRoutingStrategy

# Content-based routing
strategy = ContentBasedRoutingStrategy({
    "urgent": ["fast_pipeline"],
    "complex": ["detailed_pipeline"],
    "financial": ["finance_pipeline", "compliance_pipeline"],
})

group = GroupOrchestrator(group_id="main_group", strategy=strategy)
group.register_pipeline(fast_pipeline)
group.register_pipeline(detailed_pipeline)
group.register_pipeline(finance_pipeline)

# Execute
result = await group.orchestrate("run_1", {"query": "urgent financial report"})
```

---

## 📑 RFC #4: Parallel Execution

### Назначение

Параллельное выполнение агентов с контролем concurrency.

### Модель данных

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from enum import Enum
import asyncio


class ExecutionMode(Enum):
    """Режим выполнения."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"  # Выбирает автоматически


@dataclass
class ExecutorConfig:
    """Конфигурация executor."""
    
    mode: ExecutionMode = ExecutionMode.ADAPTIVE
    max_concurrent: int = 10
    queue_size: int = 100
    
    # Timeouts
    task_timeout: float = 300.0  # 5 min
    total_timeout: float = 3600.0  # 1 hour
    
    # Retry
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Backpressure
    enable_backpressure: bool = True
    backpressure_threshold: float = 0.8  # 80% queue full


@dataclass
class TaskResult:
    """Результат выполнения задачи."""
    
    task_id: str
    agent_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration_ms: int = 0
    retries: int = 0


@dataclass
class ExecutionStats:
    """Статистика выполнения."""
    
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    total_duration_ms: int = 0
    avg_duration_ms: float = 0.0
    
    current_queue_size: int = 0
    current_running: int = 0
    
    backpressure_events: int = 0


class PipelineExecutor:
    """Executor с параллельным выполнением."""
    
    def __init__(self, config: ExecutorConfig = None):
        self.config = config or ExecutorConfig()
        
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.queue_size)
        self._stats = ExecutionStats()
        self._running = False
    
    async def execute_parallel(
        self,
        agents: List[Any],
        input_data: dict,
    ) -> List[TaskResult]:
        """Выполнить агентов параллельно."""
        tasks = []
        
        for agent in agents:
            task = asyncio.create_task(
                self._execute_with_semaphore(agent, input_data)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [
            r if isinstance(r, TaskResult) else TaskResult(
                task_id="",
                agent_name="unknown",
                success=False,
                error=str(r),
            )
            for r in results
        ]
    
    async def _execute_with_semaphore(
        self, 
        agent: Any, 
        input_data: dict,
    ) -> TaskResult:
        """Выполнить с контролем concurrency."""
        async with self._semaphore:
            self._stats.current_running += 1
            start_time = asyncio.get_event_loop().time()
            
            try:
                result = await asyncio.wait_for(
                    agent.process(input_data),
                    timeout=self.config.task_timeout,
                )
                
                duration = int((asyncio.get_event_loop().time() - start_time) * 1000)
                self._stats.completed_tasks += 1
                
                return TaskResult(
                    task_id=generate_uuid(),
                    agent_name=agent.name,
                    success=True,
                    result=result,
                    duration_ms=duration,
                )
                
            except asyncio.TimeoutError:
                self._stats.failed_tasks += 1
                return TaskResult(
                    task_id=generate_uuid(),
                    agent_name=agent.name,
                    success=False,
                    error="timeout",
                )
                
            except Exception as e:
                self._stats.failed_tasks += 1
                return TaskResult(
                    task_id=generate_uuid(),
                    agent_name=agent.name,
                    success=False,
                    error=str(e),
                )
                
            finally:
                self._stats.current_running -= 1
                self._stats.total_tasks += 1
    
    def get_stats(self) -> ExecutionStats:
        """Получить статистику."""
        return self._stats
    
    def is_backpressure(self) -> bool:
        """Проверить backpressure."""
        if not self.config.enable_backpressure:
            return False
        
        queue_ratio = self._queue.qsize() / self.config.queue_size
        return queue_ratio >= self.config.backpressure_threshold
```

---

## 📑 RFC #4.1: Licensing

### Назначение

Ограничения по лицензии на concurrency и features.

### Модель данных

```python
from dataclasses import dataclass
from enum import Enum
from typing import Set


class LicenseTier(Enum):
    """Уровень лицензии."""
    COMMUNITY = "community"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class LicenseLimits:
    """Лимиты по лицензии."""
    
    max_concurrent_pipelines: int
    max_agents_per_pipeline: int
    max_parallel_agents: int
    
    features: Set[str]


LICENSE_LIMITS = {
    LicenseTier.COMMUNITY: LicenseLimits(
        max_concurrent_pipelines=1,
        max_agents_per_pipeline=5,
        max_parallel_agents=2,
        features={"basic_agents", "sequential_execution"},
    ),
    LicenseTier.PROFESSIONAL: LicenseLimits(
        max_concurrent_pipelines=5,
        max_agents_per_pipeline=20,
        max_parallel_agents=10,
        features={"basic_agents", "sequential_execution", "parallel_execution", 
                  "process_mining", "external_actions"},
    ),
    LicenseTier.ENTERPRISE: LicenseLimits(
        max_concurrent_pipelines=999999,
        max_agents_per_pipeline=999999,
        max_parallel_agents=999999,
        features={"*"},
    ),
}


class LicenseManager:
    """Управление лицензиями."""
    
    def __init__(self, tier: LicenseTier = LicenseTier.COMMUNITY):
        self.tier = tier
        self.limits = LICENSE_LIMITS[tier]
    
    def check_concurrent_limit(self, current: int) -> bool:
        return current < self.limits.max_concurrent_pipelines
    
    def check_agents_limit(self, count: int) -> bool:
        return count <= self.limits.max_agents_per_pipeline
    
    def check_parallel_limit(self, count: int) -> bool:
        return count <= self.limits.max_parallel_agents
    
    def check_feature(self, feature: str) -> bool:
        return "*" in self.limits.features or feature in self.limits.features
    
    def enforce(self, executor_config: ExecutorConfig) -> ExecutorConfig:
        """Применить лимиты к конфигурации."""
        return ExecutorConfig(
            mode=executor_config.mode,
            max_concurrent=min(
                executor_config.max_concurrent,
                self.limits.max_parallel_agents,
            ),
            queue_size=executor_config.queue_size,
            task_timeout=executor_config.task_timeout,
            total_timeout=executor_config.total_timeout,
        )
```

### Использование

```python
from llmteam.execution import PipelineExecutor, ExecutorConfig
from llmteam.licensing import LicenseManager, LicenseTier

license_mgr = LicenseManager(LicenseTier.PROFESSIONAL)

config = ExecutorConfig(max_concurrent=20)
config = license_mgr.enforce(config)  # Limited to 10 for Professional

executor = PipelineExecutor(config)
```

---

## 📅 План реализации

| Неделя | Задачи |
|--------|--------|
| 1 | Hierarchical Context: модели, ContextManager |
| 2 | Hierarchical Context: интеграция с v1.7.0 security |
| 3 | Pipeline Orchestrator: OrchestrationStrategy, базовые стратегии |
| 4 | Pipeline Orchestrator: ProcessMiningEngine |
| 5 | Group Orchestrator: стратегии, агрегация |
| 6 | Parallel Execution: PipelineExecutor |
| 7 | Licensing: LicenseManager, интеграция |
| +0.5 | Тестирование, документация |

**Итого: ~7.5 недель**

---

## 📁 Структура файлов

```
src/llmteam/
├── context/
│   ├── hierarchical.py        # HierarchicalContext, ContextManager
│   └── propagation.py         # ContextPropagationConfig
│
├── roles/
│   ├── __init__.py
│   ├── orchestration.py       # OrchestrationStrategy, OrchestrationContext
│   ├── process_mining.py      # ProcessMiningEngine, ProcessMetrics
│   ├── pipeline_orch.py       # PipelineOrchestrator
│   └── group_orch.py          # GroupOrchestrator
│
├── execution/
│   ├── __init__.py
│   ├── executor.py            # PipelineExecutor
│   ├── config.py              # ExecutorConfig
│   └── stats.py               # ExecutionStats
│
└── licensing/
    ├── __init__.py
    ├── models.py              # LicenseTier, LicenseLimits
    └── manager.py             # LicenseManager
```

---

## ✅ Критерии готовности

- [ ] Интеграция с v1.7.0 security (Context Access Policy)
- [ ] Все тесты проходят
- [ ] XES export совместим с ProM/Celonis
- [ ] Документация с примерами
- [ ] Performance benchmarks для parallel execution

---

## 🎯 Результат v1.8.0

```python
from llmteam import create_pipeline
from llmteam.roles import PipelineOrchestrator, GroupOrchestrator, LLMBasedStrategy
from llmteam.execution import PipelineExecutor, ExecutorConfig
from llmteam.licensing import LicenseManager, LicenseTier

# License
license_mgr = LicenseManager(LicenseTier.PROFESSIONAL)

# Executor с лицензией
executor = PipelineExecutor(
    license_mgr.enforce(ExecutorConfig(max_concurrent=20))
)

# Smart orchestration
orchestrator = PipelineOrchestrator(
    pipeline_id="smart_flow",
    strategy=LLMBasedStrategy(llm),
    enable_process_mining=True,
)

# Execute
result = await orchestrator.orchestrate("run_1", input_data)

# Analytics
metrics = orchestrator.get_process_metrics()
print(f"Bottlenecks: {metrics.bottleneck_activities}")

# Export for Celonis
xes = orchestrator.export_process_model()
```

---

**Версия: 1.8.0**
**Кодовое имя: Orchestration Intelligence**
**Зависимость: v1.7.0**
