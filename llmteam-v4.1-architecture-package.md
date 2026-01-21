# LLMTeam v4.1 — Архитектурный пакет изменений

**Дата:** 2026-01-20  
**Статус:** Утверждён для разработки  
**Автор:** @Architect

---

## Цель изменений

Реорганизация архитектуры TeamOrchestrator для обеспечения:
1. Обязательного присутствия оркестратора в каждой команде
2. Автоматического сбора отчётов от всех агентов
3. Независимости от способа управления flow (Canvas или Orchestrator)
4. Чёткого разделения между рабочими агентами и супервизором

---

## 1. Вынос TeamOrchestrator из списка агентов

### Проблема (текущее состояние)

```
LLMTeam._agents = {
    "researcher": LLMAgent,
    "writer": LLMAgent,
    "editor": LLMAgent,
    "_orchestrator": LLMAgent,  ← В общем списке
}

Canvas получает segment из 4 шагов, включая _orchestrator
```

### Решение (целевое состояние)

```
LLMTeam._orchestrator = TeamOrchestrator  ← Отдельное поле
LLMTeam._agents = {
    "researcher": LLMAgent,
    "writer": LLMAgent,
    "editor": LLMAgent,
}

Canvas получает segment из 3 шагов (только рабочие агенты)
```

### Изменения в LLMTeam

| Поле/Метод | Было | Стало |
|------------|------|-------|
| `_agents` | Все агенты включая _orchestrator | Только рабочие агенты |
| `_orchestrator` | Нет | NEW: отдельное поле TeamOrchestrator |
| `add_agent()` | Позволял role="_orchestrator" | Запрещает role начинающийся с "_" |
| `list_agents()` | Все агенты | Только рабочие агенты |
| `get_agent(id)` | Искал везде | Ищет только среди рабочих |
| `get_orchestrator()` | Нет | NEW: возвращает TeamOrchestrator |

### Изменения в Converters

| Функция | Изменение |
|---------|-----------|
| `build_segment()` | Принимает только `_agents`, игнорирует `_orchestrator` |
| `agents_to_steps()` | Конвертирует только рабочих агентов |

---

## 2. TeamOrchestrator — отдельный класс

### Принцип

TeamOrchestrator НЕ наследуется от BaseAgent. Это отдельная сущность-супервизор.

```
BaseAgent (abstract)
    ├── LLMAgent      — рабочий агент
    ├── RAGAgent      — рабочий агент
    └── KAGAgent      — рабочий агент

TeamOrchestrator      — ОТДЕЛЬНО, не агент
    └── содержит LLMAgent внутри для LLM-вызовов
```

### Структура класса TeamOrchestrator

```
TeamOrchestrator
│
├── Поля:
│   ├── _team: LLMTeam              # Ссылка на команду
│   ├── _config: OrchestratorConfig # Конфигурация
│   ├── _llm: LLMAgent              # Внутренний LLM для генерации
│   ├── _reports: List[AgentReport] # Собранные отчёты агентов
│   ├── _current_run_id: str        # ID текущего выполнения
│   └── _scope: OrchestratorScope   # TEAM или GROUP
│
├── Lifecycle:
│   ├── start_run(run_id)           # Начать отслеживание
│   └── end_run()                   # Завершить отслеживание
│
├── Reporting:
│   ├── receive_report(AgentReport) # Получить отчёт от агента
│   ├── generate_report() -> str    # Сгенерировать итоговый отчёт
│   └── get_summary() -> Dict       # Получить структурированную сводку
│
├── Recovery (если mode включает RECOVERY):
│   └── decide_recovery(error) -> RecoveryAction
│
└── Routing (если mode включает ROUTER):
    └── decide_next_agent(state) -> str
```

---

## 3. TeamOrchestrator — обязательный компонент

### Принцип

Оркестратор присутствует ВСЕГДА в каждой команде, независимо от настроек.

### Режимы работы (OrchestratorMode)

| Режим | Описание | По умолчанию |
|-------|----------|--------------|
| SUPERVISOR | Наблюдение, получение отчётов от агентов | ✅ Включён |
| REPORTER | Генерация отчётов о выполнении | ✅ Включён |
| RECOVERY | Принятие решений при ошибках агентов | ⚪ Выключен |
| ROUTER | Управление flow (выбор следующего агента) | ⚪ Выключен |

### Пресеты режимов

```
PASSIVE = SUPERVISOR | REPORTER           # По умолчанию
ACTIVE = SUPERVISOR | REPORTER | ROUTER   # Полное управление
FULL = SUPERVISOR | REPORTER | ROUTER | RECOVERY
```

### Конфигурация (OrchestratorConfig)

```
OrchestratorConfig:
    mode: OrchestratorMode = PASSIVE
    model: str = "gpt-4o-mini"
    
    # Recovery settings
    auto_retry: bool = True
    max_retries: int = 2
    escalate_on_failure: bool = True
    
    # Report settings
    generate_report: bool = True
    report_format: str = "markdown"  # "markdown" | "json" | "text"
    include_agent_outputs: bool = True
    include_timing: bool = True
```

---

## 4. Agent Reporting — встроенный механизм

### Принцип

Каждый агент автоматически отчитывается оркестратору после выполнения, независимо от того, кто управляет flow.

### Изменения в BaseAgent

Метод `process()` оборачивает выполнение:

```
async def process(input_data, context) -> AgentResult:
    started_at = now()
    
    try:
        result = await self._execute(input_data, context)  # Абстрактный
    except Exception as error:
        result = AgentResult(success=False, error=str(error))
        await self._report(input_data, result, started_at, error)
        raise
    
    await self._report(input_data, result, started_at, None)
    return result
```

### AgentReport — модель отчёта

```
AgentReport:
    # Идентификация
    agent_id: str
    agent_role: str
    agent_type: str          # "llm", "rag", "kag"
    
    # Тайминг
    started_at: datetime
    completed_at: datetime
    duration_ms: int
    
    # Результат (компактно)
    input_summary: str       # До 200 символов
    output_summary: str      # До 200 символов
    output_key: str
    
    # Статус
    success: bool
    error: str | None
    error_type: str | None
    
    # Метрики
    tokens_used: int
    model: str | None
```

### Поток данных

```
Agent.process()
    │
    ├── 1. Выполнить _execute()
    ├── 2. Сформировать AgentReport
    └── 3. Вызвать team.get_orchestrator().receive_report(report)
                │
                ▼
        TeamOrchestrator._reports.append(report)
```

---

## 5. Разделение ответственности

### Canvas vs Orchestrator

| Аспект | Canvas (SegmentRunner) | TeamOrchestrator |
|--------|------------------------|------------------|
| Управляет flow | ✅ Да (если не ROUTER mode) | Только в ROUTER mode |
| Видит агентов | Как steps в segment | Через отчёты AgentReport |
| Получает результаты | Напрямую от handlers | Через reporting механизм |
| Генерирует отчёт | ❌ Нет | ✅ Да |
| Обрабатывает ошибки | Базово (retry/fail) | Умно (LLM решает) |

### Поток выполнения (PASSIVE mode)

```
LLMTeam.run(input_data):
    1. orchestrator.start_run(run_id)
    2. segment = build_segment(self._agents)    # Без orchestrator
    3. result = canvas.run(segment, input_data) # Агенты отчитываются
    4. report = orchestrator.generate_report()
    5. return RunResult(..., report=report)
```

### Поток выполнения (ROUTER mode)

```
LLMTeam.run(input_data):
    1. orchestrator.start_run(run_id)
    2. while not done:
           next_agent = orchestrator.decide_next_agent(state)
           result = agents[next_agent].process()  # Отчитывается автоматически
    3. report = orchestrator.generate_report()
    4. return RunResult(..., report=report)
```

---

## 6. Group Orchestration — делегирование

### Принцип

При создании группы один из TeamOrchestrator получает дополнительный scope GROUP, но НЕ управляет агентами других команд напрямую.

### OrchestratorScope

```
OrchestratorScope:
    TEAM = "team"    # Управляет агентами своей команды
    GROUP = "group"  # + координирует другие команды
```

### Иерархия при GROUP scope

```
Group Orchestrator (Team A's)
│
├── Уровень GROUP:
│   ├── Видит: Team B status, Team C status (саммари)
│   ├── НЕ видит: Детали агентов Team B, Team C
│   └── Команды: "Team B, выполни X" (делегация)
│
└── Уровень TEAM:
    └── Управляет агентами Team A как обычно
```

### Размер контекста

| Уровень | Контекст | Размер |
|---------|----------|--------|
| TEAM | Детали своих агентов | ~200-300 tokens |
| GROUP | + Саммари других команд | +300-500 tokens |
| **Итого** | | ~500-800 tokens |

### Промоция в GROUP scope

```
LLMTeam.create_group(group_id, teams):
    1. self._orchestrator.promote_to_group(group_id, teams)
    2. return LLMGroup(leader=self, teams=teams)
```

---

## 7. Предустановленные промпты

### Файл: agents/prompts.py

**ORCHESTRATOR_SYSTEM_PROMPT** — базовая роль супервизора
- Описание команды и агентов
- Обязанности по режиму
- Инструкции по формату ответов

**ROUTING_PROMPT** — решение о следующем агенте
- Текущее состояние выполнения
- Описания доступных агентов
- Формат ответа: JSON с next_agent, reason

**ERROR_RECOVERY_PROMPT** — решение при ошибке
- Информация об ошибке
- Варианты: RETRY, SKIP, FALLBACK, ESCALATE, ABORT
- Формат ответа: JSON с action, reason

**REPORT_PROMPT** — генерация отчёта
- Execution log
- Agent outputs
- Формат отчёта по настройке

**TEAM_SUMMARY_TEMPLATE** — компактное представление команды для GROUP уровня

---

## 8. API изменения

### LLMTeam constructor

```python
LLMTeam(
    team_id: str,
    agents: List[Dict] = None,
    flow: str = "sequential",
    model: str = "gpt-4o-mini",
    context_mode: ContextMode = ContextMode.SHARED,
    orchestrator: OrchestratorConfig = None,  # NEW
    timeout: int = None,
)
```

### RunResult расширение

```python
RunResult(
    # Existing
    success: bool
    status: RunStatus
    output: Dict[str, Any]
    final_output: Any
    agents_called: List[str]
    duration_ms: int
    error: str | None
    
    # NEW
    report: str              # Сгенерированный отчёт от оркестратора
    summary: Dict[str, Any]  # Структурированная сводка
)
```

### Обратная совместимость

| Было | Стало | Миграция |
|------|-------|----------|
| `orchestration=True` | `orchestrator=OrchestratorConfig(mode=ACTIVE)` | Автоматическая |
| `orchestration=False` | По умолчанию PASSIVE | Автоматическая |
| `flow="adaptive"` | `orchestrator=OrchestratorConfig(mode=ACTIVE)` | Автоматическая |

---

## 9. Файловая структура изменений

```
llmteam/
├── agents/
│   ├── base.py           # MODIFY: добавить _report() метод
│   ├── report.py         # NEW: AgentReport модель
│   ├── orchestrator.py   # NEW: TeamOrchestrator класс
│   ├── prompts.py        # NEW: предустановленные промпты
│   └── __init__.py       # MODIFY: экспорты
│
├── team/
│   ├── team.py           # MODIFY: _orchestrator отдельно от _agents
│   ├── group.py          # MODIFY: делегация через scope
│   ├── converters.py     # MODIFY: build_segment без orchestrator
│   └── result.py         # MODIFY: добавить report, summary
│
└── __init__.py           # MODIFY: экспорты
```

---

## 10. Новые модели данных

### agents/report.py

```
AgentReport (dataclass)
    - agent_id, agent_role, agent_type
    - started_at, completed_at, duration_ms
    - input_summary, output_summary, output_key
    - success, error, error_type
    - tokens_used, model
    - to_dict() -> Dict
    - to_log_line() -> str
```

### agents/orchestrator.py

```
OrchestratorMode (Flag)
    - SUPERVISOR
    - REPORTER
    - ROUTER
    - RECOVERY
    - PASSIVE = SUPERVISOR | REPORTER
    - ACTIVE = PASSIVE | ROUTER
    - FULL = ACTIVE | RECOVERY

OrchestratorScope (Enum)
    - TEAM
    - GROUP

OrchestratorConfig (dataclass)
    - mode, model
    - auto_retry, max_retries, escalate_on_failure
    - generate_report, report_format
    - include_agent_outputs, include_timing

TeamOrchestrator (class)
    - Методы lifecycle, reporting, recovery, routing
```

---

## 11. Итоговая диаграмма архитектуры

```
┌─────────────────────────────────────────────────────────────────┐
│                          LLMTeam                                │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              TeamOrchestrator (ОТДЕЛЬНО)                 │  │
│  │                                                          │  │
│  │   • Не агент. Не в _agents. Не идёт в Canvas.           │  │
│  │   • Супервизор над агентами.                             │  │
│  │   • Содержит LLMAgent внутри для LLM-вызовов.           │  │
│  │                                                          │  │
│  │   Режимы: [SUPERVISOR ✅] [REPORTER ✅] [RECOVERY] [ROUTER] │  │
│  │                                                          │  │
│  │                     ▲ receive_report()                   │  │
│  └─────────────────────┼────────────────────────────────────┘  │
│                        │                                        │
│  ┌─────────────────────┼────────────────────────────────────┐  │
│  │                     │    _agents (только рабочие)        │  │
│  │                     │                                     │  │
│  │  ┌─────────┐        │   ┌─────────┐   ┌─────────┐       │  │
│  │  │ Agent 1 │────────┘   │ Agent 2 │   │ Agent 3 │       │  │
│  │  │_report()│            │_report()│   │_report()│        │  │
│  │  └────┬────┘            └────┬────┘   └────┬────┘       │  │
│  │       │                      │             │             │  │
│  └───────┼──────────────────────┼─────────────┼─────────────┘  │
│          │                      │             │                 │
│          ▼                      ▼             ▼                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  Canvas (SegmentRunner)                   │  │
│  │                                                          │  │
│  │   Получает segment из 3 агентов (без orchestrator)      │  │
│  │   Управляет flow: Agent 1 → Agent 2 → Agent 3           │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 12. Приоритет реализации

### P0 — Блокеры (неделя 1)

1. **AgentReport модель** — `agents/report.py`
2. **_report() метод в BaseAgent** — `agents/base.py`
3. **TeamOrchestrator класс** (SUPERVISOR + REPORTER) — `agents/orchestrator.py`
4. **Вынос orchestrator из _agents** — `team/team.py`
5. **Изменения в converters** — `team/converters.py`

### P1 — Важно (неделя 2)

6. **OrchestratorConfig и режимы** — `agents/orchestrator.py`
7. **Предустановленные промпты** — `agents/prompts.py`
8. **RunResult.report, RunResult.summary** — `team/result.py`
9. **Обратная совместимость** (orchestration=True → mode=ACTIVE)

### P2 — Улучшения (неделя 3)

10. **RECOVERY режим** — decide_recovery()
11. **ROUTER режим** — decide_next_agent()
12. **Интеграция с Canvas** для ROUTER mode

### P3 — Позже

13. **GROUP scope** — promote_to_group()
14. **Мультикомандная координация** — group.py изменения

---

## 13. Критерии приёмки

### Функциональные

- [ ] Orchestrator создаётся автоматически при создании LLMTeam
- [ ] Orchestrator НЕ появляется в list_agents()
- [ ] Orchestrator НЕ попадает в segment для Canvas
- [ ] Каждый агент отправляет AgentReport после process()
- [ ] generate_report() возвращает осмысленный отчёт
- [ ] RunResult содержит поля report и summary

### Обратная совместимость

- [ ] `orchestration=True` работает как раньше (включает ROUTER)
- [ ] `flow="adaptive"` работает как раньше
- [ ] Существующий код без orchestration продолжает работать

### Тесты

- [ ] Unit: AgentReport создание и сериализация
- [ ] Unit: TeamOrchestrator.receive_report()
- [ ] Unit: TeamOrchestrator.generate_report()
- [ ] Integration: LLMTeam.run() возвращает report
- [ ] Integration: Canvas не видит orchestrator

---

**Документ готов к передаче в разработку.**
