# RFC-008: Group Architecture Unification — Единая система координации команд

**Дата:** 2026-01-21  
**Статус:** DRAFT  
**Автор:** @Architect + @KorpOS  
**Приоритет:** P1  
**Зависимости:** RFC-004 (GroupOrchestrator)

---

## TL;DR

| Аспект | Значение |
|--------|----------|
| **Проблема** | Два несвязанных подхода к группам: `LLMGroup` и `GroupOrchestrator` |
| **Решение** | Унификация под `GroupOrchestrator` с ролями и bi-directional связью |
| **Ключевое изменение** | TeamOrchestrator узнаёт о группе через `GroupContext` |
| **Breaking Changes** | Да (LLMGroup API изменится) |

---

## 1. Проблема

### 1.1 Текущее состояние (v5.0.0)

В библиотеке существуют **два независимых механизма** групп:

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Подход 1: LLMGroup (/team/group.py)                               │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ LLMGroup                                                    │   │
│  │   │                                                         │   │
│  │   ├── _orchestrator_team (внутренняя команда)              │   │
│  │   │       └── LLM Agent (router)                           │   │
│  │   │                                                         │   │
│  │   └── _teams: Dict[str, LLMTeam]                           │   │
│  │                                                             │   │
│  │   Метод: Итеративный routing через LLM агента              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Подход 2: GroupOrchestrator (/orchestration/group.py)             │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ GroupOrchestrator                                           │   │
│  │   │                                                         │   │
│  │   ├── role: GroupRole.REPORT_COLLECTOR                     │   │
│  │   │                                                         │   │
│  │   └── _teams: Dict[str, LLMTeam]                           │   │
│  │                                                             │   │
│  │   Метод: Sequential/Parallel execution + report collection │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Проблемы текущей архитектуры

| # | Проблема | Влияние |
|---|----------|---------|
| 1 | **Дублирование** — два класса делают похожее | Путаница в API, сложность поддержки |
| 2 | **Команды не знают о группе** — нет обратной связи | Невозможна эскалация, нет контекста |
| 3 | **TeamOrchestrator.promote_to_group() не используется** | Мёртвый код |
| 4 | **Нет единого интерфейса** для разных стратегий | Сложно расширять |
| 5 | **LLMGroup создаёт скрытую команду** — нарушает прозрачность | Сложно отлаживать |

### 1.3 Цель

**Единая архитектура групп:**
- Один класс координации (`GroupOrchestrator`) с разными ролями
- Bi-directional связь: группа ↔ команда
- TeamOrchestrator знает о группе и может эскалировать
- `LLMGroup` становится convenience wrapper

---

## 2. Архитектура

### 2.1 Новая структура

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   User Code                                                         │
│       │                                                             │
│       ▼                                                             │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │ LLMGroup (convenience wrapper)                              │  │
│   │     │                                                       │  │
│   │     └── GroupOrchestrator                                   │  │
│   │             │                                               │  │
│   │             ├── role: GroupRole                             │  │
│   │             │   (REPORT_COLLECTOR | COORDINATOR |           │  │
│   │             │    ROUTER | AGGREGATOR | ARBITER)             │  │
│   │             │                                               │  │
│   │             └── teams + GroupContext                        │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│              ┌───────────────┼───────────────┐                      │
│              ▼               ▼               ▼                      │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│   │   LLMTeam    │  │   LLMTeam    │  │   LLMTeam    │            │
│   │      +       │  │      +       │  │      +       │            │
│   │TeamOrchestrator│TeamOrchestrator│TeamOrchestrator│            │
│   │      ↑       │  │      ↑       │  │      ↑       │            │
│   │ GroupContext │  │ GroupContext │  │ GroupContext │            │
│   └──────────────┘  └──────────────┘  └──────────────┘            │
│                                                                     │
│   Bi-directional: Group знает о Teams, Teams знают о Group         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Роли GroupOrchestrator

```python
class GroupRole(Enum):
    """
    Роли GroupOrchestrator.
    
    Определяют стратегию координации команд.
    """

    # === Passive (не требует LLM) ===
    
    REPORT_COLLECTOR = "report_collector"
    """
    Пассивный сбор отчётов.
    
    Behavior:
    1. Выполняет команды (sequential или parallel)
    2. Собирает TeamReport с каждой команды
    3. Агрегирует в GroupReport
    4. Возвращает результаты
    
    НЕ принимает решений. НЕ меняет flow.
    Требует LLM: НЕТ
    """
    
    COORDINATOR = "coordinator"
    """
    Координация с передачей контекста.
    
    Behavior:
    1. Выполняет команды в заданном порядке
    2. Передаёт output команды N как input команде N+1
    3. Управляет data flow между командами
    4. Обрабатывает эскалации (retry/skip/abort)
    
    Следует заданному плану, НЕ принимает runtime решений.
    Требует LLM: НЕТ (опционально для эскалаций)
    """

    # === Active (требует LLM) ===
    
    ROUTER = "router"
    """
    Динамический routing между командами.
    
    Behavior:
    1. На каждом шаге решает какую команду вызвать
    2. Анализирует результат и решает продолжать или завершить
    3. Может зацикливаться (контролируется max_iterations)
    4. Поддерживает conversation history
    
    Требует LLM: ДА
    """
    
    AGGREGATOR = "aggregator"
    """
    Параллельное выполнение и агрегация.
    
    Behavior:
    1. Запускает команды параллельно
    2. Ждёт все результаты
    3. Агрегирует в единый output (стратегия настраивается)
    4. Стратегии: merge, reduce, vote, best_of
    
    Требует LLM: Опционально (для vote/best_of)
    """
    
    ARBITER = "arbiter"
    """
    Арбитраж конфликтов и принятие решений.
    
    Behavior:
    1. Получает результаты от нескольких команд
    2. Анализирует конфликты/расхождения
    3. Принимает решение какой результат использовать
    4. Может запросить повторное выполнение
    
    Требует LLM: ДА
    """
```

### 2.3 TeamRole — роль команды в группе

```python
class TeamRole(Enum):
    """
    Роль команды внутри группы.
    
    Определяет приоритет и права команды.
    """
    
    LEADER = "leader"
    """
    Лидер группы.
    - Вызывается первым (если не указано иное)
    - Получает нерешённые задачи
    - Может быть только один на группу
    """
    
    MEMBER = "member"
    """
    Обычный участник.
    - Вызывается по решению GroupOrchestrator
    - Может эскалировать на группу
    """
    
    SPECIALIST = "specialist"
    """
    Специализированная команда.
    - Вызывается только при явном запросе
    - Обычно для специфичных задач
    - Может быть вызвана другими командами
    """
    
    FALLBACK = "fallback"
    """
    Резервная команда.
    - Вызывается при ошибках других команд
    - Обрабатывает edge cases
    """
```

---

## 3. GroupContext — связь группа ↔ команда

### 3.1 Структура GroupContext

```python
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from llmteam.orchestration import GroupOrchestrator


@dataclass
class GroupContext:
    """
    Контекст группы для команды.
    
    Передаётся команде при добавлении в группу.
    Обеспечивает bi-directional связь.
    """
    
    # Identity
    group_id: str
    """ID группы."""
    
    group_orchestrator: "GroupOrchestrator"
    """Ссылка на GroupOrchestrator (для эскалаций)."""
    
    # Team's role
    team_role: TeamRole
    """Роль этой команды в группе."""
    
    # Group info
    other_teams: List[str] = field(default_factory=list)
    """IDs других команд в группе."""
    
    leader_team: Optional[str] = None
    """ID команды-лидера."""
    
    # Permissions
    can_escalate: bool = True
    """Может ли команда эскалировать на группу."""
    
    can_request_team: bool = False
    """Может ли команда запрашивать другие команды напрямую."""
    
    visible_teams: Set[str] = field(default_factory=set)
    """Какие команды видны этой команде (для request_team)."""
    
    # Shared state
    shared_context: Dict[str, Any] = field(default_factory=dict)
    """Общий контекст группы (read-only для команд)."""
    
    # Callbacks
    on_escalation: Optional[callable] = None
    """Callback для эскалации (устанавливается GroupOrchestrator)."""


@dataclass
class EscalationRequest:
    """Запрос на эскалацию от команды."""
    
    source_team_id: str
    """Команда-источник."""
    
    source_agent_id: Optional[str] = None
    """Агент-источник (если эскалация от агента)."""
    
    reason: str = ""
    """Причина эскалации."""
    
    error: Optional[Exception] = None
    """Ошибка (если есть)."""
    
    context: Dict[str, Any] = field(default_factory=dict)
    """Дополнительный контекст."""
    
    suggested_action: Optional[str] = None
    """Предложенное действие от команды."""


@dataclass  
class EscalationResponse:
    """Ответ на эскалацию от GroupOrchestrator."""
    
    action: "EscalationAction"
    """Действие для команды."""
    
    reason: str = ""
    """Обоснование решения."""
    
    retry_with: Optional[Dict[str, Any]] = None
    """Данные для retry (если action=RETRY)."""
    
    route_to_team: Optional[str] = None
    """Команда для перенаправления (если action=REROUTE)."""
    
    additional_context: Dict[str, Any] = field(default_factory=dict)
    """Дополнительный контекст."""


class EscalationAction(Enum):
    """Действия при эскалации."""
    
    RETRY = "retry"
    """Повторить выполнение."""
    
    SKIP = "skip"
    """Пропустить и продолжить."""
    
    REROUTE = "reroute"
    """Перенаправить на другую команду."""
    
    ABORT = "abort"
    """Прервать выполнение группы."""
    
    CONTINUE = "continue"
    """Продолжить как есть."""
    
    HUMAN = "human"
    """Требуется вмешательство человека."""
```

### 3.2 Протокол регистрации

```python
# В GroupOrchestrator

class GroupOrchestrator:
    """Координатор группы команд."""
    
    def __init__(
        self,
        group_id: Optional[str] = None,
        role: GroupRole = GroupRole.REPORT_COLLECTOR,
        model: str = "gpt-4o-mini",  # Для ролей требующих LLM
    ):
        self.group_id = group_id or f"group_{uuid.uuid4().hex[:8]}"
        self._role = role
        self._model = model
        
        self._teams: Dict[str, "LLMTeam"] = {}
        self._team_roles: Dict[str, TeamRole] = {}
        self._leader_id: Optional[str] = None
        
        self._last_report: Optional[GroupReport] = None
        self._shared_context: Dict[str, Any] = {}
        
        # LLM для ролей ROUTER/ARBITER
        self._llm = None
    
    def add_team(
        self,
        team: "LLMTeam",
        role: TeamRole = TeamRole.MEMBER,
    ) -> None:
        """
        Добавить команду в группу.
        
        Args:
            team: Команда для добавления
            role: Роль команды в группе
        """
        # Проверка: только один LEADER
        if role == TeamRole.LEADER:
            if self._leader_id is not None:
                raise ValueError(
                    f"Group already has leader: {self._leader_id}. "
                    f"Remove it first or use MEMBER role."
                )
            self._leader_id = team.team_id
        
        # Регистрируем команду
        self._teams[team.team_id] = team
        self._team_roles[team.team_id] = role
        
        # Создаём контекст для команды
        context = GroupContext(
            group_id=self.group_id,
            group_orchestrator=self,
            team_role=role,
            other_teams=[tid for tid in self._teams.keys() if tid != team.team_id],
            leader_team=self._leader_id,
            can_escalate=True,
            can_request_team=(role in (TeamRole.LEADER, TeamRole.SPECIALIST)),
            visible_teams=set(self._teams.keys()) - {team.team_id},
            shared_context=self._shared_context,
            on_escalation=self._handle_escalation,
        )
        
        # ВАЖНО: Уведомляем команду о включении в группу
        team._join_group(context)
        
        # Обновляем контексты других команд
        self._update_team_contexts()
    
    def remove_team(self, team_id: str) -> bool:
        """Удалить команду из группы."""
        if team_id not in self._teams:
            return False
        
        team = self._teams.pop(team_id)
        self._team_roles.pop(team_id, None)
        
        if self._leader_id == team_id:
            self._leader_id = None
        
        # Уведомляем команду о выходе
        team._leave_group()
        
        # Обновляем контексты оставшихся команд
        self._update_team_contexts()
        
        return True
    
    def _update_team_contexts(self) -> None:
        """Обновить контексты всех команд."""
        team_ids = list(self._teams.keys())
        
        for team_id, team in self._teams.items():
            if team._group_context:
                team._group_context.other_teams = [
                    tid for tid in team_ids if tid != team_id
                ]
                team._group_context.leader_team = self._leader_id
                team._group_context.visible_teams = set(team_ids) - {team_id}
    
    async def _handle_escalation(
        self,
        request: EscalationRequest,
    ) -> EscalationResponse:
        """
        Обработать эскалацию от команды.
        
        Вызывается через GroupContext.on_escalation.
        """
        # Логируем
        self._log_escalation(request)
        
        # Решение зависит от роли
        if self._role in (GroupRole.ROUTER, GroupRole.ARBITER):
            # Используем LLM для принятия решения
            return await self._llm_escalation_decision(request)
        
        elif self._role == GroupRole.COORDINATOR:
            # Стандартная логика: retry → skip → abort
            return self._coordinator_escalation_decision(request)
        
        else:
            # REPORT_COLLECTOR / AGGREGATOR: просто логируем
            return EscalationResponse(
                action=EscalationAction.CONTINUE,
                reason="Report collector does not handle escalations",
            )
    
    def _coordinator_escalation_decision(
        self,
        request: EscalationRequest,
    ) -> EscalationResponse:
        """Решение для COORDINATOR роли."""
        # Если ошибка — retry один раз
        if request.error:
            return EscalationResponse(
                action=EscalationAction.RETRY,
                reason="Coordinator auto-retry on error",
            )
        
        # Иначе — продолжаем
        return EscalationResponse(
            action=EscalationAction.CONTINUE,
            reason="No error, continuing",
        )
    
    async def _llm_escalation_decision(
        self,
        request: EscalationRequest,
    ) -> EscalationResponse:
        """Решение через LLM для ROUTER/ARBITER."""
        llm = self._get_llm()
        
        prompt = self._build_escalation_prompt(request)
        response = await llm.complete(prompt)
        
        return self._parse_escalation_response(response)
```

### 3.3 LLMTeam — интеграция с группой

```python
# В LLMTeam

class LLMTeam:
    """Команда AI-агентов."""
    
    def __init__(self, ...):
        # ... existing code ...
        
        # Group context (None если не в группе)
        self._group_context: Optional[GroupContext] = None
    
    # === Group Integration ===
    
    def _join_group(self, context: GroupContext) -> None:
        """
        INTERNAL: Вызывается GroupOrchestrator при добавлении в группу.
        
        Не вызывать напрямую! Используйте GroupOrchestrator.add_team().
        """
        self._group_context = context
        
        # Уведомляем TeamOrchestrator
        if self._orchestrator:
            self._orchestrator._set_group_context(context)
        
        # Event
        self._emit_event("group.joined", {
            "group_id": context.group_id,
            "role": context.team_role.value,
        })
    
    def _leave_group(self) -> None:
        """
        INTERNAL: Вызывается GroupOrchestrator при удалении из группы.
        """
        if self._group_context:
            group_id = self._group_context.group_id
            self._group_context = None
            
            if self._orchestrator:
                self._orchestrator._clear_group_context()
            
            self._emit_event("group.left", {"group_id": group_id})
    
    @property
    def is_in_group(self) -> bool:
        """Команда входит в группу?"""
        return self._group_context is not None
    
    @property
    def group_id(self) -> Optional[str]:
        """ID группы (если есть)."""
        return self._group_context.group_id if self._group_context else None
    
    @property
    def group_role(self) -> Optional[TeamRole]:
        """Роль в группе (если есть)."""
        return self._group_context.team_role if self._group_context else None
    
    async def escalate_to_group(
        self,
        reason: str,
        context: Optional[Dict] = None,
        error: Optional[Exception] = None,
    ) -> EscalationResponse:
        """
        Эскалировать на группу.
        
        Args:
            reason: Причина эскалации
            context: Дополнительный контекст
            error: Ошибка (если есть)
        
        Returns:
            EscalationResponse от GroupOrchestrator
        
        Raises:
            RuntimeError: Если команда не в группе или эскалация запрещена
        """
        if not self._group_context:
            raise RuntimeError(
                f"Team '{self.team_id}' is not in a group. "
                f"Cannot escalate."
            )
        
        if not self._group_context.can_escalate:
            raise RuntimeError(
                f"Team '{self.team_id}' is not allowed to escalate. "
                f"Check group configuration."
            )
        
        request = EscalationRequest(
            source_team_id=self.team_id,
            reason=reason,
            context=context or {},
            error=error,
        )
        
        # Вызываем callback установленный GroupOrchestrator
        if self._group_context.on_escalation:
            return await self._group_context.on_escalation(request)
        
        # Fallback: напрямую к GroupOrchestrator
        return await self._group_context.group_orchestrator._handle_escalation(request)
    
    async def request_team(
        self,
        target_team_id: str,
        task: Dict[str, Any],
    ) -> Any:
        """
        Запросить выполнение от другой команды в группе.
        
        Доступно только для LEADER и SPECIALIST ролей.
        
        Args:
            target_team_id: ID целевой команды
            task: Задача для выполнения
        
        Returns:
            Результат выполнения целевой команды
        
        Raises:
            RuntimeError: Если команда не в группе
            PermissionError: Если нет прав на request_team
            ValueError: Если целевая команда не видна
        """
        if not self._group_context:
            raise RuntimeError(f"Team '{self.team_id}' is not in a group")
        
        if not self._group_context.can_request_team:
            raise PermissionError(
                f"Team '{self.team_id}' (role={self._group_context.team_role.value}) "
                f"is not allowed to request other teams"
            )
        
        if target_team_id not in self._group_context.visible_teams:
            raise ValueError(
                f"Team '{target_team_id}' is not visible to '{self.team_id}'"
            )
        
        return await self._group_context.group_orchestrator.route_to_team(
            source_team_id=self.team_id,
            target_team_id=target_team_id,
            task=task,
        )
```

### 3.4 TeamOrchestrator — изменения

```python
# В TeamOrchestrator

class TeamOrchestrator:
    """Супервайзер команды агентов."""
    
    def __init__(self, team: "LLMTeam", config: Optional[OrchestratorConfig] = None):
        # ... existing code ...
        
        # Group context (None если команда не в группе)
        self._group_context: Optional[GroupContext] = None
    
    # === Group Integration ===
    
    def _set_group_context(self, context: GroupContext) -> None:
        """
        INTERNAL: Установить контекст группы.
        
        Вызывается из LLMTeam._join_group().
        """
        self._group_context = context
        self._scope = OrchestratorScope.GROUP
    
    def _clear_group_context(self) -> None:
        """
        INTERNAL: Очистить контекст группы.
        
        Вызывается из LLMTeam._leave_group().
        """
        self._group_context = None
        self._scope = OrchestratorScope.TEAM
    
    @property
    def is_in_group(self) -> bool:
        """Команда в группе?"""
        return self._group_context is not None
    
    # === Modified behavior when in group ===
    
    async def handle_agent_error(
        self,
        agent_id: str,
        error: Exception,
        context: Dict[str, Any],
    ) -> RecoveryDecision:
        """
        Обработка ошибки агента.
        
        Если команда в группе — может эскалировать на GroupOrchestrator.
        """
        # Сначала пробуем локально
        if self._config.auto_retry:
            return RecoveryDecision(
                action=RecoveryAction.RETRY,
                reason="Auto-retry enabled",
            )
        
        # Если в группе и разрешена эскалация — эскалируем
        if self._group_context and self._group_context.can_escalate:
            response = await self._team.escalate_to_group(
                reason=f"Agent '{agent_id}' failed: {error}",
                context=context,
                error=error,
            )
            
            # Конвертируем EscalationResponse в RecoveryDecision
            return self._escalation_to_recovery(response)
        
        # Default: abort
        return RecoveryDecision(
            action=RecoveryAction.ABORT,
            reason="No recovery options available",
        )
    
    def _escalation_to_recovery(
        self,
        response: EscalationResponse,
    ) -> RecoveryDecision:
        """Конвертация EscalationResponse в RecoveryDecision."""
        action_map = {
            EscalationAction.RETRY: RecoveryAction.RETRY,
            EscalationAction.SKIP: RecoveryAction.SKIP,
            EscalationAction.ABORT: RecoveryAction.ABORT,
            EscalationAction.CONTINUE: RecoveryAction.SKIP,  # Continue = skip current
            EscalationAction.REROUTE: RecoveryAction.ESCALATE,  # Will be handled by group
            EscalationAction.HUMAN: RecoveryAction.ESCALATE,
        }
        
        return RecoveryDecision(
            action=action_map.get(response.action, RecoveryAction.ABORT),
            reason=response.reason,
            retry_with_changes=response.retry_with,
        )
    
    def generate_report(self) -> str:
        """Генерация отчёта с учётом группы."""
        report = self._generate_base_report()
        
        # Добавляем информацию о группе
        if self._group_context:
            report += "\n\n## Group Context\n"
            report += f"- **Group ID:** {self._group_context.group_id}\n"
            report += f"- **Role:** {self._group_context.team_role.value}\n"
            report += f"- **Leader:** {self._group_context.leader_team or 'None'}\n"
            report += f"- **Other teams:** {', '.join(self._group_context.other_teams)}\n"
        
        return report
    
    def get_summary(self) -> Dict[str, Any]:
        """Получить саммари с информацией о группе."""
        summary = self._get_base_summary()
        
        if self._group_context:
            summary["group"] = {
                "group_id": self._group_context.group_id,
                "team_role": self._group_context.team_role.value,
                "leader_team": self._group_context.leader_team,
                "can_escalate": self._group_context.can_escalate,
            }
        
        return summary
```

---

## 4. LLMGroup как Convenience Wrapper

### 4.1 Упрощённый интерфейс

```python
# /llmteam/team/group.py

class LLMGroup:
    """
    Convenience wrapper для GroupOrchestrator.
    
    Предоставляет простой API для работы с группами команд.
    
    Example:
        # Простое создание группы
        group = LLMGroup(
            group_id="customer_service",
            teams=[support_team, billing_team, tech_team],
            leader=support_team,
        )
        
        result = await group.run({"query": "Help with my bill"})
        
        # Или через team.create_group()
        group = support_team.create_group(
            group_id="customer_service",
            teams=[billing_team, tech_team],
        )
    """
    
    def __init__(
        self,
        group_id: str,
        teams: List["LLMTeam"],
        leader: Optional["LLMTeam"] = None,
        role: GroupRole = GroupRole.COORDINATOR,
        model: str = "gpt-4o-mini",
        **kwargs,
    ):
        """
        Создать группу команд.
        
        Args:
            group_id: Уникальный ID группы
            teams: Список команд
            leader: Команда-лидер (первая по умолчанию)
            role: Роль GroupOrchestrator
            model: Модель для LLM (если роль требует)
        """
        from llmteam.orchestration import GroupOrchestrator
        
        # Создаём orchestrator
        self._orchestrator = GroupOrchestrator(
            group_id=group_id,
            role=role,
            model=model,
        )
        
        # Определяем лидера
        leader = leader or (teams[0] if teams else None)
        
        # Добавляем команды
        for team in teams:
            team_role = TeamRole.LEADER if team is leader else TeamRole.MEMBER
            self._orchestrator.add_team(team, role=team_role)
    
    @property
    def group_id(self) -> str:
        """ID группы."""
        return self._orchestrator.group_id
    
    @property
    def orchestrator(self) -> "GroupOrchestrator":
        """Underlying GroupOrchestrator."""
        return self._orchestrator
    
    @property
    def teams(self) -> List["LLMTeam"]:
        """Все команды."""
        return list(self._orchestrator._teams.values())
    
    @property
    def leader(self) -> Optional["LLMTeam"]:
        """Команда-лидер."""
        if self._orchestrator._leader_id:
            return self._orchestrator._teams.get(self._orchestrator._leader_id)
        return None
    
    def add_team(
        self,
        team: "LLMTeam",
        role: TeamRole = TeamRole.MEMBER,
    ) -> None:
        """Добавить команду."""
        self._orchestrator.add_team(team, role=role)
    
    def remove_team(self, team_id: str) -> bool:
        """Удалить команду."""
        return self._orchestrator.remove_team(team_id)
    
    def get_team(self, team_id: str) -> Optional["LLMTeam"]:
        """Получить команду по ID."""
        return self._orchestrator.get_team(team_id)
    
    async def run(
        self,
        input_data: Dict[str, Any],
        run_id: Optional[str] = None,
        parallel: bool = False,
    ) -> "GroupResult":
        """
        Выполнить группу.
        
        Args:
            input_data: Входные данные
            run_id: ID выполнения
            parallel: Параллельное выполнение (для AGGREGATOR)
        
        Returns:
            GroupResult с результатами и отчётами
        """
        return await self._orchestrator.execute(
            input_data=input_data,
            run_id=run_id,
            parallel=parallel,
        )
    
    @property
    def last_report(self) -> Optional["GroupReport"]:
        """Последний отчёт."""
        return self._orchestrator.last_report
    
    def __repr__(self) -> str:
        return (
            f"<LLMGroup id='{self.group_id}' "
            f"teams={len(self.teams)} "
            f"role={self._orchestrator._role.value}>"
        )
    
    def __len__(self) -> int:
        return len(self.teams)
```

---

## 5. Примеры использования

### 5.1 Базовое использование (COORDINATOR)

```python
from llmteam import LLMTeam, LLMGroup
from llmteam.orchestration import GroupRole

# Создаём команды
research = LLMTeam(team_id="research", agents=[...])
writing = LLMTeam(team_id="writing", agents=[...])
editing = LLMTeam(team_id="editing", agents=[...])

# Создаём группу с координацией
group = LLMGroup(
    group_id="content_pipeline",
    teams=[research, writing, editing],
    role=GroupRole.COORDINATOR,  # Sequential с передачей контекста
)

# Выполняем
result = await group.run({"topic": "AI trends 2026"})

# Результат содержит output всех команд
print(result.output)  # {"research": ..., "writing": ..., "editing": ...}
print(result.report.summary)  # "Executed 3 teams: 3 succeeded, 0 failed"
```

### 5.2 Динамический routing (ROUTER)

```python
from llmteam.orchestration import GroupRole

# Создаём группу с routing
group = LLMGroup(
    group_id="customer_service",
    teams=[support, billing, tech],
    leader=support,  # Default если routing не определился
    role=GroupRole.ROUTER,  # LLM решает куда направить
    model="gpt-4o",
)

# Запрос будет направлен к нужной команде
result = await group.run({
    "query": "I have a technical issue with my billing dashboard"
})

# Может вызвать несколько команд по цепочке
print(result.team_results.keys())  # ["tech", "billing"]
```

### 5.3 Эскалация из команды

```python
# Внутри агента или команды
async def handle_complex_case(team: LLMTeam, data: dict):
    try:
        # Пробуем обработать
        result = await process(data)
        return result
    except ComplexCaseError as e:
        # Эскалируем на группу
        if team.is_in_group:
            response = await team.escalate_to_group(
                reason="Case too complex for this team",
                context={"data": data, "partial_result": partial},
                error=e,
            )
            
            if response.action == EscalationAction.REROUTE:
                # Группа направила на другую команду
                return response.additional_context.get("result")
            elif response.action == EscalationAction.HUMAN:
                # Нужен человек
                return {"status": "pending_human_review"}
        
        raise  # Если не в группе — пробрасываем
```

### 5.4 Запрос к другой команде (LEADER/SPECIALIST)

```python
# Команда-лидер может запросить специалиста
async def leader_workflow(team: LLMTeam, query: dict):
    # Основная обработка
    result = await team.run(query)
    
    # Если нужен специалист
    if needs_legal_review(result):
        legal_result = await team.request_team(
            target_team_id="legal",
            task={"review": result.output, "type": "compliance"}
        )
        result.output["legal_review"] = legal_result
    
    return result
```

---

## 6. Migration Guide

### 6.1 Breaking Changes

| Изменение | Было | Стало |
|-----------|------|-------|
| `LLMGroup` API | `LLMGroup(leader, teams)` | `LLMGroup(teams, leader=)` |
| Роль по умолчанию | Implicit ROUTER | Explicit `role=` parameter |
| `_orchestrator_team` | Создавался автоматически | Удалён, используется `GroupOrchestrator` |

### 6.2 Code Migration

```python
# BEFORE (v5.0.0)
group = LLMGroup(
    group_id="service",
    leader=support,
    teams=[billing, tech],
)

# AFTER (v5.1.0)
# Вариант 1: Явный ROUTER (как было)
group = LLMGroup(
    group_id="service",
    teams=[support, billing, tech],
    leader=support,
    role=GroupRole.ROUTER,
)

# Вариант 2: COORDINATOR (проще, без LLM)
group = LLMGroup(
    group_id="service",
    teams=[support, billing, tech],
    leader=support,
    role=GroupRole.COORDINATOR,
)
```

### 6.3 Deprecation warnings

```python
# В LLMGroup.__init__
if 'leader' in kwargs and 'teams' not in kwargs:
    warnings.warn(
        "LLMGroup(leader=, teams=) signature is deprecated. "
        "Use LLMGroup(teams=, leader=) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
```

---

## 7. Оценка

### 7.1 Риски

| Риск | Вероятность | Влияние | Митигация |
|------|-------------|---------|-----------|
| Breaking changes в LLMGroup | Высокая | Среднее | Deprecation warnings |
| Сложность GroupContext | Средняя | Низкое | Хорошая документация |
| Циклические эскалации | Низкая | Высокое | Max escalation depth |
| Performance overhead | Низкая | Низкое | Lazy context updates |

### 7.2 Целесообразность

| Преимущество | Вес |
|--------------|-----|
| Единая архитектура групп | ⭐⭐⭐⭐⭐ |
| Bi-directional связь | ⭐⭐⭐⭐ |
| Эскалация из команды | ⭐⭐⭐⭐ |
| Гибкие роли GroupOrchestrator | ⭐⭐⭐ |
| Прозрачность (нет скрытых команд) | ⭐⭐⭐ |

**Вердикт: ВЫСОКО ЦЕЛЕСООБРАЗНО**

---

## 8. План реализации

| Неделя | Задачи |
|--------|--------|
| W1 | `GroupContext`, `TeamRole`, `EscalationRequest/Response` |
| W2 | `GroupOrchestrator` — новые роли (COORDINATOR, ROUTER) |
| W3 | `LLMTeam` — `_join_group`, `escalate_to_group`, `request_team` |
| W4 | `TeamOrchestrator` — интеграция с `GroupContext` |
| W5 | `LLMGroup` — refactor как wrapper |
| W6 | Тесты, документация, migration guide |

**Общее время: 6 недель**

---

## 9. Definition of Done

- [ ] `GroupContext` dataclass реализован
- [ ] `TeamRole` enum реализован
- [ ] `EscalationRequest/Response` реализованы
- [ ] `GroupOrchestrator` поддерживает все роли
- [ ] `GroupOrchestrator.add_team()` уведомляет команду
- [ ] `LLMTeam._join_group()/_leave_group()` реализованы
- [ ] `LLMTeam.escalate_to_group()` работает
- [ ] `LLMTeam.request_team()` работает
- [ ] `TeamOrchestrator` знает о группе и эскалирует
- [ ] `LLMGroup` переписан как wrapper
- [ ] Тесты: escalation flow, request_team, all roles
- [ ] Документация обновлена
- [ ] Migration guide написан

---

## 10. Open Questions

1. **Max escalation depth** — сколько уровней эскалации разрешить? (предложение: 3)

2. **Shared context persistence** — нужно ли сохранять shared_context между run'ами?

3. **Team visibility** — должны ли MEMBER команды видеть друг друга?

4. **Nested groups** — поддерживать ли группы групп? (предложение: НЕТ для MVP)

---

**Рекомендация: ПРИНЯТЬ**

Критично для production-ready библиотеки. Устраняет архитектурный долг и обеспечивает полноценную координацию команд.
