# RFC-008 Implementation Code

**Готовый код для внедрения RFC-008**

---

## 1. Новые модели (`/orchestration/models.py`)

```python
"""
Models for group orchestration.

RFC-008: Group Architecture Unification
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

if TYPE_CHECKING:
    from llmteam.orchestration.group import GroupOrchestrator


class GroupRole(Enum):
    """
    Роли GroupOrchestrator.
    
    Определяют стратегию координации команд.
    """

    # === Passive (не требует LLM) ===
    
    REPORT_COLLECTOR = "report_collector"
    """Пассивный сбор отчётов. НЕ принимает решений."""
    
    COORDINATOR = "coordinator"
    """Координация с передачей контекста между командами."""

    # === Active (требует LLM) ===
    
    ROUTER = "router"
    """Динамический routing между командами через LLM."""
    
    AGGREGATOR = "aggregator"
    """Параллельное выполнение и агрегация результатов."""
    
    ARBITER = "arbiter"
    """Арбитраж конфликтов между командами через LLM."""


class TeamRole(Enum):
    """
    Роль команды внутри группы.
    """
    
    LEADER = "leader"
    """Лидер группы. Вызывается первым, получает нерешённые задачи."""
    
    MEMBER = "member"
    """Обычный участник."""
    
    SPECIALIST = "specialist"
    """Специализированная команда. Вызывается по запросу."""
    
    FALLBACK = "fallback"
    """Резервная команда. Вызывается при ошибках."""


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
    """Ссылка на GroupOrchestrator."""
    
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
    """Какие команды видны этой команде."""
    
    # Shared state
    shared_context: Dict[str, Any] = field(default_factory=dict)
    """Общий контекст группы (read-only для команд)."""
    
    # Callbacks
    on_escalation: Optional[Callable] = field(default=None, repr=False)
    """Callback для эскалации."""


@dataclass
class EscalationRequest:
    """Запрос на эскалацию от команды."""
    
    source_team_id: str
    """Команда-источник."""
    
    source_agent_id: Optional[str] = None
    """Агент-источник (если эскалация от агента)."""
    
    reason: str = ""
    """Причина эскалации."""
    
    error: Optional[Exception] = field(default=None, repr=False)
    """Ошибка (если есть)."""
    
    context: Dict[str, Any] = field(default_factory=dict)
    """Дополнительный контекст."""
    
    suggested_action: Optional[str] = None
    """Предложенное действие от команды."""
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    """Время создания."""


@dataclass  
class EscalationResponse:
    """Ответ на эскалацию от GroupOrchestrator."""
    
    action: EscalationAction
    """Действие для команды."""
    
    reason: str = ""
    """Обоснование решения."""
    
    retry_with: Optional[Dict[str, Any]] = None
    """Данные для retry (если action=RETRY)."""
    
    route_to_team: Optional[str] = None
    """Команда для перенаправления (если action=REROUTE)."""
    
    additional_context: Dict[str, Any] = field(default_factory=dict)
    """Дополнительный контекст."""
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    """Время создания."""


@dataclass
class TeamReport:
    """Отчёт от команды."""
    
    team_id: str
    run_id: str
    success: bool
    duration_ms: int
    agents_executed: List[str] = field(default_factory=list)
    agent_reports: List[Dict[str, Any]] = field(default_factory=list)
    output_summary: str = ""
    errors: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Group context
    team_role: Optional[str] = None
    escalations_sent: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "team_id": self.team_id,
            "run_id": self.run_id,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "agents_executed": self.agents_executed,
            "agent_reports": self.agent_reports,
            "output_summary": self.output_summary,
            "errors": self.errors,
            "created_at": self.created_at.isoformat(),
            "team_role": self.team_role,
            "escalations_sent": self.escalations_sent,
        }


@dataclass
class GroupReport:
    """Агрегированный отчёт от GroupOrchestrator."""
    
    group_id: str
    role: str
    run_id: str = ""
    teams_count: int = 0
    teams_succeeded: int = 0
    teams_failed: int = 0
    total_duration_ms: int = 0
    team_reports: List[TeamReport] = field(default_factory=list)
    escalations_handled: int = 0
    summary: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "group_id": self.group_id,
            "role": self.role,
            "run_id": self.run_id,
            "teams_count": self.teams_count,
            "teams_succeeded": self.teams_succeeded,
            "teams_failed": self.teams_failed,
            "total_duration_ms": self.total_duration_ms,
            "team_reports": [r.to_dict() for r in self.team_reports],
            "escalations_handled": self.escalations_handled,
            "summary": self.summary,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class GroupResult:
    """Результат выполнения группы."""
    
    success: bool
    output: Dict[str, Any] = field(default_factory=dict)
    team_results: Dict[str, Any] = field(default_factory=dict)
    report: Optional[GroupReport] = None
    errors: List[str] = field(default_factory=list)
    run_id: str = ""
    duration_ms: int = 0
```

---

## 2. Обновлённый GroupOrchestrator (`/orchestration/group.py`)

```python
"""
GroupOrchestrator for multi-team coordination.

RFC-008: Group Architecture Unification
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from llmteam.orchestration.models import (
    GroupRole,
    TeamRole,
    GroupContext,
    EscalationRequest,
    EscalationResponse,
    EscalationAction,
    TeamReport,
    GroupReport,
    GroupResult,
)

if TYPE_CHECKING:
    from llmteam.team import LLMTeam
    from llmteam.team.result import RunResult


class GroupOrchestrator:
    """
    Orchestrator for a group of teams.

    RFC-008: Unified group coordination with multiple roles.

    Roles:
    - REPORT_COLLECTOR: Passive report collection (default)
    - COORDINATOR: Sequential execution with context passing
    - ROUTER: Dynamic LLM-based routing
    - AGGREGATOR: Parallel execution with result aggregation
    - ARBITER: Conflict resolution between teams

    Usage:
        orch = GroupOrchestrator(
            group_id="my_group",
            role=GroupRole.COORDINATOR,
        )
        orch.add_team(team1, role=TeamRole.LEADER)
        orch.add_team(team2, role=TeamRole.MEMBER)

        result = await orch.execute({"query": "..."})
    """

    def __init__(
        self,
        group_id: Optional[str] = None,
        role: GroupRole = GroupRole.REPORT_COLLECTOR,
        model: str = "gpt-4o-mini",
        max_iterations: int = 10,
        max_escalation_depth: int = 3,
    ):
        """
        Initialize GroupOrchestrator.

        Args:
            group_id: Unique group identifier (auto-generated if None).
            role: Orchestrator role.
            model: LLM model for roles requiring it (ROUTER, ARBITER).
            max_iterations: Max routing iterations (for ROUTER).
            max_escalation_depth: Max escalation depth to prevent loops.
        """
        self.group_id = group_id or f"group_{uuid.uuid4().hex[:8]}"
        self._role = role
        self._model = model
        self._max_iterations = max_iterations
        self._max_escalation_depth = max_escalation_depth

        # Teams
        self._teams: Dict[str, "LLMTeam"] = {}
        self._team_roles: Dict[str, TeamRole] = {}
        self._leader_id: Optional[str] = None

        # State
        self._last_report: Optional[GroupReport] = None
        self._shared_context: Dict[str, Any] = {}
        self._escalation_count: int = 0
        self._current_run_id: Optional[str] = None

        # LLM (lazy init)
        self._llm = None

    # === Team Management ===

    def add_team(
        self,
        team: "LLMTeam",
        role: TeamRole = TeamRole.MEMBER,
    ) -> None:
        """
        Add a team to the group.

        Args:
            team: LLMTeam instance to add.
            role: Role of the team in the group.

        Raises:
            ValueError: If trying to add second LEADER.
        """
        # Validate: only one LEADER
        if role == TeamRole.LEADER:
            if self._leader_id is not None:
                raise ValueError(
                    f"Group already has leader: {self._leader_id}. "
                    f"Remove it first or use MEMBER role."
                )
            self._leader_id = team.team_id

        # Register team
        self._teams[team.team_id] = team
        self._team_roles[team.team_id] = role

        # Create context for team
        context = self._create_team_context(team.team_id, role)

        # IMPORTANT: Notify team about joining group
        team._join_group(context)

        # Update contexts of other teams
        self._update_team_contexts()

    def remove_team(self, team_id: str) -> bool:
        """
        Remove a team from the group.

        Args:
            team_id: ID of team to remove.

        Returns:
            True if team was removed, False if not found.
        """
        if team_id not in self._teams:
            return False

        team = self._teams.pop(team_id)
        self._team_roles.pop(team_id, None)

        if self._leader_id == team_id:
            self._leader_id = None

        # Notify team about leaving
        team._leave_group()

        # Update remaining teams
        self._update_team_contexts()

        return True

    def _create_team_context(
        self,
        team_id: str,
        role: TeamRole,
    ) -> GroupContext:
        """Create GroupContext for a team."""
        return GroupContext(
            group_id=self.group_id,
            group_orchestrator=self,
            team_role=role,
            other_teams=[tid for tid in self._teams.keys() if tid != team_id],
            leader_team=self._leader_id,
            can_escalate=True,
            can_request_team=(role in (TeamRole.LEADER, TeamRole.SPECIALIST)),
            visible_teams=set(self._teams.keys()) - {team_id},
            shared_context=self._shared_context,
            on_escalation=self._handle_escalation,
        )

    def _update_team_contexts(self) -> None:
        """Update contexts of all teams."""
        team_ids = list(self._teams.keys())

        for team_id, team in self._teams.items():
            if hasattr(team, "_group_context") and team._group_context:
                team._group_context.other_teams = [
                    tid for tid in team_ids if tid != team_id
                ]
                team._group_context.leader_team = self._leader_id
                team._group_context.visible_teams = set(team_ids) - {team_id}

    def list_teams(self) -> List[str]:
        """List team IDs in the group."""
        return list(self._teams.keys())

    def get_team(self, team_id: str) -> Optional["LLMTeam"]:
        """Get a team by ID."""
        return self._teams.get(team_id)

    def get_team_role(self, team_id: str) -> Optional[TeamRole]:
        """Get team's role in the group."""
        return self._team_roles.get(team_id)

    @property
    def teams_count(self) -> int:
        """Number of teams in the group."""
        return len(self._teams)

    @property
    def leader(self) -> Optional["LLMTeam"]:
        """Leader team."""
        return self._teams.get(self._leader_id) if self._leader_id else None

    # === Execution ===

    async def execute(
        self,
        input_data: Dict[str, Any],
        run_id: Optional[str] = None,
        parallel: bool = False,
    ) -> GroupResult:
        """
        Execute all teams according to role.

        Args:
            input_data: Input data for teams.
            run_id: Run identifier.
            parallel: Force parallel execution (for AGGREGATOR).

        Returns:
            GroupResult with results and reports.
        """
        run_id = run_id or str(uuid.uuid4())
        self._current_run_id = run_id
        self._escalation_count = 0
        start_time = datetime.utcnow()

        try:
            # Execute based on role
            if self._role == GroupRole.REPORT_COLLECTOR:
                result = await self._execute_report_collector(input_data, parallel)
            elif self._role == GroupRole.COORDINATOR:
                result = await self._execute_coordinator(input_data)
            elif self._role == GroupRole.ROUTER:
                result = await self._execute_router(input_data)
            elif self._role == GroupRole.AGGREGATOR:
                result = await self._execute_aggregator(input_data)
            elif self._role == GroupRole.ARBITER:
                result = await self._execute_arbiter(input_data)
            else:
                result = await self._execute_report_collector(input_data, parallel)

            # Finalize
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            result.run_id = run_id
            result.duration_ms = duration_ms

            if result.report:
                result.report.run_id = run_id
                result.report.total_duration_ms = duration_ms
                self._last_report = result.report

            return result

        finally:
            self._current_run_id = None

    async def _execute_report_collector(
        self,
        input_data: Dict[str, Any],
        parallel: bool,
    ) -> GroupResult:
        """Execute in REPORT_COLLECTOR mode."""
        if parallel:
            team_results, team_reports, errors = await self._run_teams_parallel(input_data)
        else:
            team_results, team_reports, errors = await self._run_teams_sequential(input_data)

        report = self._create_group_report(team_reports)

        return GroupResult(
            success=len(errors) == 0,
            output=self._merge_outputs(team_results),
            team_results=team_results,
            report=report,
            errors=errors,
        )

    async def _execute_coordinator(
        self,
        input_data: Dict[str, Any],
    ) -> GroupResult:
        """Execute in COORDINATOR mode with context passing."""
        team_results: Dict[str, Any] = {}
        team_reports: List[TeamReport] = []
        errors: List[str] = []

        # Determine execution order (leader first, then members)
        execution_order = self._get_execution_order()

        current_data = input_data.copy()

        for team_id in execution_order:
            team = self._teams[team_id]

            try:
                result = await team.run(current_data)
                team_results[team_id] = result

                report = self._create_team_report(team_id, team, result)
                team_reports.append(report)

                if not result.success:
                    # Handle via escalation if possible
                    response = await self._handle_team_failure(team_id, result)
                    if response.action == EscalationAction.ABORT:
                        errors.append(f"{team_id}: {result.error}")
                        break
                    elif response.action == EscalationAction.SKIP:
                        continue

                # Pass output to next team
                if hasattr(result, "output") and result.output:
                    if isinstance(result.output, dict):
                        current_data.update(result.output)
                    else:
                        current_data[f"{team_id}_output"] = result.output

            except Exception as e:
                errors.append(f"{team_id}: {str(e)}")
                # Try to continue with next team
                continue

        report = self._create_group_report(team_reports)

        return GroupResult(
            success=len(errors) == 0,
            output=self._merge_outputs(team_results),
            team_results=team_results,
            report=report,
            errors=errors,
        )

    async def _execute_router(
        self,
        input_data: Dict[str, Any],
    ) -> GroupResult:
        """Execute in ROUTER mode with LLM-based routing."""
        team_results: Dict[str, Any] = {}
        team_reports: List[TeamReport] = []
        errors: List[str] = []
        teams_called: List[str] = []

        current_data = input_data.copy()
        iterations = 0

        while iterations < self._max_iterations:
            iterations += 1

            # Decide which team to call
            next_team_id = await self._route_decision(current_data, teams_called)

            if not next_team_id:
                break  # Router decided to finish

            if next_team_id not in self._teams:
                # Fallback to leader
                next_team_id = self._leader_id or list(self._teams.keys())[0]

            team = self._teams[next_team_id]
            teams_called.append(next_team_id)

            try:
                result = await team.run(current_data)
                team_results[next_team_id] = result

                report = self._create_team_report(next_team_id, team, result)
                team_reports.append(report)

                if not result.success:
                    errors.append(f"{next_team_id}: {result.error}")
                    break

                # Update data
                if hasattr(result, "output") and result.output:
                    if isinstance(result.output, dict):
                        current_data.update(result.output)
                    else:
                        current_data[f"{next_team_id}_output"] = result.output

                # Check if we should continue
                should_continue = await self._should_continue_routing(
                    current_data, teams_called, result
                )
                if not should_continue:
                    break

            except Exception as e:
                errors.append(f"{next_team_id}: {str(e)}")
                break

        report = self._create_group_report(team_reports)

        return GroupResult(
            success=len(errors) == 0,
            output=self._merge_outputs(team_results),
            team_results=team_results,
            report=report,
            errors=errors,
        )

    async def _execute_aggregator(
        self,
        input_data: Dict[str, Any],
    ) -> GroupResult:
        """Execute in AGGREGATOR mode with parallel execution."""
        team_results, team_reports, errors = await self._run_teams_parallel(input_data)

        # Aggregate results
        aggregated_output = await self._aggregate_results(team_results)

        report = self._create_group_report(team_reports)

        return GroupResult(
            success=len(errors) == 0,
            output=aggregated_output,
            team_results=team_results,
            report=report,
            errors=errors,
        )

    async def _execute_arbiter(
        self,
        input_data: Dict[str, Any],
    ) -> GroupResult:
        """Execute in ARBITER mode with conflict resolution."""
        # First, run all teams in parallel
        team_results, team_reports, errors = await self._run_teams_parallel(input_data)

        if len(team_results) < 2:
            # Nothing to arbitrate
            report = self._create_group_report(team_reports)
            return GroupResult(
                success=len(errors) == 0,
                output=self._merge_outputs(team_results),
                team_results=team_results,
                report=report,
                errors=errors,
            )

        # Arbitrate between results
        final_output = await self._arbitrate_results(team_results)

        report = self._create_group_report(team_reports)

        return GroupResult(
            success=len(errors) == 0,
            output=final_output,
            team_results=team_results,
            report=report,
            errors=errors,
        )

    # === Escalation Handling ===

    async def _handle_escalation(
        self,
        request: EscalationRequest,
    ) -> EscalationResponse:
        """
        Handle escalation from a team.

        Called via GroupContext.on_escalation.
        """
        self._escalation_count += 1

        # Check depth limit
        if self._escalation_count > self._max_escalation_depth:
            return EscalationResponse(
                action=EscalationAction.ABORT,
                reason=f"Max escalation depth ({self._max_escalation_depth}) exceeded",
            )

        # Handle based on role
        if self._role in (GroupRole.ROUTER, GroupRole.ARBITER):
            return await self._llm_escalation_decision(request)
        elif self._role == GroupRole.COORDINATOR:
            return self._coordinator_escalation_decision(request)
        else:
            # REPORT_COLLECTOR / AGGREGATOR: just log
            return EscalationResponse(
                action=EscalationAction.CONTINUE,
                reason=f"{self._role.value} does not handle escalations",
            )

    def _coordinator_escalation_decision(
        self,
        request: EscalationRequest,
    ) -> EscalationResponse:
        """Decision for COORDINATOR role."""
        if request.error:
            return EscalationResponse(
                action=EscalationAction.RETRY,
                reason="Coordinator auto-retry on error",
            )
        return EscalationResponse(
            action=EscalationAction.CONTINUE,
            reason="No error, continuing",
        )

    async def _llm_escalation_decision(
        self,
        request: EscalationRequest,
    ) -> EscalationResponse:
        """LLM-based escalation decision."""
        llm = self._get_llm()
        if not llm:
            return EscalationResponse(
                action=EscalationAction.ABORT,
                reason="No LLM available for escalation decision",
            )

        prompt = self._build_escalation_prompt(request)

        try:
            response = await llm.complete(
                prompt=prompt,
                system_prompt="You are a group orchestrator handling an escalation. Decide the best action.",
                temperature=0.1,
                max_tokens=200,
            )
            return self._parse_escalation_response(response)
        except Exception:
            return EscalationResponse(
                action=EscalationAction.ABORT,
                reason="LLM call failed",
            )

    async def route_to_team(
        self,
        source_team_id: str,
        target_team_id: str,
        task: Dict[str, Any],
    ) -> Any:
        """
        Route a task from one team to another.

        Called by LLMTeam.request_team().
        """
        if target_team_id not in self._teams:
            raise ValueError(f"Team '{target_team_id}' not found in group")

        target_team = self._teams[target_team_id]
        result = await target_team.run(task)

        return result.output if hasattr(result, "output") else result

    async def _handle_team_failure(
        self,
        team_id: str,
        result: "RunResult",
    ) -> EscalationResponse:
        """Handle team execution failure."""
        request = EscalationRequest(
            source_team_id=team_id,
            reason=f"Team execution failed: {result.error}",
            error=Exception(result.error) if result.error else None,
        )
        return await self._handle_escalation(request)

    # === Helper Methods ===

    def _get_execution_order(self) -> List[str]:
        """Get team execution order (leader first)."""
        order = []

        # Leader first
        if self._leader_id and self._leader_id in self._teams:
            order.append(self._leader_id)

        # Then members
        for team_id, role in self._team_roles.items():
            if team_id not in order and role == TeamRole.MEMBER:
                order.append(team_id)

        # Specialists last (only if explicitly needed)
        # Fallback teams are not in normal order

        return order

    async def _run_teams_sequential(
        self,
        input_data: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[TeamReport], List[str]]:
        """Run teams sequentially."""
        team_results: Dict[str, Any] = {}
        team_reports: List[TeamReport] = []
        errors: List[str] = []

        for team_id, team in self._teams.items():
            try:
                result = await team.run(input_data)
                team_results[team_id] = result
                report = self._create_team_report(team_id, team, result)
                team_reports.append(report)
            except Exception as e:
                errors.append(f"{team_id}: {str(e)}")

        return team_results, team_reports, errors

    async def _run_teams_parallel(
        self,
        input_data: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[TeamReport], List[str]]:
        """Run teams in parallel."""

        async def run_team(team_id: str, team: "LLMTeam"):
            result = await team.run(input_data)
            report = self._create_team_report(team_id, team, result)
            return team_id, result, report

        tasks = [run_team(tid, t) for tid, t in self._teams.items()]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        team_results: Dict[str, Any] = {}
        team_reports: List[TeamReport] = []
        errors: List[str] = []

        for item in completed:
            if isinstance(item, Exception):
                errors.append(str(item))
            else:
                team_id, result, report = item
                team_results[team_id] = result
                team_reports.append(report)

        return team_results, team_reports, errors

    def _create_team_report(
        self,
        team_id: str,
        team: "LLMTeam",
        result: "RunResult",
    ) -> TeamReport:
        """Create TeamReport from execution result."""
        agents_executed = []
        agent_reports = []

        orchestrator = team.get_orchestrator()
        if orchestrator and hasattr(orchestrator, "_reports"):
            agent_reports = [r.to_dict() for r in orchestrator._reports]
            agents_executed = [r.agent_role for r in orchestrator._reports]

        if not agents_executed and hasattr(result, "agents_called"):
            agents_executed = result.agents_called or []

        return TeamReport(
            team_id=team_id,
            run_id=getattr(result, "run_id", "") or "",
            success=result.success if hasattr(result, "success") else True,
            duration_ms=getattr(result, "duration_ms", 0) or 0,
            agents_executed=agents_executed,
            agent_reports=agent_reports,
            output_summary=str(result.output)[:200] if hasattr(result, "output") else "",
            errors=[result.error] if hasattr(result, "error") and result.error else [],
            team_role=self._team_roles.get(team_id, TeamRole.MEMBER).value,
        )

    def _create_group_report(
        self,
        team_reports: List[TeamReport],
    ) -> GroupReport:
        """Create aggregated group report."""
        succeeded = sum(1 for r in team_reports if r.success)
        failed = len(team_reports) - succeeded

        summary = f"Executed {len(team_reports)} teams: {succeeded} succeeded, {failed} failed"

        return GroupReport(
            group_id=self.group_id,
            role=self._role.value,
            teams_count=len(team_reports),
            teams_succeeded=succeeded,
            teams_failed=failed,
            team_reports=team_reports,
            escalations_handled=self._escalation_count,
            summary=summary,
        )

    def _merge_outputs(self, team_results: Dict[str, Any]) -> Dict[str, Any]:
        """Merge outputs from all teams."""
        merged = {}
        for team_id, result in team_results.items():
            if hasattr(result, "output"):
                merged[team_id] = result.output
            elif hasattr(result, "final_output"):
                merged[team_id] = result.final_output
            else:
                merged[team_id] = result
        return merged

    async def _route_decision(
        self,
        current_data: Dict[str, Any],
        teams_called: List[str],
    ) -> Optional[str]:
        """LLM decision for ROUTER role."""
        llm = self._get_llm()
        if not llm:
            # Fallback: round-robin
            for team_id in self._teams:
                if team_id not in teams_called:
                    return team_id
            return None

        # Build prompt
        prompt = self._build_routing_prompt(current_data, teams_called)

        try:
            response = await llm.complete(
                prompt=prompt,
                system_prompt="You are a router deciding which team should handle the task.",
                temperature=0.1,
                max_tokens=100,
            )
            return self._parse_routing_response(response)
        except Exception:
            return self._leader_id

    async def _should_continue_routing(
        self,
        current_data: Dict[str, Any],
        teams_called: List[str],
        last_result: "RunResult",
    ) -> bool:
        """Decide if routing should continue."""
        llm = self._get_llm()
        if not llm:
            return len(teams_called) < len(self._teams)

        # Simple heuristic for now
        return len(teams_called) < self._max_iterations

    async def _aggregate_results(
        self,
        team_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Aggregate results from parallel execution."""
        # Default: merge all outputs
        return self._merge_outputs(team_results)

    async def _arbitrate_results(
        self,
        team_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Arbitrate between conflicting results."""
        llm = self._get_llm()
        if not llm:
            # Fallback: use leader's result
            if self._leader_id and self._leader_id in team_results:
                result = team_results[self._leader_id]
                return result.output if hasattr(result, "output") else {}
            return self._merge_outputs(team_results)

        # LLM arbitration
        prompt = self._build_arbitration_prompt(team_results)

        try:
            response = await llm.complete(
                prompt=prompt,
                system_prompt="You are an arbiter choosing the best result.",
                temperature=0.1,
                max_tokens=500,
            )
            return self._parse_arbitration_response(response, team_results)
        except Exception:
            return self._merge_outputs(team_results)

    def _get_llm(self):
        """Get or create LLM provider."""
        if self._llm is not None:
            return self._llm

        try:
            from llmteam.providers import OpenAIProvider
            self._llm = OpenAIProvider(model=self._model)
            return self._llm
        except ImportError:
            return None

    # === Prompt Builders ===

    def _build_escalation_prompt(self, request: EscalationRequest) -> str:
        """Build prompt for escalation decision."""
        return f"""
Escalation Request:
- Source Team: {request.source_team_id}
- Reason: {request.reason}
- Error: {str(request.error) if request.error else "None"}
- Context: {json.dumps(request.context, default=str)}

Available actions:
- RETRY: Retry the failed operation
- SKIP: Skip and continue with next step
- REROUTE: Route to another team
- ABORT: Stop execution
- CONTINUE: Continue as-is

Respond with JSON:
{{"action": "ACTION_NAME", "reason": "explanation", "route_to_team": "team_id_if_reroute"}}
"""

    def _build_routing_prompt(
        self,
        current_data: Dict[str, Any],
        teams_called: List[str],
    ) -> str:
        """Build prompt for routing decision."""
        team_info = []
        for team_id, role in self._team_roles.items():
            called = "✓" if team_id in teams_called else ""
            team_info.append(f"- {team_id} ({role.value}) {called}")

        return f"""
Current task data:
{json.dumps(current_data, default=str, indent=2)}

Available teams:
{chr(10).join(team_info)}

Teams already called: {teams_called}

Which team should handle this next? Or respond "DONE" if task is complete.

Respond with just the team_id or "DONE".
"""

    def _build_arbitration_prompt(
        self,
        team_results: Dict[str, Any],
    ) -> str:
        """Build prompt for arbitration."""
        results_str = []
        for team_id, result in team_results.items():
            output = result.output if hasattr(result, "output") else result
            results_str.append(f"Team '{team_id}':\n{json.dumps(output, default=str)}")

        return f"""
Multiple teams have produced results. Choose the best one or synthesize.

Results:
{chr(10).join(results_str)}

Respond with JSON:
{{"chosen_team": "team_id", "reason": "explanation"}}
or
{{"synthesized": true, "output": {{...}}, "reason": "explanation"}}
"""

    def _parse_escalation_response(self, response: str) -> EscalationResponse:
        """Parse LLM escalation response."""
        try:
            data = json.loads(response)
            action = EscalationAction[data.get("action", "ABORT").upper()]
            return EscalationResponse(
                action=action,
                reason=data.get("reason", ""),
                route_to_team=data.get("route_to_team"),
            )
        except (json.JSONDecodeError, KeyError):
            return EscalationResponse(
                action=EscalationAction.ABORT,
                reason="Could not parse LLM response",
            )

    def _parse_routing_response(self, response: str) -> Optional[str]:
        """Parse routing response."""
        response = response.strip().strip('"').strip("'")
        if response.upper() == "DONE":
            return None
        if response in self._teams:
            return response
        # Try to find team_id in response
        for team_id in self._teams:
            if team_id in response:
                return team_id
        return None

    def _parse_arbitration_response(
        self,
        response: str,
        team_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Parse arbitration response."""
        try:
            data = json.loads(response)
            if data.get("synthesized"):
                return data.get("output", {})
            chosen = data.get("chosen_team")
            if chosen and chosen in team_results:
                result = team_results[chosen]
                return result.output if hasattr(result, "output") else {}
        except json.JSONDecodeError:
            pass
        return self._merge_outputs(team_results)

    # === Properties ===

    @property
    def role(self) -> GroupRole:
        """Current orchestrator role."""
        return self._role

    @property
    def last_report(self) -> Optional[GroupReport]:
        """Last execution report."""
        return self._last_report

    def __repr__(self) -> str:
        return (
            f"<GroupOrchestrator id='{self.group_id}' "
            f"teams={len(self._teams)} "
            f"role={self._role.value}>"
        )
```

---

## 3. LLMTeam additions

Добавить в `/team/team.py`:

```python
# В начале файла, добавить импорты:
from llmteam.orchestration.models import (
    GroupContext,
    TeamRole,
    EscalationRequest,
    EscalationResponse,
)

# В класс LLMTeam добавить:

class LLMTeam:
    def __init__(self, ...):
        # ... existing code ...
        
        # Group context (None если не в группе)
        self._group_context: Optional[GroupContext] = None
    
    # === Group Integration (RFC-008) ===
    
    def _join_group(self, context: GroupContext) -> None:
        """
        INTERNAL: Called by GroupOrchestrator when adding to group.
        
        Do not call directly! Use GroupOrchestrator.add_team().
        """
        self._group_context = context
        
        # Notify TeamOrchestrator
        if self._orchestrator:
            self._orchestrator._set_group_context(context)
        
        # Emit event
        self._emit_event("group.joined", {
            "group_id": context.group_id,
            "role": context.team_role.value,
        })
    
    def _leave_group(self) -> None:
        """
        INTERNAL: Called by GroupOrchestrator when removing from group.
        """
        if self._group_context:
            group_id = self._group_context.group_id
            self._group_context = None
            
            if self._orchestrator:
                self._orchestrator._clear_group_context()
            
            self._emit_event("group.left", {"group_id": group_id})
    
    @property
    def is_in_group(self) -> bool:
        """Is team in a group?"""
        return self._group_context is not None
    
    @property
    def group_id(self) -> Optional[str]:
        """Group ID (if in group)."""
        return self._group_context.group_id if self._group_context else None
    
    @property
    def group_role(self) -> Optional[TeamRole]:
        """Role in group (if in group)."""
        return self._group_context.team_role if self._group_context else None
    
    async def escalate_to_group(
        self,
        reason: str,
        context: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
        source_agent: Optional[str] = None,
    ) -> EscalationResponse:
        """
        Escalate to group orchestrator.
        
        Args:
            reason: Escalation reason
            context: Additional context
            error: Exception (if any)
            source_agent: Source agent ID (if from agent)
        
        Returns:
            EscalationResponse from GroupOrchestrator
        
        Raises:
            RuntimeError: If team is not in group or escalation not allowed
        """
        if not self._group_context:
            raise RuntimeError(
                f"Team '{self.team_id}' is not in a group. Cannot escalate."
            )
        
        if not self._group_context.can_escalate:
            raise RuntimeError(
                f"Team '{self.team_id}' is not allowed to escalate."
            )
        
        request = EscalationRequest(
            source_team_id=self.team_id,
            source_agent_id=source_agent,
            reason=reason,
            context=context or {},
            error=error,
        )
        
        if self._group_context.on_escalation:
            return await self._group_context.on_escalation(request)
        
        return await self._group_context.group_orchestrator._handle_escalation(request)
    
    async def request_team(
        self,
        target_team_id: str,
        task: Dict[str, Any],
    ) -> Any:
        """
        Request execution from another team in the group.
        
        Only available for LEADER and SPECIALIST roles.
        
        Args:
            target_team_id: Target team ID
            task: Task to execute
        
        Returns:
            Result from target team
        
        Raises:
            RuntimeError: If not in group
            PermissionError: If not allowed to request teams
            ValueError: If target team not visible
        """
        if not self._group_context:
            raise RuntimeError(f"Team '{self.team_id}' is not in a group")
        
        if not self._group_context.can_request_team:
            raise PermissionError(
                f"Team '{self.team_id}' (role={self._group_context.team_role.value}) "
                f"is not allowed to request other teams"
            )
        
        if target_team_id not in self._group_context.visible_teams:
            raise ValueError(f"Team '{target_team_id}' is not visible")
        
        return await self._group_context.group_orchestrator.route_to_team(
            source_team_id=self.team_id,
            target_team_id=target_team_id,
            task=task,
        )
```

---

## 4. TeamOrchestrator additions

Добавить в `/agents/orchestrator.py`:

```python
# В TeamOrchestrator добавить:

class TeamOrchestrator:
    def __init__(self, team: "LLMTeam", config: Optional[OrchestratorConfig] = None):
        # ... existing code ...
        
        # Group context (RFC-008)
        self._group_context: Optional["GroupContext"] = None
    
    # === Group Integration (RFC-008) ===
    
    def _set_group_context(self, context: "GroupContext") -> None:
        """
        INTERNAL: Set group context.
        
        Called from LLMTeam._join_group().
        """
        self._group_context = context
        self._scope = OrchestratorScope.GROUP
    
    def _clear_group_context(self) -> None:
        """
        INTERNAL: Clear group context.
        
        Called from LLMTeam._leave_group().
        """
        self._group_context = None
        self._scope = OrchestratorScope.TEAM
    
    @property
    def is_in_group(self) -> bool:
        """Is team in a group?"""
        return self._group_context is not None
    
    @property
    def group_id(self) -> Optional[str]:
        """Group ID (if in group)."""
        return self._group_context.group_id if self._group_context else None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary including group info."""
        summary = {
            "run_id": self._current_run_id,
            "agents_executed": len(self._reports),
            "agents_succeeded": sum(1 for r in self._reports if r.success),
            "agents_failed": sum(1 for r in self._reports if not r.success),
            "total_duration_ms": sum(r.duration_ms for r in self._reports),
            "total_tokens_used": sum(r.tokens_used for r in self._reports),
        }
        
        # Add group info if in group
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

## 5. Updated `__init__.py` exports

В `/orchestration/__init__.py`:

```python
"""
Orchestration module.

RFC-004: GroupOrchestrator
RFC-008: Group Architecture Unification
"""

from llmteam.orchestration.models import (
    GroupRole,
    TeamRole,
    GroupContext,
    EscalationRequest,
    EscalationResponse,
    EscalationAction,
    TeamReport,
    GroupReport,
    GroupResult,
)

from llmteam.orchestration.group import GroupOrchestrator

__all__ = [
    # Roles
    "GroupRole",
    "TeamRole",
    # Context
    "GroupContext",
    # Escalation
    "EscalationRequest",
    "EscalationResponse",
    "EscalationAction",
    # Reports
    "TeamReport",
    "GroupReport",
    "GroupResult",
    # Orchestrator
    "GroupOrchestrator",
]
```

---

**Готово к внедрению!**
