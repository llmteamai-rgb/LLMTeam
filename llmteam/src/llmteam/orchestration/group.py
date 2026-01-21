"""
GroupOrchestrator for multi-team coordination.

RFC-004: Separate class for coordinating multiple LLMTeams.
RFC-009: Group Architecture Unification with bi-directional context.
"""

import asyncio
import json
import uuid
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from llmteam.orchestration.reports import TeamReport, GroupReport, GroupResult
from llmteam.orchestration.models import (
    TeamRole,
    GroupContext,
    EscalationRequest,
    EscalationResponse,
    EscalationAction,
)

if TYPE_CHECKING:
    from llmteam.team import LLMTeam
    from llmteam.team.result import RunResult


class GroupRole(Enum):
    """
    Roles for GroupOrchestrator.

    RFC-009: Extended with new coordination strategies.

    Passive roles (no LLM required):
    - REPORT_COLLECTOR: Passive report collection
    - COORDINATOR: Sequential with context passing

    Active roles (LLM required):
    - ROUTER: Dynamic LLM-based routing
    - AGGREGATOR: Parallel execution with aggregation
    - ARBITER: Conflict resolution
    """

    # === Passive (no LLM required) ===

    REPORT_COLLECTOR = "report_collector"
    """
    Default role. Passive report collection.

    Behavior:
    1. Executes all teams (sequential or parallel)
    2. Collects TeamReport from each team
    3. Aggregates into GroupReport
    4. Returns results

    Does not make decisions. Does not change flow.
    Requires LLM: NO
    """

    COORDINATOR = "coordinator"
    """
    Coordination with context passing.

    Behavior:
    1. Executes teams in order
    2. Passes output of team N as input to team N+1
    3. Manages data flow between teams
    4. Handles escalations (retry/skip/abort)

    Follows given plan. Does NOT make runtime decisions.
    Requires LLM: NO (optional for escalations)
    """

    # === Active (LLM required) ===

    ROUTER = "router"
    """
    Dynamic routing between teams.

    Behavior:
    1. On each step decides which team to call
    2. Analyzes result and decides to continue or finish
    3. Can loop (controlled by max_iterations)
    4. Maintains conversation history

    Requires LLM: YES
    """

    AGGREGATOR = "aggregator"
    """
    Parallel execution and aggregation.

    Behavior:
    1. Runs teams in parallel
    2. Waits for all results
    3. Aggregates into single output
    4. Strategies: merge, reduce, vote, best_of

    Requires LLM: Optional (for vote/best_of)
    """

    ARBITER = "arbiter"
    """
    Conflict arbitration and decision making.

    Behavior:
    1. Receives results from multiple teams
    2. Analyzes conflicts/discrepancies
    3. Decides which result to use
    4. Can request re-execution

    Requires LLM: YES
    """


class GroupOrchestrator:
    """
    Orchestrator for a group of teams.

    RFC-004: Coordinates multiple LLMTeams and collects reports.
    RFC-009: Unified group coordination with bi-directional context.

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
            role: Orchestrator role (default: REPORT_COLLECTOR).
            model: LLM model for roles requiring it (ROUTER, ARBITER).
            max_iterations: Max routing iterations (for ROUTER).
            max_escalation_depth: Max escalation depth to prevent loops.
        """
        self.group_id = group_id or f"group_{uuid.uuid4().hex[:8]}"
        self._role = role
        self._model = model
        self._max_iterations = max_iterations
        self._max_escalation_depth = max_escalation_depth

        # Teams (RFC-009: with roles)
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

    # === Team Management (RFC-009: with roles) ===

    def add_team(
        self,
        team: "LLMTeam",
        role: TeamRole = TeamRole.MEMBER,
    ) -> None:
        """
        Add a team to the group.

        RFC-009: Teams receive GroupContext on addition.

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

        # RFC-009: Create context for team
        context = self._create_team_context(team.team_id, role)

        # RFC-009: Notify team about joining group
        if hasattr(team, "_join_group"):
            team._join_group(context)

        # Update contexts of other teams
        self._update_team_contexts()

    def remove_team(self, team_id: str) -> bool:
        """
        Remove a team from the group.

        RFC-009: Team is notified about leaving.

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

        # RFC-009: Notify team about leaving
        if hasattr(team, "_leave_group"):
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
        """Leader team (RFC-009)."""
        return self._teams.get(self._leader_id) if self._leader_id else None

    # === Execution (RFC-009: role-based strategies) ===

    async def execute(
        self,
        input_data: Dict[str, Any],
        run_id: Optional[str] = None,
        parallel: bool = False,
    ) -> GroupResult:
        """
        Execute all teams according to role.

        RFC-009: Different execution strategies based on role.

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
                result.report.escalations_handled = self._escalation_count
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

    # === Escalation Handling (RFC-009) ===

    async def _handle_escalation(
        self,
        request: EscalationRequest,
    ) -> EscalationResponse:
        """
        Handle escalation from a team.

        RFC-009: Called via GroupContext.on_escalation.
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

        RFC-009: Called by LLMTeam.request_team().
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
        """Create TeamReport from team execution result."""
        agents_executed = []
        agent_reports = []

        orchestrator = team.get_orchestrator()
        if orchestrator and hasattr(orchestrator, "_reports"):
            agent_reports = [r.to_dict() for r in orchestrator._reports]
            agents_executed = [r.agent_role for r in orchestrator._reports]

        if not agents_executed and hasattr(result, "agents_called"):
            agents_executed = result.agents_called or []

        report = TeamReport(
            team_id=team_id,
            run_id=getattr(result, "run_id", "") or "",
            success=result.success if hasattr(result, "success") else True,
            duration_ms=getattr(result, "duration_ms", 0) or 0,
            agents_executed=agents_executed,
            agent_reports=agent_reports,
            output_summary=str(result.output)[:200] if hasattr(result, "output") else "",
            errors=[result.error] if hasattr(result, "error") and result.error else [],
        )

        # RFC-009: Add team role
        report.team_role = self._team_roles.get(team_id, TeamRole.MEMBER).value

        return report

    def _create_group_report(
        self,
        team_reports: List[TeamReport],
    ) -> GroupReport:
        """Create aggregated group report."""
        succeeded = sum(1 for r in team_reports if r.success)
        failed = len(team_reports) - succeeded

        summary = f"Executed {len(team_reports)} teams: {succeeded} succeeded, {failed} failed"

        report = GroupReport(
            group_id=self.group_id,
            role=self._role.value,
            teams_count=len(team_reports),
            teams_succeeded=succeeded,
            teams_failed=failed,
            total_duration_ms=0,  # Set later
            team_reports=team_reports,
            summary=summary,
        )

        # RFC-009: Add run_id and escalations
        report.run_id = self._current_run_id or ""
        report.escalations_handled = self._escalation_count

        return report

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
