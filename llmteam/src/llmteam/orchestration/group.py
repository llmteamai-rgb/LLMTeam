"""
GroupOrchestrator for multi-team coordination.

RFC-004: Separate class for coordinating multiple LLMTeams.
"""

import asyncio
import uuid
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from llmteam.orchestration.reports import TeamReport, GroupReport, GroupResult

if TYPE_CHECKING:
    from llmteam.team import LLMTeam
    from llmteam.team.result import RunResult


class GroupRole(Enum):
    """
    Roles for GroupOrchestrator.

    MVP: only REPORT_COLLECTOR.
    Future roles (P3+): COORDINATOR, AGGREGATOR, ARBITER.
    """

    REPORT_COLLECTOR = "report_collector"
    """
    Default role.

    Behavior:
    1. Executes all teams
    2. Collects TeamReport from each TeamOrchestrator
    3. Aggregates into GroupReport
    4. Returns result + reports

    Does not make decisions, does not change flow.
    """

    # Future roles:
    # COORDINATOR = "coordinator"
    # AGGREGATOR = "aggregator"
    # ARBITER = "arbiter"


class GroupOrchestrator:
    """
    Orchestrator for a group of teams.

    RFC-004: Coordinates multiple LLMTeams and collects reports.

    MVP: role REPORT_COLLECTOR - collects reports from TeamOrchestrators.

    Usage:
        orch = GroupOrchestrator(group_id="my_group")
        orch.add_team(team1)
        orch.add_team(team2)

        result = await orch.execute({"query": "..."})
        print(result.report.summary)
    """

    def __init__(
        self,
        group_id: Optional[str] = None,
        role: GroupRole = GroupRole.REPORT_COLLECTOR,
    ):
        """
        Initialize GroupOrchestrator.

        Args:
            group_id: Unique group identifier (auto-generated if None).
            role: Orchestrator role (default: REPORT_COLLECTOR).
        """
        self.group_id = group_id or f"group_{uuid.uuid4().hex[:8]}"
        self._role = role
        self._teams: Dict[str, "LLMTeam"] = {}
        self._last_report: Optional[GroupReport] = None

    # === Team Management ===

    def add_team(self, team: "LLMTeam") -> None:
        """
        Add a team to the group.

        Args:
            team: LLMTeam instance to add.
        """
        self._teams[team.team_id] = team

    def remove_team(self, team_id: str) -> bool:
        """
        Remove a team from the group.

        Args:
            team_id: ID of team to remove.

        Returns:
            True if team was removed, False if not found.
        """
        return self._teams.pop(team_id, None) is not None

    def list_teams(self) -> List[str]:
        """
        List team IDs in the group.

        Returns:
            List of team IDs.
        """
        return list(self._teams.keys())

    def get_team(self, team_id: str) -> Optional["LLMTeam"]:
        """
        Get a team by ID.

        Args:
            team_id: Team ID.

        Returns:
            LLMTeam or None if not found.
        """
        return self._teams.get(team_id)

    @property
    def teams_count(self) -> int:
        """Number of teams in the group."""
        return len(self._teams)

    # === Execution ===

    async def execute(
        self,
        input_data: Dict[str, Any],
        parallel: bool = False,
    ) -> GroupResult:
        """
        Execute all teams and collect reports.

        Args:
            input_data: Input data for all teams.
            parallel: Execute teams in parallel (default: sequential).

        Returns:
            GroupResult with results and reports.
        """
        start_time = datetime.utcnow()

        if parallel:
            team_results, team_reports, errors = await self._execute_parallel(input_data)
        else:
            team_results, team_reports, errors = await self._execute_sequential(input_data)

        # Create aggregated report
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        group_report = self._create_group_report(team_reports, duration_ms)
        self._last_report = group_report

        return GroupResult(
            success=len(errors) == 0,
            output=self._merge_outputs(team_results),
            team_results=team_results,
            report=group_report,
            errors=errors,
        )

    async def _execute_sequential(
        self,
        input_data: Dict[str, Any],
    ) -> Tuple[Dict[str, "RunResult"], List[TeamReport], List[str]]:
        """Execute teams sequentially."""
        team_results: Dict[str, Any] = {}
        team_reports: List[TeamReport] = []
        errors: List[str] = []

        for team_id, team in self._teams.items():
            try:
                result = await team.run(input_data)
                team_results[team_id] = result

                # Collect report from TeamOrchestrator
                report = self._create_team_report(team_id, team, result)
                team_reports.append(report)

            except Exception as e:
                errors.append(f"{team_id}: {str(e)}")

        return team_results, team_reports, errors

    async def _execute_parallel(
        self,
        input_data: Dict[str, Any],
    ) -> Tuple[Dict[str, "RunResult"], List[TeamReport], List[str]]:
        """Execute teams in parallel."""

        async def run_team(team_id: str, team: "LLMTeam") -> Tuple[str, Any, Optional[TeamReport]]:
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
                if report:
                    team_reports.append(report)

        return team_results, team_reports, errors

    def _create_team_report(
        self,
        team_id: str,
        team: "LLMTeam",
        result: "RunResult",
    ) -> TeamReport:
        """Create TeamReport from team execution result."""
        # Get agent reports from orchestrator if available
        agent_reports = []
        agents_executed = []

        orchestrator = team.get_orchestrator()
        if orchestrator and hasattr(orchestrator, "_reports"):
            agent_reports = [r.to_dict() for r in orchestrator._reports]
            agents_executed = [r.agent_role for r in orchestrator._reports]

        # If no orchestrator, try to get from result
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
        )

    def _create_group_report(
        self,
        team_reports: List[TeamReport],
        duration_ms: int,
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
            total_duration_ms=duration_ms,
            team_reports=team_reports,
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
        return f"<GroupOrchestrator id='{self.group_id}' teams={len(self._teams)} role={self._role.value}>"
