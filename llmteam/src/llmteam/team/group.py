"""
LLMGroup - Multi-team coordination.

Group orchestrator manages routing between teams.
"""

from typing import Any, Dict, List, Optional
import uuid

from llmteam.agents.presets import create_group_orchestrator_config
from llmteam.team.result import RunResult, RunStatus


class LLMGroup:
    """
    Group of LLMTeam instances.

    Provides coordination between multiple teams using a group orchestrator
    (which is just an LLMAgent with a specialized prompt).

    Example:
        support_team = LLMTeam(team_id="support", agents=[...])
        billing_team = LLMTeam(team_id="billing", agents=[...])

        group = support_team.create_group(
            group_id="customer_service",
            teams=[billing_team]
        )
        result = await group.run({"query": "I need help with my bill"})
    """

    def __init__(
        self,
        group_id: str,
        leader: "LLMTeam",
        teams: List["LLMTeam"],
        model: str = "gpt-4o-mini",
        max_iterations: int = 10,
    ):
        """
        Initialize group.

        Args:
            group_id: Unique group identifier
            leader: Leader team (also acts as default)
            teams: Other teams in the group
            model: Model for group orchestrator
            max_iterations: Max routing iterations
        """
        from llmteam.team.team import LLMTeam

        self.group_id = group_id
        self._leader = leader
        self._model = model
        self._max_iterations = max_iterations

        # All teams including leader
        self._teams: Dict[str, LLMTeam] = {leader.team_id: leader}
        for team in teams:
            self._teams[team.team_id] = team

        # Create orchestrator team
        self._orchestrator_team = self._create_orchestrator_team()

    def _create_orchestrator_team(self) -> "LLMTeam":
        """Create internal team with group orchestrator."""
        from llmteam.team.team import LLMTeam

        orch_config = create_group_orchestrator_config(
            available_teams=list(self._teams.keys()),
            model=self._model,
        )

        team = LLMTeam(
            team_id=f"{self.group_id}_orchestrator",
            agents=[orch_config],
        )

        return team

    @property
    def leader(self) -> "LLMTeam":
        """Get leader team."""
        return self._leader

    @property
    def teams(self) -> List["LLMTeam"]:
        """Get all teams."""
        return list(self._teams.values())

    def get_team(self, team_id: str) -> Optional["LLMTeam"]:
        """Get team by ID."""
        return self._teams.get(team_id)

    async def run(
        self,
        input_data: Dict[str, Any],
        run_id: Optional[str] = None,
    ) -> RunResult:
        """
        Execute group.

        Uses group orchestrator to route between teams.

        Args:
            input_data: Input data
            run_id: Optional run identifier

        Returns:
            RunResult from final team
        """
        run_id = run_id or str(uuid.uuid4())
        iterations = 0
        current_data = input_data.copy()
        teams_called = []
        all_outputs = {}

        while iterations < self._max_iterations:
            iterations += 1

            # Ask orchestrator which team should handle
            orch_input = {
                "task": current_data,
                "available_teams": list(self._teams.keys()),
                "capabilities": self._get_team_capabilities(),
                "history": teams_called,
            }

            orch_result = await self._orchestrator_team.run(orch_input, run_id=f"{run_id}_orch_{iterations}")

            if not orch_result.success:
                return RunResult(
                    success=False,
                    status=RunStatus.FAILED,
                    error=f"Orchestrator failed: {orch_result.error}",
                    agents_called=teams_called,
                    iterations=iterations,
                )

            # Parse orchestrator decision
            decision = self._parse_decision(orch_result.output)

            if not decision.get("should_continue", True):
                break

            next_team_id = decision.get("next_team")
            if not next_team_id or next_team_id not in self._teams:
                # Default to leader
                next_team_id = self._leader.team_id

            # Execute team
            team = self._teams[next_team_id]
            team_result = await team.run(current_data, run_id=f"{run_id}_{next_team_id}")

            teams_called.append(next_team_id)
            all_outputs[next_team_id] = team_result.output

            if not team_result.success:
                return RunResult(
                    success=False,
                    status=RunStatus.FAILED,
                    error=f"Team '{next_team_id}' failed: {team_result.error}",
                    output=all_outputs,
                    agents_called=teams_called,
                    iterations=iterations,
                )

            # Update data for next iteration
            current_data.update(team_result.output)

        return RunResult(
            success=True,
            status=RunStatus.COMPLETED,
            output=all_outputs,
            final_output=current_data,
            agents_called=teams_called,
            iterations=iterations,
        )

    def _get_team_capabilities(self) -> Dict[str, str]:
        """Get capabilities description for each team."""
        capabilities = {}
        for team_id, team in self._teams.items():
            agent_types = [a.agent_type.value for a in team.list_agents()]
            agent_roles = [a.role for a in team.list_agents()]
            capabilities[team_id] = f"Agents: {agent_roles}, Types: {set(agent_types)}"
        return capabilities

    def _parse_decision(self, output: Any) -> Dict[str, Any]:
        """Parse orchestrator decision from output."""
        if isinstance(output, dict):
            return output

        # Try to parse JSON from string
        if isinstance(output, str):
            import json

            try:
                # Find JSON in output
                start = output.find("{")
                end = output.rfind("}") + 1
                if start >= 0 and end > start:
                    return json.loads(output[start:end])
            except json.JSONDecodeError:
                pass

        # Default: continue with leader
        return {"next_team": self._leader.team_id, "should_continue": True}

    def to_config(self) -> Dict[str, Any]:
        """Export group configuration."""
        return {
            "group_id": self.group_id,
            "leader": self._leader.team_id,
            "teams": [t.to_config() for t in self._teams.values()],
            "model": self._model,
            "max_iterations": self._max_iterations,
        }

    def __repr__(self) -> str:
        return f"<LLMGroup id='{self.group_id}' teams={len(self._teams)}>"

    def __len__(self) -> int:
        return len(self._teams)
