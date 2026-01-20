"""
Team registry for LLMTeam.

Provides specialized registry for managing teams within a group.
"""

from typing import TYPE_CHECKING, List, Optional

from llmteam.registry.base import BaseRegistry

if TYPE_CHECKING:
    from llmteam.team import LLMTeam


class TeamRegistry(BaseRegistry["LLMTeam"]):
    """
    Specialized registry for teams.

    Manages teams within a group orchestrator.

    Example:
        registry = TeamRegistry()
        registry.register("triage", triage_team)
        registry.register("billing", billing_team)
        team = registry.get("triage")
    """

    def __init__(self, allow_overwrite: bool = False):
        """
        Initialize team registry.

        Args:
            allow_overwrite: Whether to allow overwriting teams.
        """
        super().__init__(name="team_registry", allow_overwrite=allow_overwrite)

    def register_team(self, team: "LLMTeam") -> None:
        """
        Register a team using its team_id as key.

        Args:
            team: Team to register.
        """
        self.register(team.team_id, team)

    def get_by_name(self, name: str) -> Optional["LLMTeam"]:
        """
        Find a team by name.

        Args:
            name: Team name to search for.

        Returns:
            Team with matching name, or None.
        """
        for team in self._items.values():
            if team.name == name:
                return team
        return None

    def list_healthy(self) -> List["LLMTeam"]:
        """
        List teams that are healthy (no recent critical escalations).

        Returns:
            List of healthy teams.
        """
        healthy = []
        for team in self._items.values():
            # Check if team has health_score method
            if hasattr(team, "health_score"):
                if team.health_score() >= 0.5:
                    healthy.append(team)
            else:
                # Assume healthy if no health tracking
                healthy.append(team)
        return healthy

    def list_available(self) -> List["LLMTeam"]:
        """
        List teams that are available for work.

        Returns:
            List of available teams.
        """
        available = []
        for team in self._items.values():
            # Check if team has is_available method
            if hasattr(team, "is_available"):
                if team.is_available():
                    available.append(team)
            else:
                # Assume available if no availability tracking
                available.append(team)
        return available

    def get_team_ids(self) -> List[str]:
        """
        Get list of all team IDs.

        Returns:
            List of team IDs.
        """
        return [t.team_id for t in self._items.values()]
