"""
Agent registry for LLMTeam.

Provides specialized registry for managing agents within a team.

Note: In v4.0.0, agents are created through LLMTeam.add_agent().
This registry is used internally by LLMTeam for agent management.
"""

from typing import TYPE_CHECKING, List, Optional

from llmteam.agents.base import BaseAgent
from llmteam.agents.types import AgentType
from llmteam.registry.base import BaseRegistry

if TYPE_CHECKING:
    from llmteam.team import LLMTeam


class AgentRegistry(BaseRegistry[BaseAgent]):
    """
    Specialized registry for agents.

    Manages agents within a team. Used internally by LLMTeam.

    Note:
        In v4.0.0, agents should be created through LLMTeam.add_agent()
        rather than directly through this registry.

    Example:
        # Typical usage is through LLMTeam:
        team = LLMTeam(team_id="my_team")
        team.add_agent({"type": "llm", "role": "writer", "prompt": "..."})

        # Internal registry access:
        agent = team.get_agent("writer")
    """

    def __init__(
        self,
        team: Optional["LLMTeam"] = None,
        allow_overwrite: bool = False,
    ):
        """
        Initialize agent registry.

        Args:
            team: The team that owns this registry.
            allow_overwrite: Whether to allow overwriting agents.
        """
        super().__init__(name="agent_registry", allow_overwrite=allow_overwrite)
        self._team = team

    @property
    def team(self) -> Optional["LLMTeam"]:
        """The team that owns this registry."""
        return self._team

    def register(self, key: str, agent: BaseAgent) -> None:
        """
        Register an agent.

        Args:
            key: Unique key for the agent (usually agent_id).
            agent: Agent to register.

        Raises:
            TypeError: If agent is not a BaseAgent instance.
        """
        if not isinstance(agent, BaseAgent):
            raise TypeError(f"Expected BaseAgent, got {type(agent).__name__}")

        super().register(key, agent)

    def register_agent(self, agent: BaseAgent) -> None:
        """
        Register an agent using its agent_id as key.

        Args:
            agent: Agent to register.
        """
        self.register(agent.agent_id, agent)

    def unregister(self, key: str) -> BaseAgent:
        """
        Unregister an agent.

        Args:
            key: Key of agent to remove.

        Returns:
            The removed agent.
        """
        return super().unregister(key)

    def get_by_role(self, role: str) -> Optional[BaseAgent]:
        """
        Find an agent by role.

        Args:
            role: Agent role to search for.

        Returns:
            Agent with matching role, or None.
        """
        for agent in self._items.values():
            if agent.role == role:
                return agent
        return None

    def list_by_type(self, agent_type: AgentType) -> List[BaseAgent]:
        """
        List agents of a specific type.

        Args:
            agent_type: AgentType to filter by.

        Returns:
            List of agents of that type.
        """
        return [a for a in self._items.values() if a.agent_type == agent_type]

    def get_agent_ids(self) -> List[str]:
        """
        Get list of all agent IDs.

        Returns:
            List of agent IDs.
        """
        return [a.agent_id for a in self._items.values()]

    def get_roles(self) -> List[str]:
        """
        Get list of all agent roles.

        Returns:
            List of agent roles.
        """
        return [a.role for a in self._items.values()]
