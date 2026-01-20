"""
Base agent class.

Internal abstract base class - NOT exported in public API.
Agents are created only via LLMTeam.add_agent().
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

from llmteam.agents.types import AgentType, AgentStatus
from llmteam.agents.config import AgentConfig
from llmteam.agents.state import AgentState
from llmteam.agents.result import AgentResult
from llmteam.agents.report import AgentReport

if TYPE_CHECKING:
    from llmteam.team import LLMTeam


class BaseAgent(ABC):
    """
    Base agent class.

    INTERNAL CLASS - not exported in public API.
    Agents are created only via LLMTeam.add_agent().

    Contract:
    - agent_type: type of agent (LLM, RAG, KAG)
    - _team: required reference to team
    - process(): main execution method
    """

    # Class Attributes (overridden in subclasses)
    agent_type: AgentType = AgentType.LLM

    # Instance Attributes
    agent_id: str
    role: str
    name: str
    description: str

    _team: "LLMTeam"  # Required!
    _config: AgentConfig
    _state: Optional[AgentState]  # Runtime state

    def __init__(self, team: "LLMTeam", config: AgentConfig):
        """
        Initialize agent.

        IMPORTANT: Constructor is not public. Called only from LLMTeam.

        Args:
            team: Owner team (required)
            config: Agent configuration

        Raises:
            TypeError: If team is not provided
        """
        if team is None:
            raise TypeError(
                f"{self.__class__.__name__} requires 'team' argument. "
                f"Use LLMTeam.add_agent() instead of direct instantiation."
            )

        self._team = team
        self._config = config
        self._state = None

        # Copy from config
        self.agent_id = config.id or config.role
        self.role = config.role
        self.name = config.name or config.role
        self.description = config.description

    # Properties

    @property
    def team(self) -> "LLMTeam":
        """Owner team (always exists)."""
        return self._team

    @property
    def config(self) -> AgentConfig:
        """Agent configuration."""
        return self._config

    @property
    def state(self) -> Optional[AgentState]:
        """Current runtime state."""
        return self._state

    @property
    def status(self) -> AgentStatus:
        """Current status."""
        return self._state.status if self._state else AgentStatus.IDLE

    # Abstract Methods

    @abstractmethod
    async def process(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> AgentResult:
        """
        Main execution method.

        Args:
            input_data: Input data (from team.run())
            context: Context from mailbox (results from other agents)

        Returns:
            AgentResult with execution result
        """
        ...

    # Lifecycle Hooks

    async def on_start(self, state: AgentState) -> None:
        """Hook: before process() execution."""
        pass

    async def on_complete(self, result: AgentResult) -> None:
        """Hook: after successful execution."""
        pass

    async def on_error(self, error: Exception) -> None:
        """Hook: on error."""
        pass

    # Execution Wrapper (called by TeamRunner)

    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
        run_id: str,
    ) -> AgentResult:
        """
        Execution wrapper with lifecycle hooks.

        Called by TeamRunner, not directly.
        """
        started_at = datetime.utcnow()

        # Create state
        self._state = AgentState(
            agent_id=self.agent_id,
            run_id=run_id,
            input_data=input_data,
            context=context,
        )
        self._state.mark_started()

        try:
            # Pre-hook
            await self.on_start(self._state)

            # Execute
            result = await self.process(input_data, context)
            result.agent_id = self.agent_id
            result.agent_type = self.agent_type

            # Update state
            self._state.mark_completed()
            self._state.tokens_used = result.tokens_used

            # Post-hook
            await self.on_complete(result)

            # Report to orchestrator
            await self._report(
                started_at=started_at,
                input_data=input_data,
                output=result.output,
                success=True,
                tokens_used=result.tokens_used,
            )

            return result

        except Exception as e:
            self._state.mark_failed(e)
            await self.on_error(e)

            # Report failure to orchestrator
            await self._report(
                started_at=started_at,
                input_data=input_data,
                output=None,
                success=False,
                error=e,
            )

            return AgentResult(
                output=None,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                success=False,
                error=str(e),
            )

    # Reporting (to orchestrator)

    async def _report(
        self,
        started_at: datetime,
        input_data: Dict[str, Any],
        output: Any,
        success: bool,
        error: Optional[Exception] = None,
        tokens_used: int = 0,
    ) -> None:
        """
        Send report to team orchestrator.

        Args:
            started_at: When execution started
            input_data: Input data
            output: Output from process()
            success: Whether execution succeeded
            error: Exception if failed
            tokens_used: Tokens consumed
        """
        # Get orchestrator from team
        orchestrator = self._team.get_orchestrator()
        if orchestrator is None:
            return  # No orchestrator to report to

        # Get model name if available
        model = getattr(self, "model", None)

        # Create report
        report = AgentReport.create(
            agent_id=self.agent_id,
            agent_role=self.role,
            agent_type=self.agent_type.value,
            started_at=started_at,
            input_data=input_data,
            output=output,
            success=success,
            error=error,
            tokens_used=tokens_used,
            model=model,
        )

        # Send to orchestrator
        orchestrator.receive_report(report)

    # Escalation

    async def escalate(self, reason: str, context: Optional[Dict] = None) -> Any:
        """
        Escalate to team.

        Agent -> Team Orchestrator -> Group Orchestrator (if exists)
        """
        return await self._team.escalate(
            source_agent=self.agent_id,
            reason=reason,
            context=context,
        )

    # Serialization

    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent configuration."""
        return {
            "type": self.agent_type.value,
            "id": self.agent_id,
            "role": self.role,
            "name": self.name,
            "description": self.description,
        }

    # Magic Methods

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id='{self.agent_id}' team='{self._team.team_id}'>"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BaseAgent):
            return (
                self.agent_id == other.agent_id
                and self._team.team_id == other._team.team_id
            )
        return False

    def __hash__(self) -> int:
        return hash((self.agent_id, self._team.team_id))
