"""
Base agent class.

Internal abstract base class - NOT exported in public API.
Agents are created only via LLMTeam.add_agent().

RFC-007: Agent Encapsulation
- Direct agent.process() is FORBIDDEN - raises RuntimeError
- Use team.run() instead for proper logging, reporting, and flow control
- _process() is internal, called only by TeamOrchestrator
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

from llmteam.agents.types import AgentType, AgentStatus
from llmteam.agents.config import AgentConfig
from llmteam.agents.state import AgentState
from llmteam.agents.result import AgentResult
from llmteam.agents.report import AgentReport
from llmteam.agents.retry import AgentRetryExecutor, RetryMetrics
from llmteam.tools import ToolExecutor, ToolDefinition

if TYPE_CHECKING:
    from llmteam.team import LLMTeam


class BaseAgent(ABC):
    """
    Base agent class.

    INTERNAL CLASS - not exported in public API.
    Agents are created only via LLMTeam.add_agent().

    RFC-007 Contract:
    - agent_type: type of agent (LLM, RAG, KAG)
    - _team: required reference to team
    - _execute(): abstract method for subclass implementation
    - _process(): internal execution (orchestrator only)
    - process(): FORBIDDEN - raises RuntimeError
    - __call__(): FORBIDDEN - raises RuntimeError
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

        # RFC-012: Per-agent retry executor
        self._retry_executor: Optional[AgentRetryExecutor] = None
        if config.retry_policy or config.circuit_breaker:
            self._retry_executor = AgentRetryExecutor(
                agent_id=config.id or config.role,
                retry_policy=config.retry_policy,
                circuit_breaker_policy=config.circuit_breaker,
            )

        # RFC-013: Per-agent tool executor
        self._tool_executor: Optional[ToolExecutor] = None
        if config.tools:
            self._tool_executor = ToolExecutor(tools=config.tools)

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

    @property
    def retry_executor(self) -> Optional[AgentRetryExecutor]:
        """Per-agent retry executor (RFC-012). None if no policy configured."""
        return self._retry_executor

    @property
    def tool_executor(self) -> Optional[ToolExecutor]:
        """Per-agent tool executor (RFC-013). None if no tools configured."""
        return self._tool_executor

    # Abstract Methods

    @abstractmethod
    async def _execute(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> AgentResult:
        """
        INTERNAL: Subclass implementation.

        Override this method in subclasses to implement agent logic.
        Do NOT call directly - use team.run() instead.

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

    # Forbidden Methods (RFC-007)

    async def process(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """
        FORBIDDEN: Direct agent.process() is not allowed.

        Use team.run() instead to ensure proper logging, reporting,
        and flow control through the orchestrator.

        Raises:
            RuntimeError: Always. Use team.run() instead.
        """
        raise RuntimeError(
            f"Direct call to {self.__class__.__name__}.process() is forbidden. "
            f"Use team.run() instead to ensure proper logging, reporting, and flow control."
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        FORBIDDEN: Direct agent call is not allowed.

        Raises:
            RuntimeError: Always. Use team.run() instead.
        """
        raise RuntimeError(
            f"Direct call to {self.__class__.__name__}() is forbidden. "
            f"Use team.run() instead."
        )

    # Internal Execution (called by TeamOrchestrator only)

    async def _process(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
        run_id: str,
    ) -> AgentResult:
        """
        INTERNAL: Execution wrapper with lifecycle hooks and retry.

        Called ONLY by TeamOrchestrator/TeamRunner, never directly.
        Do NOT call this method from user code - use team.run() instead.

        RFC-012: If retry_policy or circuit_breaker is configured,
        wraps _execute() with retry/circuit breaker logic.
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

            # Execute with retry (RFC-012) or direct
            retry_metrics: Optional[RetryMetrics] = None

            if self._retry_executor:
                result, retry_metrics = await self._retry_executor.execute(
                    self._execute, input_data, context
                )
            else:
                result = await self._execute(input_data, context)

            result.agent_id = self.agent_id
            result.agent_type = self.agent_type

            # Attach retry metrics if available
            if retry_metrics:
                if result.context_payload is None:
                    result.context_payload = {}
                result.context_payload["retry_metrics"] = retry_metrics.to_dict()

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

            # Build error result with retry metrics if available
            error_result = AgentResult(
                output=None,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                success=False,
                error=str(e),
            )

            # Attach circuit breaker info if relevant
            if self._retry_executor:
                cb_metrics = self._retry_executor.get_circuit_breaker_metrics()
                if cb_metrics:
                    error_result.context_payload = {
                        "circuit_breaker": cb_metrics,
                    }
                # Flag for escalation if circuit breaker opened
                from llmteam.clients.circuit_breaker import CircuitBreakerOpen
                if isinstance(e, CircuitBreakerOpen):
                    error_result.should_escalate = True
                    error_result.escalation_reason = (
                        f"Circuit breaker open for agent '{self.agent_id}': {e}"
                    )

            # Report failure to orchestrator
            await self._report(
                started_at=started_at,
                input_data=input_data,
                output=None,
                success=False,
                error=e,
            )

            return error_result

    # Backward compatibility alias (deprecated)
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
        run_id: str,
    ) -> AgentResult:
        """
        DEPRECATED: Use _process() instead.

        This method exists for backward compatibility and will be removed.
        """
        import warnings
        warnings.warn(
            "BaseAgent.execute() is deprecated, internal code should use _process()",
            DeprecationWarning,
            stacklevel=2
        )
        return await self._process(input_data, context, run_id)

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
