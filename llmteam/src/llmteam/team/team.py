"""
LLMTeam - Main team container.

Orchestrates AI agents using ExecutionEngine internally (if available).
When engine module is not installed, only ROUTER mode works.
"""

from typing import (
    TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, List, Optional, Union,
)
import uuid

if TYPE_CHECKING:
    from llmteam.configuration import ConfigurationSession
    from llmteam.quality import CostEstimate
    from llmteam.orchestration.models import GroupContext, EscalationResponse
    from llmteam.team.lifecycle import TeamLifecycle, TeamState

from llmteam.agents.factory import AgentFactory
from llmteam.agents.config import AgentConfig
from llmteam.agents.base import BaseAgent
from llmteam.agents.orchestrator import (
    TeamOrchestrator,
    OrchestratorConfig,
    OrchestratorMode,
)
from llmteam.team.result import RunResult, RunStatus, ContextMode
from llmteam.team.snapshot import TeamSnapshot
from llmteam.quality import QualityManager
from llmteam.cost import CostTracker, Budget, BudgetManager, BudgetStatus

# Check if engine module is available
try:
    from llmteam.team.converters import build_segment, result_from_segment_result
    _ENGINE_AVAILABLE = True
except ImportError:
    _ENGINE_AVAILABLE = False
    build_segment = None  # type: ignore
    result_from_segment_result = None  # type: ignore


class LLMTeam:
    """
    Team of AI agents.

    Supports only three agent types: LLM, RAG, KAG.
    All external logic should be executed outside LLMTeam.

    Example:
        team = LLMTeam(
            team_id="content",
            agents=[
                {"type": "rag", "role": "retriever", "collection": "docs"},
                {"type": "llm", "role": "writer", "prompt": "Write about: {query}"}
            ]
        )
        result = await team.run({"query": "AI trends"})
    """

    def __init__(
        self,
        team_id: str,
        agents: Optional[List[Dict]] = None,
        flow: Union[str, Dict] = "sequential",
        model: str = "gpt-4o-mini",
        context_mode: ContextMode = ContextMode.SHARED,
        orchestration: bool = False,
        orchestrator: Optional[OrchestratorConfig] = None,
        timeout: Optional[int] = None,
        quality: Union[int, str] = 50,
        max_cost_per_run: Optional[float] = None,
        enforce_lifecycle: bool = False,
        **kwargs,
    ):
        """
        Initialize team.

        Args:
            team_id: Unique team identifier
            agents: List of agent configurations
            flow: Execution flow ("sequential", "a -> b -> c", or DAG dict)
            model: Default model for LLM agents
            context_mode: Context sharing mode (SHARED/NOT_SHARED)
            orchestration: DEPRECATED. Use orchestrator=OrchestratorConfig(mode=ACTIVE)
            orchestrator: Orchestrator configuration (PASSIVE by default)
            timeout: Execution timeout in seconds
            quality: Quality slider 0-100 or preset name (RFC-008)
                     Presets: "draft"=20, "economy"=30, "balanced"=50,
                     "production"=75, "best"=95, "auto"=adaptive
            max_cost_per_run: Optional cost limit per run in USD (safety)
            enforce_lifecycle: Enable lifecycle state enforcement (RFC-014).
                             When True, team must be configured and marked
                             ready before running.
        """
        self.team_id = team_id
        self._flow = flow
        self._model = model
        self._context_mode = context_mode
        self._timeout = timeout
        self._max_cost_per_run = max_cost_per_run

        # Quality management (RFC-008)
        self._quality_manager = QualityManager(quality)

        # RFC-010: Cost tracking & budget management
        self._cost_tracker = CostTracker()
        self._budget_manager: Optional[BudgetManager] = None
        if max_cost_per_run:
            self._budget_manager = BudgetManager(
                Budget(max_cost=max_cost_per_run)
            )

        # RFC-014: Lifecycle enforcement (opt-in)
        self._lifecycle: Optional["TeamLifecycle"] = None
        if enforce_lifecycle:
            from llmteam.team.lifecycle import TeamLifecycle
            self._lifecycle = TeamLifecycle()

        # Internal state
        self._agents: Dict[str, BaseAgent] = {}  # Only working agents
        self._orchestrator: Optional[TeamOrchestrator] = None  # Separate field
        self._runner = None  # Lazy init
        self._runtime = None
        self._current_run_id: Optional[str] = None
        self._event_callbacks: Dict[str, List[Callable]] = {}
        self._mailbox: Dict[str, Any] = {}

        # RFC-009: Group context (None if not in a group)
        self._group_context: Optional["GroupContext"] = None

        # Create agents from config
        if agents:
            for config in agents:
                self.add_agent(config)

        # Initialize orchestrator
        # Always present (PASSIVE by default), separate from agents
        if orchestrator:
            self._orchestrator = TeamOrchestrator(team=self, config=orchestrator)
        elif orchestration or flow == "adaptive":
            # Backward compatibility: orchestration=True → ACTIVE mode
            active_config = OrchestratorConfig(mode=OrchestratorMode.ACTIVE)
            self._orchestrator = TeamOrchestrator(team=self, config=active_config)
        else:
            # Default: PASSIVE mode (SUPERVISOR + REPORTER)
            self._orchestrator = TeamOrchestrator(team=self)

    # Agent Management

    def add_agent(self, config: Union[Dict, AgentConfig]) -> BaseAgent:
        """
        Create and add agent from configuration.

        Args:
            config: Agent configuration dict or AgentConfig

        Returns:
            Created agent

        Raises:
            ValueError: If agent with same ID already exists
            ValueError: If agent type is not supported
            ValueError: If role starts with "_" (reserved for internal use)
        """
        # Check for reserved role prefix
        role = config.get("role", "") if isinstance(config, dict) else config.role
        if role.startswith("_"):
            raise ValueError(
                f"Role '{role}' is reserved (starts with '_'). "
                f"Use orchestrator parameter instead."
            )

        agent = AgentFactory.create(team=self, config=config)

        if agent.agent_id in self._agents:
            raise ValueError(f"Agent '{agent.agent_id}' already exists in team")

        self._agents[agent.agent_id] = agent
        return agent

    def add_llm_agent(
        self,
        role: str,
        prompt: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> BaseAgent:
        """Shortcut to add LLM agent."""
        config = {
            "type": "llm",
            "role": role,
            "prompt": prompt,
            "model": model or self._model,
            **kwargs,
        }
        return self.add_agent(config)

    def add_rag_agent(
        self,
        role: str = "rag",
        collection: str = "default",
        **kwargs,
    ) -> BaseAgent:
        """Shortcut to add RAG agent."""
        config = {
            "type": "rag",
            "role": role,
            "collection": collection,
            **kwargs,
        }
        return self.add_agent(config)

    def add_kag_agent(
        self,
        role: str = "kag",
        **kwargs,
    ) -> BaseAgent:
        """Shortcut to add KAG agent."""
        config = {
            "type": "kag",
            "role": role,
            **kwargs,
        }
        return self.add_agent(config)

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get working agent by ID (excludes orchestrator)."""
        return self._agents.get(agent_id)

    def list_agents(self) -> List[BaseAgent]:
        """List working agents (excludes orchestrator)."""
        return list(self._agents.values())

    def get_orchestrator(self) -> Optional[TeamOrchestrator]:
        """
        Get team orchestrator.

        Returns:
            TeamOrchestrator instance (always present)
        """
        return self._orchestrator

    @property
    def is_router_mode(self) -> bool:
        """Check if orchestrator is in ROUTER mode."""
        if self._orchestrator is None:
            return False
        return self._orchestrator.is_router

    # Quality Management (RFC-008)

    @property
    def quality(self) -> int:
        """Get current quality level (0-100)."""
        return self._quality_manager.quality

    @quality.setter
    def quality(self, value: Union[int, str]) -> None:
        """Set quality level (0-100 or preset name)."""
        self._quality_manager.quality = value

    def get_quality_manager(self) -> QualityManager:
        """Get the QualityManager instance."""
        return self._quality_manager

    # Cost Tracking (RFC-010)

    @property
    def cost_tracker(self) -> CostTracker:
        """Get the CostTracker instance (RFC-010)."""
        return self._cost_tracker

    @property
    def budget_manager(self) -> Optional[BudgetManager]:
        """Get the BudgetManager instance if configured (RFC-010)."""
        return self._budget_manager

    async def estimate_cost(
        self,
        input_data: Optional[Dict[str, Any]] = None,
        complexity: str = "medium",
    ) -> "CostEstimate":
        """
        Estimate cost for a run (RFC-008).

        Args:
            input_data: Optional input data (for future token estimation)
            complexity: Task complexity ("simple", "medium", "complex")

        Returns:
            CostEstimate with min/max range

        Example:
            estimate = await team.estimate_cost()
            print(f"Estimated cost: ${estimate.min:.2f} - ${estimate.max:.2f}")
        """
        from llmteam.quality import CostEstimator

        estimator = CostEstimator()

        # If we have agents, use detailed estimation
        if self._agents:
            agents_config = [
                {
                    "role": agent.role,
                    "model": getattr(agent, "model", self._model),
                }
                for agent in self._agents.values()
            ]
            return estimator.estimate_detailed(
                quality=self.quality,
                agents=agents_config,
            )

        # Otherwise use simple estimation
        return estimator.estimate(
            quality=self.quality,
            complexity=complexity,
        )

    # Lifecycle (RFC-014)

    @property
    def lifecycle(self) -> Optional["TeamLifecycle"]:
        """Get the lifecycle manager (RFC-014). None if not enforced."""
        return self._lifecycle

    @property
    def state(self) -> Optional[str]:
        """Current lifecycle state (RFC-014). None if lifecycle not enforced."""
        if self._lifecycle:
            return self._lifecycle.state.value
        return None

    def mark_configuring(self) -> None:
        """
        Transition to CONFIGURING state (RFC-014).

        Raises:
            LifecycleError: If transition is invalid
            RuntimeError: If lifecycle is not enforced
        """
        if not self._lifecycle:
            raise RuntimeError("Lifecycle not enforced. Use enforce_lifecycle=True.")
        from llmteam.team.lifecycle import TeamState
        self._lifecycle.transition_to(TeamState.CONFIGURING)

    def mark_ready(self) -> None:
        """
        Transition to READY state (RFC-014).

        Team is ready to run after configuration.

        Raises:
            LifecycleError: If transition is invalid
            RuntimeError: If lifecycle is not enforced
        """
        if not self._lifecycle:
            raise RuntimeError("Lifecycle not enforced. Use enforce_lifecycle=True.")
        from llmteam.team.lifecycle import TeamState
        self._lifecycle.transition_to(TeamState.READY)

    # Execution

    async def run(
        self,
        input_data: Dict[str, Any],
        run_id: Optional[str] = None,
        quality: Optional[Union[int, str]] = None,
        importance: Optional[str] = None,
    ) -> RunResult:
        """
        Execute team.

        In PASSIVE mode: Converts agents to SegmentDefinition, delegates to SegmentRunner.
        In ROUTER mode: Orchestrator decides which agent runs next.

        Args:
            input_data: Input data for execution
            run_id: Optional run identifier
            quality: Override quality for this run (0-100 or preset) (RFC-008)
            importance: Task importance ("high", "medium", "low") adjusts quality

        Returns:
            RunResult with output and metadata
        """
        from datetime import datetime

        run_id = run_id or str(uuid.uuid4())
        self._current_run_id = run_id
        started_at = datetime.utcnow()

        # Determine effective quality (RFC-008)
        effective_quality = self.quality
        if quality is not None:
            effective_quality = QualityManager(quality).quality
        elif importance is not None:
            effective_quality = self._quality_manager.with_importance(importance)

        if not self._agents:
            return RunResult(
                success=False,
                status=RunStatus.FAILED,
                error="No agents in team",
            )

        # RFC-014: Lifecycle enforcement
        if self._lifecycle:
            from llmteam.team.lifecycle import TeamState, LifecycleError
            try:
                self._lifecycle.ensure_ready()
                self._lifecycle.transition_to(TeamState.RUNNING)
            except LifecycleError as e:
                return RunResult(
                    success=False,
                    status=RunStatus.FAILED,
                    error=str(e),
                )

        # Start orchestrator tracking
        if self._orchestrator:
            self._orchestrator.start_run(run_id)

        # RFC-010: Start cost tracking
        self._cost_tracker.start_run(run_id, self.team_id)

        try:
            # Check if ROUTER mode is enabled
            if self.is_router_mode:
                result = await self._run_router_mode(input_data, run_id, started_at)
            else:
                result = await self._run_canvas_mode(input_data, run_id, started_at)

            # Add orchestrator report
            if self._orchestrator:
                result.report = self._orchestrator.generate_report()
                result.summary = self._orchestrator.get_summary()

            # RFC-010: Attach cost info
            run_cost = self._cost_tracker.end_run()
            result.tokens_used = run_cost.total_tokens
            if run_cost.total_cost > 0:
                if result.summary is None:
                    result.summary = {}
                result.summary["cost"] = run_cost.to_dict()

            # RFC-014: Transition to COMPLETED
            if self._lifecycle:
                from llmteam.team.lifecycle import TeamState
                target = TeamState.COMPLETED if result.success else TeamState.FAILED
                self._lifecycle.transition_to(target)

            return result

        except Exception as e:
            # RFC-010: End cost tracking on error
            if self._cost_tracker.current_run:
                self._cost_tracker.end_run()

            # RFC-014: Transition to FAILED
            if self._lifecycle:
                from llmteam.team.lifecycle import TeamState
                self._lifecycle.transition_to(TeamState.FAILED)

            return RunResult(
                success=False,
                status=RunStatus.FAILED,
                error=str(e),
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

        finally:
            # End orchestrator tracking
            if self._orchestrator:
                self._orchestrator.end_run()
            self._current_run_id = None

    async def stream(
        self,
        input_data: Dict[str, Any],
        run_id: Optional[str] = None,
    ) -> AsyncIterator["StreamEvent"]:
        """
        Execute team with streaming events (RFC-011).

        Yields StreamEvent objects during execution for real-time progress.
        Only supported in ROUTER mode.

        Args:
            input_data: Input data for execution
            run_id: Optional run identifier

        Yields:
            StreamEvent objects (RUN_STARTED, AGENT_STARTED, AGENT_COMPLETED, etc.)

        Example:
            async for event in team.stream({"query": "Hello"}):
                if event.type == StreamEventType.TOKEN:
                    print(event.data["token"], end="")
                elif event.type == StreamEventType.AGENT_COMPLETED:
                    print(f"Agent {event.agent_id} done")
        """
        from datetime import datetime
        from llmteam.events.streaming import StreamEventType, StreamEvent

        run_id = run_id or str(uuid.uuid4())
        self._current_run_id = run_id

        if not self._agents:
            yield StreamEvent(
                type=StreamEventType.RUN_FAILED,
                data={"error": "No agents in team"},
                run_id=run_id,
            )
            return

        if not self.is_router_mode:
            yield StreamEvent(
                type=StreamEventType.RUN_FAILED,
                data={"error": "Streaming only supported in ROUTER mode"},
                run_id=run_id,
            )
            return

        # Emit RUN_STARTED
        yield StreamEvent(
            type=StreamEventType.RUN_STARTED,
            data={"team_id": self.team_id, "agents": list(self._agents.keys())},
            run_id=run_id,
        )

        # Start orchestrator and cost tracking
        if self._orchestrator:
            self._orchestrator.start_run(run_id)
        self._cost_tracker.start_run(run_id, self.team_id)

        outputs = {}
        agents_called = []
        successful_agents = set()
        current_state = {"input": input_data, "outputs": outputs}
        iterations = 0
        max_iterations = len(self._agents)
        error_msg = None

        try:
            while iterations < max_iterations:
                available_agents = [
                    a for a in self._agents.keys()
                    if a not in successful_agents
                ]
                if not available_agents:
                    break

                decision = await self._orchestrator.decide_next_agent(
                    current_state=current_state,
                    available_agents=available_agents,
                )

                if decision.next_agent is None or decision.next_agent == "":
                    break

                agent = self._agents.get(decision.next_agent)
                if agent is None:
                    iterations += 1
                    continue

                if decision.next_agent in successful_agents:
                    break

                # Emit AGENT_STARTED
                yield StreamEvent(
                    type=StreamEventType.AGENT_STARTED,
                    data={"role": agent.role, "reason": decision.reason},
                    run_id=run_id,
                    agent_id=agent.agent_id,
                )

                # Build context
                context = {"outputs": outputs}
                if outputs:
                    last_agent = agents_called[-1] if agents_called else None
                    if last_agent and last_agent in outputs:
                        context["previous"] = outputs[last_agent]

                # Execute agent
                result = await agent.execute(
                    input_data=input_data,
                    context=context,
                    run_id=run_id,
                )

                # RFC-010: Record token usage
                if result.tokens_used > 0:
                    agent_model = getattr(agent, "model", self._model)
                    estimated_input = int(result.tokens_used * 0.6)
                    estimated_output = result.tokens_used - estimated_input
                    self._cost_tracker.record_usage(
                        model=agent_model,
                        input_tokens=estimated_input,
                        output_tokens=estimated_output,
                        agent_id=agent.agent_id,
                    )

                    # Emit COST_UPDATE
                    yield StreamEvent(
                        type=StreamEventType.COST_UPDATE,
                        data={
                            "current_cost": self._cost_tracker.current_cost,
                            "tokens": result.tokens_used,
                        },
                        run_id=run_id,
                        agent_id=agent.agent_id,
                    )

                    # Check budget
                    if self._budget_manager:
                        status = self._budget_manager.check(
                            self._cost_tracker.current_cost
                        )
                        if status == BudgetStatus.EXCEEDED:
                            yield StreamEvent(
                                type=StreamEventType.RUN_FAILED,
                                data={"error": "Budget exceeded"},
                                run_id=run_id,
                            )
                            break

                # Store result
                outputs[agent.agent_id] = result.output
                agents_called.append(agent.agent_id)

                # Emit AGENT_COMPLETED or AGENT_FAILED
                if result.success:
                    successful_agents.add(agent.agent_id)
                    yield StreamEvent(
                        type=StreamEventType.AGENT_COMPLETED,
                        data={"output": result.output},
                        run_id=run_id,
                        agent_id=agent.agent_id,
                    )
                    break
                else:
                    yield StreamEvent(
                        type=StreamEventType.AGENT_FAILED,
                        data={"error": result.error or "Unknown error"},
                        run_id=run_id,
                        agent_id=agent.agent_id,
                    )

                current_state["outputs"] = outputs
                current_state["last_agent"] = agent.agent_id
                current_state["last_result"] = result.output
                iterations += 1

        except Exception as e:
            error_msg = str(e)
            yield StreamEvent(
                type=StreamEventType.RUN_FAILED,
                data={"error": error_msg},
                run_id=run_id,
            )

        finally:
            # End cost tracking
            if self._cost_tracker.current_run:
                run_cost = self._cost_tracker.end_run()
            else:
                run_cost = None

            if self._orchestrator:
                self._orchestrator.end_run()
            self._current_run_id = None

        # Emit RUN_COMPLETED
        if not error_msg:
            final_output = None
            if agents_called and agents_called[-1] in outputs:
                final_output = outputs[agents_called[-1]]

            yield StreamEvent(
                type=StreamEventType.RUN_COMPLETED,
                data={
                    "success": len(successful_agents) > 0,
                    "output": final_output,
                    "agents_called": agents_called,
                    "cost": run_cost.to_dict() if run_cost else None,
                },
                run_id=run_id,
            )

    async def _run_canvas_mode(
        self,
        input_data: Dict[str, Any],
        run_id: str,
        started_at,
    ) -> RunResult:
        """
        Run team in Canvas mode (PASSIVE orchestrator).

        Canvas (ExecutionEngine) controls the flow.
        Requires engine module: pip install llmteam-ai[engine]
        """
        from datetime import datetime

        # Check if engine is available
        if not _ENGINE_AVAILABLE:
            return RunResult(
                success=False,
                status=RunStatus.FAILED,
                error=(
                    "Engine module not installed. Canvas mode requires: "
                    "pip install llmteam-ai[engine]. "
                    "Use ROUTER mode (orchestrator=OrchestratorConfig(mode=OrchestratorMode.ACTIVE)) "
                    "for basic functionality without engine."
                ),
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

        # Build segment from agents (orchestrator NOT included)
        segment = build_segment(
            team_id=self.team_id,
            agents=self._agents,
            flow=self._flow,
        )

        # Get or create runner
        runner = self._get_runner()

        # Build runtime context
        runtime = self._build_runtime(run_id)

        # Run workflow
        from llmteam.engine.engine import RunConfig

        config = RunConfig()
        if self._timeout:
            config.timeout_seconds = self._timeout

        segment_result = await runner.run(
            segment=segment,
            input_data=input_data,
            runtime=runtime,
            config=config,
        )

        # Convert result
        result = result_from_segment_result(segment_result, self._agents)
        result.started_at = started_at
        result.completed_at = datetime.utcnow()

        return result

    async def _run_router_mode(
        self,
        input_data: Dict[str, Any],
        run_id: str,
        started_at,
    ) -> RunResult:
        """
        Run team in Router mode (ACTIVE orchestrator).

        Orchestrator decides which agent runs next.
        For simple triage: ONE agent is selected and runs once.
        """
        from datetime import datetime

        outputs = {}
        agents_called = []
        successful_agents = set()  # Track successful completions
        current_state = {"input": input_data, "outputs": outputs}
        iterations = 0
        max_iterations = len(self._agents)  # Max = one call per agent

        while iterations < max_iterations:
            # Filter out agents that already succeeded
            available_agents = [
                a for a in self._agents.keys()
                if a not in successful_agents
            ]

            # If all agents have run or none available, stop
            if not available_agents:
                break

            # Ask orchestrator which agent to run
            decision = await self._orchestrator.decide_next_agent(
                current_state=current_state,
                available_agents=available_agents,
            )

            # Check if done
            if decision.next_agent is None or decision.next_agent == "":
                break

            # Get agent
            agent = self._agents.get(decision.next_agent)
            if agent is None:
                # Invalid agent, skip
                iterations += 1
                continue

            # Skip if already ran successfully (safety check)
            if decision.next_agent in successful_agents:
                break

            # Build context from previous outputs
            context = {"outputs": outputs}
            if outputs:
                # Pass last output as input for next agent
                last_agent = agents_called[-1] if agents_called else None
                if last_agent and last_agent in outputs:
                    context["previous"] = outputs[last_agent]

            # Execute agent
            result = await agent.execute(
                input_data=input_data,
                context=context,
                run_id=run_id,
            )

            # RFC-010: Record token usage for cost tracking
            if result.tokens_used > 0:
                agent_model = getattr(agent, "model", self._model)
                # Estimate input/output split (60/40 heuristic when no detail)
                estimated_input = int(result.tokens_used * 0.6)
                estimated_output = result.tokens_used - estimated_input
                self._cost_tracker.record_usage(
                    model=agent_model,
                    input_tokens=estimated_input,
                    output_tokens=estimated_output,
                    agent_id=agent.agent_id,
                )

                # Check budget
                if self._budget_manager:
                    from llmteam.cost import BudgetExceededError
                    status = self._budget_manager.check(self._cost_tracker.current_cost)
                    if status == BudgetStatus.EXCEEDED:
                        break  # Stop execution

            # Store result
            outputs[agent.agent_id] = result.output
            agents_called.append(agent.agent_id)

            # Track success
            if result.success:
                successful_agents.add(agent.agent_id)
                # For simple triage, one successful agent = done
                break

            # Update state for next iteration (if retrying or using fallback)
            current_state["outputs"] = outputs
            current_state["last_agent"] = agent.agent_id
            current_state["last_result"] = result.output

            iterations += 1

        # Build result
        final_output = None
        if agents_called and agents_called[-1] in outputs:
            final_output = outputs[agents_called[-1]]

        return RunResult(
            success=len(successful_agents) > 0,
            status=RunStatus.COMPLETED if successful_agents else RunStatus.FAILED,
            output=outputs,
            final_output=final_output,
            agents_called=agents_called,
            iterations=iterations,
            started_at=started_at,
            completed_at=datetime.utcnow(),
        )

    # Pause/Resume

    async def pause(self) -> TeamSnapshot:
        """
        Pause execution and return snapshot.

        Requires engine module: pip install llmteam-ai[engine]

        Returns:
            TeamSnapshot for resume
        """
        if not _ENGINE_AVAILABLE:
            raise RuntimeError(
                "Engine module not installed. Pause/resume requires: "
                "pip install llmteam-ai[engine]"
            )

        if not self._current_run_id:
            raise RuntimeError("No active run to pause")

        runner = self._get_runner()
        segment_snapshot = await runner.pause(self._current_run_id)

        return TeamSnapshot.from_segment_snapshot(segment_snapshot, self.team_id)

    async def resume(self, snapshot: TeamSnapshot) -> RunResult:
        """
        Resume execution from snapshot.

        Requires engine module: pip install llmteam-ai[engine]

        Args:
            snapshot: TeamSnapshot from pause()

        Returns:
            RunResult
        """
        if not _ENGINE_AVAILABLE:
            raise RuntimeError(
                "Engine module not installed. Pause/resume requires: "
                "pip install llmteam-ai[engine]"
            )

        runner = self._get_runner()

        # Build segment
        segment = build_segment(
            team_id=self.team_id,
            agents=self._agents,
            flow=self._flow,
        )

        # Convert to segment snapshot
        segment_snapshot = snapshot.to_segment_snapshot()

        # Build runtime
        runtime = self._build_runtime(snapshot.run_id)

        # Resume
        segment_result = await runner.resume(
            snapshot=segment_snapshot,
            segment=segment,
            runtime=runtime,
        )

        return result_from_segment_result(segment_result, self._agents)

    async def cancel(self) -> bool:
        """
        Cancel current execution.

        Requires engine module: pip install llmteam-ai[engine]

        Returns:
            True if cancelled successfully
        """
        if not _ENGINE_AVAILABLE:
            raise RuntimeError(
                "Engine module not installed. Cancel requires: "
                "pip install llmteam-ai[engine]"
            )

        if not self._current_run_id:
            return False

        runner = self._get_runner()
        return await runner.cancel(self._current_run_id)

    # Events

    def on(self, event: str, callback: Callable) -> None:
        """
        Register event callback.

        Args:
            event: Event name (e.g., "agent_complete", "step_start")
            callback: Callback function
        """
        if event not in self._event_callbacks:
            self._event_callbacks[event] = []
        self._event_callbacks[event].append(callback)

    def off(self, event: str, callback: Callable) -> None:
        """Remove event callback."""
        if event in self._event_callbacks:
            self._event_callbacks[event] = [
                cb for cb in self._event_callbacks[event] if cb != callback
            ]

    # Escalation

    async def escalate(
        self,
        source_agent: str,
        reason: str,
        context: Optional[Dict] = None,
    ) -> Any:
        """
        Handle escalation from agent.

        Args:
            source_agent: Agent ID that escalated
            reason: Escalation reason
            context: Additional context

        Returns:
            Escalation decision
        """
        # TODO: Integrate with escalation subsystem
        return {
            "action": "continue",
            "source": source_agent,
            "reason": reason,
        }

    # Groups

    def create_group(
        self,
        group_id: str,
        teams: List["LLMTeam"],
    ) -> "LLMGroup":
        """
        Create a group with this team as leader.

        Args:
            group_id: Group identifier
            teams: Other teams in the group

        Returns:
            LLMGroup instance
        """
        from llmteam.team.group import LLMGroup

        return LLMGroup(
            group_id=group_id,
            leader=self,
            teams=teams,
            model=self._model,
        )

    # RFC-009: Group Integration

    def _join_group(self, context: "GroupContext") -> None:
        """
        INTERNAL: Called by GroupOrchestrator when adding team to group.

        Do not call directly! Use GroupOrchestrator.add_team().

        Args:
            context: GroupContext with group info
        """
        self._group_context = context

        # Notify TeamOrchestrator
        if self._orchestrator and hasattr(self._orchestrator, "_set_group_context"):
            self._orchestrator._set_group_context(context)

    def _leave_group(self) -> None:
        """
        INTERNAL: Called by GroupOrchestrator when removing team from group.
        """
        if self._group_context:
            self._group_context = None

            if self._orchestrator and hasattr(self._orchestrator, "_clear_group_context"):
                self._orchestrator._clear_group_context()

    @property
    def is_in_group(self) -> bool:
        """Is team part of a group? (RFC-009)"""
        return self._group_context is not None

    @property
    def group_id(self) -> Optional[str]:
        """Group ID (if team is in a group). (RFC-009)"""
        return self._group_context.group_id if self._group_context else None

    @property
    def group_role(self) -> Optional[str]:
        """Team's role in the group (if in a group). (RFC-009)"""
        if self._group_context:
            return self._group_context.team_role.value
        return None

    async def escalate_to_group(
        self,
        reason: str,
        context: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
        source_agent: Optional[str] = None,
    ) -> "EscalationResponse":
        """
        Escalate issue to group orchestrator (RFC-009).

        Args:
            reason: Escalation reason
            context: Additional context
            error: Exception (if any)
            source_agent: Source agent ID (if escalation from agent)

        Returns:
            EscalationResponse from GroupOrchestrator

        Raises:
            RuntimeError: If team is not in a group or escalation not allowed
        """
        from llmteam.orchestration.models import EscalationRequest, EscalationResponse, EscalationAction

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
        Request execution from another team in the group (RFC-009).

        Only available for LEADER and SPECIALIST roles.

        Args:
            target_team_id: Target team ID
            task: Task data to execute

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

    # Configuration (RFC-005)

    async def configure(
        self,
        task: str,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> "ConfigurationSession":
        """
        Start configuration session via CONFIGURATOR mode (RFC-005).

        Allows interactive team configuration with LLM assistance.

        Args:
            task: Task description in natural language
            constraints: Task constraints (tone, length, format, etc.)

        Returns:
            ConfigurationSession for iterative configuration

        Example:
            session = await team.configure(
                task="Generate LinkedIn posts from press releases",
                constraints={"tone": "professional", "length": "<300"}
            )

            # Review suggestions
            print(session.suggested_agents)

            # Test
            test = await session.test_run({"press_release": "..."})
            print(test.analysis)

            # Apply
            await session.apply()
        """
        from llmteam.configuration import ConfigurationSession

        # Enable CONFIGURATOR mode
        if self._orchestrator:
            self._orchestrator._config.mode |= OrchestratorMode.CONFIGURATOR

        # Create session
        session = ConfigurationSession(
            session_id=str(uuid.uuid4()),
            team=self,
            task=task,
            constraints=constraints or {},
        )

        # Analyze and suggest
        await session.analyze()
        await session.suggest()

        return session

    def remove_agent(self, agent_id: str) -> bool:
        """
        Remove agent from team.

        Args:
            agent_id: Agent ID to remove

        Returns:
            True if removed, False if not found
        """
        if agent_id in self._agents:
            del self._agents[agent_id]
            return True
        return False

    def set_flow(self, flow: Union[str, Dict]) -> None:
        """
        Set execution flow.

        Args:
            flow: Flow string ("a -> b -> c") or DAG dict
        """
        self._flow = flow

    # Serialization

    def to_config(self) -> Dict[str, Any]:
        """Export team configuration."""
        return {
            "team_id": self.team_id,
            "agents": [agent.to_dict() for agent in self._agents.values()],
            "flow": self._flow,
            "model": self._model,
            "context_mode": self._context_mode.value,
            "quality": self.quality,
            "max_cost_per_run": self._max_cost_per_run,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LLMTeam":
        """Create team from configuration."""
        return cls(
            team_id=config["team_id"],
            agents=config.get("agents", []),
            flow=config.get("flow", "sequential"),
            model=config.get("model", "gpt-4o-mini"),
            context_mode=ContextMode(config.get("context_mode", "shared")),
            quality=config.get("quality", 50),
            max_cost_per_run=config.get("max_cost_per_run"),
        )

    @classmethod
    def from_segment(cls, segment, team_id: Optional[str] = None) -> "LLMTeam":
        """
        Create team from WorkflowDefinition (formerly SegmentDefinition).

        Requires engine module: pip install llmteam-ai[engine]

        Args:
            segment: WorkflowDefinition
            team_id: Optional team ID (default: workflow_id)

        Returns:
            LLMTeam
        """
        if not _ENGINE_AVAILABLE:
            raise RuntimeError(
                "Engine module not installed. from_segment requires: "
                "pip install llmteam-ai[engine]"
            )

        # Support both new (workflow_id) and old (segment_id) names
        workflow_id = getattr(segment, 'workflow_id', None) or getattr(segment, 'segment_id', None)

        team = cls(
            team_id=team_id or workflow_id,
            flow={"edges": [e.to_dict() for e in segment.edges]},
        )

        # Convert steps to agents
        for step in segment.steps:
            if step.type in ("llm", "rag", "kag"):
                config = {"type": step.type, "role": step.step_id, **step.config}
                team.add_agent(config)

        return team

    # Internal

    def _get_runner(self):
        """Get or create ExecutionEngine.

        Requires engine module: pip install llmteam-ai[engine]
        """
        if not _ENGINE_AVAILABLE:
            raise RuntimeError(
                "Engine module not installed. Install with: "
                "pip install llmteam-ai[engine]"
            )

        if self._runner is None:
            from llmteam.engine.engine import ExecutionEngine

            self._runner = ExecutionEngine()
        return self._runner

    def _build_runtime(self, run_id: str):
        """Build RuntimeContext for execution."""
        from llmteam.runtime import RuntimeContextFactory

        if self._runtime:
            return self._runtime.child_context(run_id)

        # Create runtime with default LLM provider
        factory = RuntimeContextFactory()

        # Auto-register OpenAI provider as default
        try:
            from llmteam.providers import OpenAIProvider

            # Register default provider
            default_provider = OpenAIProvider(model=self._model)
            factory.register_llm("default", default_provider)

            # Register model-specific providers for agents
            models_used = set()
            for agent in self._agents.values():
                if hasattr(agent, "model") and agent.model:
                    models_used.add(agent.model)

            for model in models_used:
                if model != self._model:
                    factory.register_llm(model, OpenAIProvider(model=model))

        except ImportError:
            pass  # OpenAI not installed, provider must be set manually

        return factory.create_runtime(
            tenant_id="default",
            instance_id=run_id,
        )

    def set_runtime(self, runtime) -> None:
        """Set runtime context for execution."""
        self._runtime = runtime

    # Magic methods

    def __repr__(self) -> str:
        return f"<LLMTeam id='{self.team_id}' agents={len(self._agents)}>"

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._agents
