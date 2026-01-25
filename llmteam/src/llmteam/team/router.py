"""
Router loop for ROUTER mode execution.

RFC-021: Quality parameter removed. Agents use their configured models.

Provides shared router iteration logic used by both
_run_router_mode() and stream() methods.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Set

if TYPE_CHECKING:
    from llmteam.agents.base import BaseAgent
    from llmteam.agents.orchestrator import TeamOrchestrator
    from llmteam.cost import BudgetManager, CostTracker


class RouterEventType(Enum):
    """Internal router event types."""

    # User input (captured at start)
    USER_INPUT = auto()

    # Agent lifecycle
    AGENT_SELECTED = auto()
    AGENT_STARTED = auto()
    AGENT_COMPLETED = auto()
    AGENT_FAILED = auto()

    # Tool events (RFC-017)
    TOOL_CALL = auto()
    TOOL_RESULT = auto()

    # Cost events (RFC-010)
    COST_UPDATE = auto()
    BUDGET_EXCEEDED = auto()

    # Loop control
    LOOP_DONE = auto()


@dataclass
class RouterEvent:
    """
    Internal router event.

    Used by _router_loop() to communicate with callers.
    Callers (stream, _run_router_mode) map these to their output format.
    """

    type: RouterEventType
    agent_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouterState:
    """
    Shared state for router loop.

    Tracks outputs, called agents, and iteration count.
    """

    outputs: Dict[str, Any] = field(default_factory=dict)
    agents_called: List[str] = field(default_factory=list)
    successful_agents: Set[str] = field(default_factory=set)
    current_state: Dict[str, Any] = field(default_factory=dict)
    iterations: int = 0
    error: Optional[str] = None

    def __post_init__(self):
        if not self.current_state:
            self.current_state = {"outputs": self.outputs}

    @property
    def final_output(self) -> Any:
        """Get output from last called agent."""
        if self.agents_called and self.agents_called[-1] in self.outputs:
            return self.outputs[self.agents_called[-1]]
        return None

    @property
    def success(self) -> bool:
        """Did at least one agent succeed?"""
        return len(self.successful_agents) > 0


async def router_loop(
    agents: Dict[str, "BaseAgent"],
    orchestrator: "TeamOrchestrator",
    input_data: Dict[str, Any],
    run_id: str,
    cost_tracker: Optional["CostTracker"] = None,
    budget_manager: Optional["BudgetManager"] = None,
    default_model: str = "gpt-4o-mini",
    collect_tool_events: bool = False,
) -> AsyncIterator[RouterEvent]:
    """
    Shared router iteration logic.

    RFC-021: Quality parameter removed. Agents use their configured models.

    Generator yielding RouterEvent objects during execution.
    Both stream() and _run_router_mode() consume this.

    Args:
        agents: Dict of agent_id -> BaseAgent
        orchestrator: TeamOrchestrator for routing decisions
        input_data: Input data for execution
        run_id: Run identifier
        cost_tracker: Optional CostTracker for token accounting
        budget_manager: Optional BudgetManager for budget checks
        default_model: Default model name for cost tracking
        collect_tool_events: Whether to collect and yield tool events

    Yields:
        RouterEvent objects for each significant event

    Example:
        async for event in router_loop(...):
            if event.type == RouterEventType.AGENT_COMPLETED:
                print(f"Agent {event.agent_id} done")
    """
    from llmteam.cost import BudgetStatus

    # Initialize state
    state = RouterState()
    state.current_state = {"input": input_data, "outputs": state.outputs}

    max_iterations = len(agents)

    # Emit USER_INPUT event at the start of routing
    yield RouterEvent(
        type=RouterEventType.USER_INPUT,
        data={
            "input": input_data,
        },
    )

    while state.iterations < max_iterations:
        # All agents are available (can be called again if needed for retry)
        # But filter out agents that already succeeded to avoid duplicates
        available_agents = [
            a for a in agents.keys()
            if a not in state.successful_agents
        ]

        # If no agents left to run, we're done
        if not available_agents:
            break

        # Ask orchestrator which agent to run next
        decision = await orchestrator.decide_next_agent(
            current_state=state.current_state,
            available_agents=available_agents,
        )

        # Yield selection event (orchestrator decision)
        yield RouterEvent(
            type=RouterEventType.AGENT_SELECTED,
            agent_id=decision.next_agent,
            data={
                "selected_by": "orchestrator",
                "reason": decision.reason,
                "confidence": getattr(decision, "confidence", None),
                "is_task_complete": getattr(decision, "is_task_complete", False),
            },
        )

        # Check if orchestrator says task is complete
        if getattr(decision, "is_task_complete", False):
            break

        # Check if no next agent (orchestrator signals completion)
        if decision.next_agent is None or decision.next_agent == "":
            break

        # Get agent
        agent = agents.get(decision.next_agent)
        if agent is None:
            state.iterations += 1
            continue

        # Safety: skip if already ran successfully in THIS run
        if decision.next_agent in state.successful_agents:
            state.iterations += 1
            continue

        # Yield AGENT_STARTED
        yield RouterEvent(
            type=RouterEventType.AGENT_STARTED,
            agent_id=agent.agent_id,
            data={"role": agent.role, "reason": decision.reason},
        )

        # Build context from previous outputs - rich context for chaining
        context = {
            "outputs": state.outputs,  # All outputs by agent_id
            "original_input": input_data,  # Original user input
            "agents_called": state.agents_called.copy(),  # Execution order
        }

        # Add previous agent's output for easy access
        if state.agents_called:
            last_agent = state.agents_called[-1]
            if last_agent in state.outputs:
                context["previous"] = state.outputs[last_agent]
                context["previous_output"] = state.outputs[last_agent]  # Explicit name
                context["previous_agent"] = last_agent

        # Build chain history for agents that need full context
        if state.outputs:
            chain_history = []
            for agent_id in state.agents_called:
                if agent_id in state.outputs:
                    chain_history.append({
                        "agent": agent_id,
                        "output": state.outputs[agent_id],
                    })
            context["chain_history"] = chain_history

        # RFC-021: Quality context removed. Agents use their configured models.

        # Set up tool event collection if requested
        agent_tool_events: list = []
        if collect_tool_events and hasattr(agent, "_event_callback"):
            async def _tool_event_cb(event_type: str, data: dict, agent_id: str) -> None:
                agent_tool_events.append((event_type, data, agent_id))
            agent._event_callback = _tool_event_cb

        # Execute agent
        result = await agent.execute(
            input_data=input_data,
            context=context,
            run_id=run_id,
        )

        # Clear callback
        if collect_tool_events and hasattr(agent, "_event_callback"):
            agent._event_callback = None

        # Yield tool events if collected
        for evt_type, evt_data, evt_agent_id in agent_tool_events:
            if evt_type == "tool_call":
                yield RouterEvent(
                    type=RouterEventType.TOOL_CALL,
                    agent_id=evt_agent_id,
                    data=evt_data,
                )
            elif evt_type == "tool_result":
                yield RouterEvent(
                    type=RouterEventType.TOOL_RESULT,
                    agent_id=evt_agent_id,
                    data=evt_data,
                )

        # Cost tracking (RFC-010)
        if cost_tracker and result.tokens_used > 0:
            agent_model = getattr(agent, "model", default_model)
            estimated_input = int(result.tokens_used * 0.6)
            estimated_output = result.tokens_used - estimated_input
            cost_tracker.record_usage(
                model=agent_model,
                input_tokens=estimated_input,
                output_tokens=estimated_output,
                agent_id=agent.agent_id,
            )

            yield RouterEvent(
                type=RouterEventType.COST_UPDATE,
                agent_id=agent.agent_id,
                data={
                    "tokens": result.tokens_used,
                    "current_cost": cost_tracker.current_cost,
                },
            )

            # Budget check
            if budget_manager:
                status = budget_manager.check(cost_tracker.current_cost)
                if status == BudgetStatus.EXCEEDED:
                    yield RouterEvent(
                        type=RouterEventType.BUDGET_EXCEEDED,
                        agent_id=agent.agent_id,
                        data={"current_cost": cost_tracker.current_cost},
                    )
                    state.error = "Budget exceeded"
                    break

        # Store result
        state.outputs[agent.agent_id] = result.output
        state.agents_called.append(agent.agent_id)

        # Track success/failure
        if result.success:
            state.successful_agents.add(agent.agent_id)
            yield RouterEvent(
                type=RouterEventType.AGENT_COMPLETED,
                agent_id=agent.agent_id,
                data={"output": result.output},
            )
            # DON'T break here - let orchestrator decide if more agents needed
            # The loop continues and orchestrator will return is_task_complete=True
            # when the task is fully done
        else:
            yield RouterEvent(
                type=RouterEventType.AGENT_FAILED,
                agent_id=agent.agent_id,
                data={"error": result.error or "Unknown error"},
            )

        # Update state for next iteration - pass output to next agent
        state.current_state["outputs"] = state.outputs
        state.current_state["last_agent"] = agent.agent_id
        state.current_state["last_result"] = result.output
        # Also add previous output directly for easy access
        state.current_state["previous_output"] = result.output

        state.iterations += 1

    # Yield final state
    yield RouterEvent(
        type=RouterEventType.LOOP_DONE,
        data={
            "outputs": state.outputs,
            "agents_called": state.agents_called,
            "successful_agents": list(state.successful_agents),
            "iterations": state.iterations,
            "success": state.success,
            "final_output": state.final_output,
            "error": state.error,
        },
    )
