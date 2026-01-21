"""
Converters for team execution.

Converts agents and flow definitions to WorkflowDefinition for ExecutionEngine.
"""

from typing import Any, Dict, List, Optional, Union
import re

from llmteam.engine.models import (
    StepDefinition,
    EdgeDefinition,
    WorkflowDefinition,
    SegmentDefinition,  # Backward compatibility wrapper
)


def parse_flow_string(flow: str, agent_ids: List[str]) -> List[EdgeDefinition]:
    """
    Parse flow string to edge definitions.

    Supported formats:
    - "a -> b -> c" : sequential
    - "a, b -> c" : parallel a and b, then c
    - "a -> b, c -> d" : a, then parallel b and c, then d

    Args:
        flow: Flow string
        agent_ids: List of agent IDs for validation

    Returns:
        List of EdgeDefinition
    """
    if not flow or flow == "sequential":
        # Default: sequential flow through all agents
        return create_sequential_edges(agent_ids)

    edges = []
    # Split by ->
    parts = [p.strip() for p in flow.split("->")]

    for i in range(len(parts) - 1):
        from_part = parts[i]
        to_part = parts[i + 1]

        # Parse comma-separated agents
        from_agents = [a.strip() for a in from_part.split(",")]
        to_agents = [a.strip() for a in to_part.split(",")]

        # Create edges from each 'from' to each 'to'
        for from_agent in from_agents:
            for to_agent in to_agents:
                if from_agent and to_agent:
                    edges.append(
                        EdgeDefinition(
                            from_step=from_agent,
                            to_step=to_agent,
                        )
                    )

    return edges


def parse_flow_dict(flow: Dict[str, Any]) -> List[EdgeDefinition]:
    """
    Parse flow dictionary to edge definitions.

    Expected format:
    {
        "edges": [
            {"from": "a", "to": "b"},
            {"from": "b", "to": "c", "condition": "approved"},
            {"from": "b", "to": "a", "condition": "rejected"}
        ]
    }

    Args:
        flow: Flow dictionary

    Returns:
        List of EdgeDefinition
    """
    edges = []

    for edge_data in flow.get("edges", []):
        from_step = edge_data.get("from") or edge_data.get("from_step")
        to_step = edge_data.get("to") or edge_data.get("to_step")
        condition = edge_data.get("condition")

        if from_step and to_step:
            edges.append(
                EdgeDefinition(
                    from_step=from_step,
                    to_step=to_step,
                    condition=condition,
                )
            )

    return edges


def create_sequential_edges(agent_ids: List[str]) -> List[EdgeDefinition]:
    """
    Create sequential edges for agents.

    Args:
        agent_ids: List of agent IDs in order

    Returns:
        List of EdgeDefinition
    """
    edges = []
    for i in range(len(agent_ids) - 1):
        edges.append(
            EdgeDefinition(
                from_step=agent_ids[i],
                to_step=agent_ids[i + 1],
            )
        )
    return edges


# Mapping from AgentType values to Canvas step types
AGENT_TYPE_TO_STEP_TYPE = {
    "llm": "llm_agent",
    "rag": "rag",
    "kag": "kag",
}


def agents_to_steps(agents: Dict[str, Any]) -> List[StepDefinition]:
    """
    Convert agents to step definitions.

    Each agent becomes a step with type mapped to canvas handler.

    Args:
        agents: Dict of agent_id -> BaseAgent

    Returns:
        List of StepDefinition
    """
    steps = []

    for agent_id, agent in agents.items():
        # Map agent type to canvas step type
        agent_type_value = agent.agent_type.value
        step_type = AGENT_TYPE_TO_STEP_TYPE.get(agent_type_value, agent_type_value)

        # Build config from agent
        config = {
            "role": agent.role,
            "name": agent.name,
            "description": agent.description,
        }

        # Add type-specific config
        if hasattr(agent, "prompt"):
            config["prompt"] = agent.prompt
        if hasattr(agent, "system_prompt"):
            config["system_prompt"] = agent.system_prompt
        if hasattr(agent, "model"):
            config["model"] = agent.model
        if hasattr(agent, "temperature"):
            config["temperature"] = agent.temperature
        if hasattr(agent, "max_tokens"):
            config["max_tokens"] = agent.max_tokens
        if hasattr(agent, "top_k"):
            config["top_k"] = agent.top_k
        if hasattr(agent, "collection"):
            config["collection"] = agent.collection
        if hasattr(agent, "use_context"):
            config["use_context"] = agent.use_context

        steps.append(
            StepDefinition(
                step_id=agent_id,
                type=step_type,  # "llm_agent", "rag", "kag"
                config=config,
            )
        )

    return steps


def build_segment(
    team_id: str,
    agents: Dict[str, Any],
    flow: Union[str, Dict[str, Any]],
    entrypoint: Optional[str] = None,
) -> SegmentDefinition:
    """
    Build SegmentDefinition from agents and flow.

    Args:
        team_id: Team ID (becomes segment_id)
        agents: Dict of agent_id -> BaseAgent
        flow: Flow definition (string or dict)
        entrypoint: Optional entrypoint agent (default: first agent)

    Returns:
        SegmentDefinition ready for SegmentRunner
    """
    # Get agent IDs in order
    agent_ids = list(agents.keys())

    if not agent_ids:
        raise ValueError("At least one agent is required")

    # Determine entrypoint
    if entrypoint is None:
        entrypoint = agent_ids[0]

    # Convert agents to steps
    steps = agents_to_steps(agents)

    # Parse flow to edges
    if isinstance(flow, str):
        edges = parse_flow_string(flow, agent_ids)
    elif isinstance(flow, dict):
        edges = parse_flow_dict(flow)
    else:
        # Default: sequential
        edges = create_sequential_edges(agent_ids)

    return SegmentDefinition(
        segment_id=team_id,
        name=f"Team {team_id}",
        entrypoint=entrypoint,
        steps=steps,
        edges=edges,
    )


def result_from_segment_result(segment_result, agents_map: Dict[str, Any]):
    """
    Convert SegmentResult to RunResult.

    Args:
        segment_result: Result from SegmentRunner
        agents_map: Dict of agent_id -> BaseAgent

    Returns:
        RunResult
    """
    from llmteam.team.result import RunResult, RunStatus
    from llmteam.engine.engine import ExecutionStatus
    # Backward compatibility alias
    SegmentStatus = ExecutionStatus

    # Map status
    status_map = {
        SegmentStatus.COMPLETED: RunStatus.COMPLETED,
        SegmentStatus.FAILED: RunStatus.FAILED,
        SegmentStatus.PAUSED: RunStatus.PAUSED,
        SegmentStatus.CANCELLED: RunStatus.CANCELLED,
        SegmentStatus.TIMEOUT: RunStatus.TIMEOUT,
        SegmentStatus.RUNNING: RunStatus.RUNNING,
        SegmentStatus.PENDING: RunStatus.PENDING,
    }

    status = status_map.get(segment_result.status, RunStatus.FAILED)

    # Get final output from last completed step
    final_output = None
    if segment_result.completed_steps and segment_result.step_outputs:
        last_step = segment_result.completed_steps[-1]
        last_output = segment_result.step_outputs.get(last_step, {})
        # Extract text output
        if isinstance(last_output, dict):
            final_output = last_output.get("output", last_output)
        else:
            final_output = last_output

    return RunResult(
        success=segment_result.status == SegmentStatus.COMPLETED,
        status=status,
        output=segment_result.step_outputs,
        final_output=final_output,
        agents_called=segment_result.completed_steps,
        iterations=segment_result.steps_completed,
        duration_ms=segment_result.duration_ms,
        error=str(segment_result.error) if segment_result.error else None,
        started_at=segment_result.started_at,
        completed_at=segment_result.completed_at,
    )
