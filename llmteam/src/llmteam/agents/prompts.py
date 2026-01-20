"""
Prompts for TeamOrchestrator.

Predefined prompts for routing, recovery, and reporting decisions.
"""

from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from llmteam.agents.report import AgentReport
    from llmteam.team import LLMTeam


# System prompts

ORCHESTRATOR_SYSTEM_PROMPT = """You are a team orchestrator responsible for supervising agent execution.

Team: {team_id}
Available Agents: {agent_descriptions}

Your responsibilities depend on your current mode:
- SUPERVISOR: Monitor agent execution and collect reports
- REPORTER: Generate summaries of execution
- ROUTER: Decide which agent should execute next
- RECOVERY: Decide how to handle errors

Always respond in the specified JSON format."""


# Routing prompt

ROUTING_PROMPT = """Based on the current state and execution history, decide which agent should run next.

## Current State
Input: {input_summary}
Progress: {progress}

## Execution History
{execution_log}

## Available Agents
{agent_descriptions}

## Instructions
You are orchestrating a team of agents. For each task:
1. Select ONE agent that is most appropriate to handle the task
2. After that agent completes successfully, the task is DONE
3. Do NOT call the same agent twice for the same task
4. Do NOT call another agent unless the first one failed

IMPORTANT: If an agent has already successfully handled the task, return null to indicate completion.

Respond in JSON format:
{{
    "next_agent": "<agent_id>" or null,
    "reason": "<brief explanation>",
    "confidence": <0.0-1.0>
}}

Return null for next_agent when:
- An agent has already successfully handled the task
- No more processing is needed
- The task is complete"""


# Error recovery prompt

ERROR_RECOVERY_PROMPT = """An error occurred during agent execution. Decide the best recovery action.

## Error Details
Agent: {failed_agent}
Error Type: {error_type}
Error Message: {error_message}

## Context
{context}

## Execution History
{execution_log}

## Available Actions
- RETRY: Retry the failed agent (optionally with modifications)
- SKIP: Skip this agent and continue to next
- FALLBACK: Use a different agent as fallback
- ESCALATE: Escalate to human/supervisor
- ABORT: Stop execution

## Instructions
Analyze the error and decide the best course of action.
Consider:
1. Is the error transient (retry might help)?
2. Can another agent handle this task?
3. Is the error critical (abort required)?

Respond in JSON format:
{{
    "action": "RETRY|SKIP|FALLBACK|ESCALATE|ABORT",
    "reason": "<explanation>",
    "changes": {{}},  // optional: modifications for RETRY
    "fallback": "<agent_id>"  // optional: for FALLBACK action
}}"""


# Report generation prompt

REPORT_GENERATION_PROMPT = """Generate an execution report based on the following agent logs.

## Team Information
Team: {team_id}
Run ID: {run_id}
Started: {started_at}

## Execution Log
{execution_log}

## Output Format: {format}

## Instructions
Generate a {format} report that summarizes:
1. Overall execution status
2. Each agent's contribution
3. Key outputs and decisions
4. Any issues encountered
5. Total metrics (duration, tokens)

Keep the report concise but informative."""


# Team summary for GROUP scope

TEAM_SUMMARY_TEMPLATE = """Team: {team_id}
Status: {status}
Agents: {agent_count}
Last Run: {last_run_status}
Capabilities: {capabilities}"""


# Helper functions

def build_routing_prompt(
    team: "LLMTeam",
    current_state: Dict[str, Any],
    available_agents: List[str],
    execution_history: List["AgentReport"],
) -> str:
    """
    Build routing prompt with current context.

    Args:
        team: The team being orchestrated
        current_state: Current execution state
        available_agents: List of available agent IDs
        execution_history: Reports from executed agents

    Returns:
        Formatted routing prompt
    """
    # Build agent descriptions
    agent_descriptions = []
    for agent_id in available_agents:
        agent = team.get_agent(agent_id)
        if agent:
            desc = f"- {agent_id} ({agent.agent_type.value}): {agent.description or agent.name}"
            agent_descriptions.append(desc)

    # Build execution log with clear success/failure info
    exec_log = []
    successful_agents = []
    for report in execution_history:
        status = "SUCCESS" if report.success else f"FAILED: {report.error}"
        exec_log.append(
            f"- {report.agent_id}: {status} ({report.duration_ms}ms)"
        )
        if report.success:
            successful_agents.append(report.agent_id)

    # Create input summary
    input_str = str(current_state.get("input", current_state))
    if len(input_str) > 200:
        input_str = input_str[:200] + "..."

    # Add clear progress message
    if successful_agents:
        progress = f"{len(execution_history)} agents executed. SUCCESS: {', '.join(successful_agents)}. TASK IS COMPLETE - return null."
    else:
        progress = f"{len(execution_history)} agents executed"

    return ROUTING_PROMPT.format(
        input_summary=input_str,
        progress=progress,
        execution_log="\n".join(exec_log) if exec_log else "No agents executed yet",
        agent_descriptions="\n".join(agent_descriptions),
    )


def build_recovery_prompt(
    error: Exception,
    failed_agent: str,
    context: Dict[str, Any],
    team: "LLMTeam",
) -> str:
    """
    Build error recovery prompt.

    Args:
        error: The exception that occurred
        failed_agent: ID of agent that failed
        context: Execution context
        team: The team being orchestrated

    Returns:
        Formatted recovery prompt
    """
    # Build execution log from context if available
    exec_log = []
    if "reports" in context:
        for report in context["reports"]:
            if hasattr(report, "to_log_line"):
                exec_log.append(report.to_log_line())
            else:
                exec_log.append(str(report))

    # Create context summary
    context_summary = str(context)
    if len(context_summary) > 500:
        context_summary = context_summary[:500] + "..."

    return ERROR_RECOVERY_PROMPT.format(
        failed_agent=failed_agent,
        error_type=type(error).__name__,
        error_message=str(error),
        context=context_summary,
        execution_log="\n".join(exec_log) if exec_log else "None",
    )


def build_report_prompt(
    team_id: str,
    run_id: str,
    started_at: str,
    execution_history: List["AgentReport"],
    format: str = "markdown",
) -> str:
    """
    Build report generation prompt.

    Args:
        team_id: Team identifier
        run_id: Run identifier
        started_at: Start time
        execution_history: Reports from executed agents
        format: Output format (markdown, json, text)

    Returns:
        Formatted report prompt
    """
    # Build execution log
    exec_log = []
    for report in execution_history:
        exec_log.append(
            f"Agent: {report.agent_id}\n"
            f"  Type: {report.agent_type}\n"
            f"  Duration: {report.duration_ms}ms\n"
            f"  Success: {report.success}\n"
            f"  Output: {report.output_summary}\n"
        )

    return REPORT_GENERATION_PROMPT.format(
        team_id=team_id,
        run_id=run_id,
        started_at=started_at,
        execution_log="\n".join(exec_log),
        format=format,
    )


def build_agent_description(agent) -> str:
    """
    Build description string for an agent.

    Args:
        agent: BaseAgent instance

    Returns:
        Formatted description string
    """
    return (
        f"{agent.agent_id} ({agent.agent_type.value}): "
        f"{agent.description or agent.name}"
    )
