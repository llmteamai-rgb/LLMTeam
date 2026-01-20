"""
Preset agent configurations.

Orchestrator and GroupOrchestrator are not separate classes,
but preset configs for LLMAgent.
"""

from typing import Dict, List


# Prompts

ORCHESTRATOR_PROMPT = """
You are a team orchestrator. Based on the current state and agent results,
decide which agent should execute next.

Available agents: {available_agents}
Current state: {state}
Previous results: {history}

Respond with JSON only:
{{
    "next_agent": "agent_role" | null,
    "should_continue": true | false,
    "reason": "brief explanation"
}}
"""

GROUP_ORCHESTRATOR_PROMPT = """
You are a group orchestrator managing multiple teams.
Decide which team should handle the current task.

Available teams: {available_teams}
Task: {task}
Team capabilities: {capabilities}

Respond with JSON only:
{{
    "next_team": "team_id" | null,
    "should_continue": true | false,
    "reason": "brief explanation"
}}
"""

SUMMARIZER_PROMPT = """
Summarize the following content concisely:

{content}

Provide a clear, concise summary that captures the key points.
"""

REVIEWER_PROMPT = """
Review the following content for quality and accuracy:

{content}

Provide feedback on:
1. Accuracy of information
2. Clarity of expression
3. Completeness
4. Suggestions for improvement

Respond with JSON:
{{
    "approved": true | false,
    "feedback": "detailed feedback",
    "suggestions": ["suggestion1", "suggestion2"]
}}
"""


# Factory functions


def create_orchestrator_config(
    available_agents: List[str],
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> Dict:
    """
    Create config for orchestrator agent.

    Args:
        available_agents: List of agent roles in team
        model: Model for orchestrator
        temperature: Temperature (0 = deterministic)

    Returns:
        Dict config for add_agent()
    """
    return {
        "type": "llm",
        "role": "_orchestrator",  # Prefix _ = internal agent
        "name": "Team Orchestrator",
        "description": "Decides which agent executes next",
        "prompt": ORCHESTRATOR_PROMPT,
        "model": model,
        "temperature": temperature,
        "output_format": "json",
        "metadata": {
            "available_agents": available_agents,
            "is_orchestrator": True,
        },
    }


def create_group_orchestrator_config(
    available_teams: List[str],
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> Dict:
    """
    Create config for group orchestrator.

    Args:
        available_teams: List of team_id in group
        model: Model for orchestrator
        temperature: Temperature

    Returns:
        Dict config for add_agent()
    """
    return {
        "type": "llm",
        "role": "group_orchestrator",  # v4.1: removed underscore prefix
        "name": "Group Orchestrator",
        "description": "Routes tasks between teams",
        "prompt": GROUP_ORCHESTRATOR_PROMPT,
        "model": model,
        "temperature": temperature,
        "output_format": "json",
        "metadata": {
            "available_teams": available_teams,
            "is_group_orchestrator": True,
        },
    }


def create_summarizer_config(
    role: str = "summarizer",
    model: str = "gpt-4o-mini",
    max_tokens: int = 500,
) -> Dict:
    """
    Create config for summarizer agent.

    Args:
        role: Agent role
        model: Model to use
        max_tokens: Max output tokens

    Returns:
        Dict config for add_agent()
    """
    return {
        "type": "llm",
        "role": role,
        "name": "Summarizer",
        "description": "Summarizes content concisely",
        "prompt": SUMMARIZER_PROMPT,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }


def create_reviewer_config(
    role: str = "reviewer",
    model: str = "gpt-4o-mini",
) -> Dict:
    """
    Create config for reviewer agent.

    Args:
        role: Agent role
        model: Model to use

    Returns:
        Dict config for add_agent()
    """
    return {
        "type": "llm",
        "role": role,
        "name": "Reviewer",
        "description": "Reviews content for quality",
        "prompt": REVIEWER_PROMPT,
        "model": model,
        "temperature": 0.2,
        "output_format": "json",
    }


def create_rag_config(
    role: str = "retriever",
    collection: str = "default",
    top_k: int = 5,
    **kwargs,
) -> Dict:
    """
    Create config for RAG agent.

    Args:
        role: Agent role
        collection: Vector store collection
        top_k: Number of results
        **kwargs: Additional config

    Returns:
        Dict config for add_agent()
    """
    config = {
        "type": "rag",
        "role": role,
        "name": "Document Retriever",
        "description": "Retrieves relevant documents",
        "collection": collection,
        "top_k": top_k,
    }
    config.update(kwargs)
    return config


def create_kag_config(
    role: str = "knowledge",
    max_hops: int = 2,
    max_entities: int = 10,
    **kwargs,
) -> Dict:
    """
    Create config for KAG agent.

    Args:
        role: Agent role
        max_hops: Graph traversal depth
        max_entities: Max entities to return
        **kwargs: Additional config

    Returns:
        Dict config for add_agent()
    """
    config = {
        "type": "kag",
        "role": role,
        "name": "Knowledge Graph",
        "description": "Retrieves knowledge graph entities",
        "max_hops": max_hops,
        "max_entities": max_entities,
    }
    config.update(kwargs)
    return config
