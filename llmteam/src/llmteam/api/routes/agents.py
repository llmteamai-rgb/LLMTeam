"""
Agents API Routes.

RFC-020: Teams API Extension

Endpoints for managing agents within teams.
"""

from fastapi import APIRouter, HTTPException, Request

from llmteam.api.models import (
    AgentCreateRequest,
    AgentUpdateRequest,
    AgentResponse,
    AgentListResponse,
    AgentTypeEnum,
)

router = APIRouter(prefix="/api/v1/teams/{team_id}/agents", tags=["Agents"])


@router.post("", response_model=AgentResponse, status_code=201)
async def add_agent(
    request: Request,
    team_id: str,
    body: AgentCreateRequest,
) -> AgentResponse:
    """Add an agent to a team."""
    data = request.app.state.teams.get(team_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found")

    team = data["team"]

    if body.role in team._agents:
        raise HTTPException(
            status_code=409,
            detail=f"Agent with role '{body.role}' already exists",
        )

    agent_config = {
        "role": body.role,
        "type": body.type.value,
        "prompt": body.prompt,
    }
    if body.model:
        agent_config["model"] = body.model
    if body.temperature is not None:
        agent_config["temperature"] = body.temperature
    if body.max_tokens is not None:
        agent_config["max_tokens"] = body.max_tokens
    if body.tools:
        agent_config["tools"] = body.tools
    if body.description:
        agent_config["description"] = body.description

    team.add_agent(agent_config)

    return AgentResponse(
        role=body.role,
        type=body.type,
        prompt=body.prompt,
        model=body.model,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        tools=body.tools,
        description=body.description,
    )


@router.get("", response_model=AgentListResponse)
async def list_agents(request: Request, team_id: str) -> AgentListResponse:
    """List all agents in a team."""
    data = request.app.state.teams.get(team_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found")

    team = data["team"]
    agents = []

    for role, agent in team._agents.items():
        # Get agent attributes
        agent_type = getattr(agent, "agent_type", "llm")
        prompt = getattr(agent, "prompt", getattr(agent, "prompt_template", ""))
        model = getattr(agent, "model", None)
        temperature = getattr(agent, "temperature", None)
        max_tokens = getattr(agent, "max_tokens", None)
        tools = getattr(agent, "tools", [])
        description = getattr(agent, "description", "")

        agents.append(
            AgentResponse(
                role=role,
                type=AgentTypeEnum(agent_type if agent_type in ["llm", "rag", "kag"] else "llm"),
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=[t.name if hasattr(t, "name") else str(t) for t in tools] if tools else [],
                description=description,
            )
        )

    return AgentListResponse(
        team_id=team_id,
        agents=agents,
        count=len(agents),
    )


@router.get("/{role}", response_model=AgentResponse)
async def get_agent(request: Request, team_id: str, role: str) -> AgentResponse:
    """Get agent information."""
    data = request.app.state.teams.get(team_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found")

    team = data["team"]

    if role not in team._agents:
        raise HTTPException(
            status_code=404,
            detail=f"Agent with role '{role}' not found",
        )

    agent = team._agents[role]

    agent_type = getattr(agent, "agent_type", "llm")
    prompt = getattr(agent, "prompt", getattr(agent, "prompt_template", ""))
    model = getattr(agent, "model", None)
    temperature = getattr(agent, "temperature", None)
    max_tokens = getattr(agent, "max_tokens", None)
    tools = getattr(agent, "tools", [])
    description = getattr(agent, "description", "")

    return AgentResponse(
        role=role,
        type=AgentTypeEnum(agent_type if agent_type in ["llm", "rag", "kag"] else "llm"),
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=[t.name if hasattr(t, "name") else str(t) for t in tools] if tools else [],
        description=description,
    )


@router.patch("/{role}", response_model=AgentResponse)
async def update_agent(
    request: Request,
    team_id: str,
    role: str,
    body: AgentUpdateRequest,
) -> AgentResponse:
    """Update an agent."""
    data = request.app.state.teams.get(team_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found")

    team = data["team"]

    if role not in team._agents:
        raise HTTPException(
            status_code=404,
            detail=f"Agent with role '{role}' not found",
        )

    agent = team._agents[role]

    if body.prompt is not None:
        if hasattr(agent, "prompt"):
            agent.prompt = body.prompt
        if hasattr(agent, "prompt_template"):
            agent.prompt_template = body.prompt
    if body.model is not None:
        if hasattr(agent, "model"):
            agent.model = body.model
    if body.temperature is not None:
        if hasattr(agent, "temperature"):
            agent.temperature = body.temperature
    if body.max_tokens is not None:
        if hasattr(agent, "max_tokens"):
            agent.max_tokens = body.max_tokens
    if body.description is not None:
        if hasattr(agent, "description"):
            agent.description = body.description

    # Get updated values
    agent_type = getattr(agent, "agent_type", "llm")
    prompt = getattr(agent, "prompt", getattr(agent, "prompt_template", ""))
    model = getattr(agent, "model", None)
    temperature = getattr(agent, "temperature", None)
    max_tokens = getattr(agent, "max_tokens", None)
    tools = getattr(agent, "tools", [])
    description = getattr(agent, "description", "")

    return AgentResponse(
        role=role,
        type=AgentTypeEnum(agent_type if agent_type in ["llm", "rag", "kag"] else "llm"),
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=[t.name if hasattr(t, "name") else str(t) for t in tools] if tools else [],
        description=description,
    )


@router.delete("/{role}", status_code=204)
async def remove_agent(request: Request, team_id: str, role: str) -> None:
    """Remove an agent from a team."""
    data = request.app.state.teams.get(team_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found")

    team = data["team"]

    if role not in team._agents:
        raise HTTPException(
            status_code=404,
            detail=f"Agent with role '{role}' not found",
        )

    team.remove_agent(role)
