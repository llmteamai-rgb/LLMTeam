"""
Sessions API Routes.

RFC-020: Teams API Extension
RFC-021: Quality Simplification (quality remains in ConfigurationSession)

Endpoints for configuration sessions (CONFIGURATOR mode).
"""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Request

from llmteam.api.models import (
    ConfigureRequest,
    SessionTestRequest,
    SessionResponse,
    TaskAnalysisResponse,
    SessionSuggestResponse,
    AgentSuggestionResponse,
    TestRunResponse,
    SessionStateEnum,
)

router = APIRouter(tags=["Sessions"])


@router.post(
    "/api/v1/teams/{team_id}/configure",
    response_model=SessionResponse,
    status_code=201,
)
async def start_configuration(
    request: Request,
    team_id: str,
    body: ConfigureRequest,
) -> SessionResponse:
    """
    Start a configuration session for a team.

    RFC-021: Quality is set here (design-time) and determines agent models/parameters.
    """
    data = request.app.state.teams.get(team_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found")

    team = data["team"]

    session = await team.configure(
        task=body.task,
        constraints=body.constraints,
    )

    if body.quality is not None:
        session.set_quality(body.quality)

    request.app.state.sessions[session.session_id] = {
        "session": session,
        "team_id": team_id,
        "created_at": datetime.utcnow(),
    }

    return _session_to_response(session, team_id)


@router.get("/api/v1/sessions/{session_id}", response_model=SessionResponse)
async def get_session(request: Request, session_id: str) -> SessionResponse:
    """Get configuration session information."""
    data = request.app.state.sessions.get(session_id)
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found",
        )

    return _session_to_response(data["session"], data["team_id"])


@router.post(
    "/api/v1/sessions/{session_id}/analyze",
    response_model=TaskAnalysisResponse,
)
async def analyze_task(request: Request, session_id: str) -> TaskAnalysisResponse:
    """Analyze the task in a configuration session."""
    data = request.app.state.sessions.get(session_id)
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found",
        )

    session = data["session"]
    analysis = await session.analyze()

    return TaskAnalysisResponse(
        main_goal=analysis.main_goal,
        input_type=analysis.input_type,
        output_type=analysis.output_type,
        sub_tasks=analysis.sub_tasks,
        complexity=analysis.complexity,
    )


@router.post(
    "/api/v1/sessions/{session_id}/suggest",
    response_model=SessionSuggestResponse,
)
async def get_suggestions(request: Request, session_id: str) -> SessionSuggestResponse:
    """Get agent suggestions for a configuration session."""
    data = request.app.state.sessions.get(session_id)
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found",
        )

    session = data["session"]
    await session.suggest()

    agents = [
        AgentSuggestionResponse(
            role=a.role,
            type=a.type,
            purpose=a.purpose,
            prompt_template=a.prompt_template,
            reasoning=a.reasoning,
        )
        for a in session.suggested_agents
    ]

    return SessionSuggestResponse(
        agents=agents,
        flow=session.suggested_flow,
        reasoning=session.suggestion_reasoning,
    )


@router.post(
    "/api/v1/sessions/{session_id}/test",
    response_model=TestRunResponse,
)
async def test_configuration(
    request: Request,
    session_id: str,
    body: SessionTestRequest,
) -> TestRunResponse:
    """Test the current configuration with sample data."""
    data = request.app.state.sessions.get(session_id)
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found",
        )

    session = data["session"]
    result = await session.test_run(body.input_data)

    return TestRunResponse(
        test_id=result.test_id,
        success=result.success,
        output=result.output,
        agent_outputs=result.agent_outputs,
        duration_ms=result.duration_ms,
        analysis=result.analysis,
        issues=result.issues,
        recommendations=result.recommendations,
        ready_for_production=result.ready_for_production,
    )


@router.post(
    "/api/v1/sessions/{session_id}/apply",
    response_model=SessionResponse,
)
async def apply_configuration(request: Request, session_id: str) -> SessionResponse:
    """Apply the configuration to the team."""
    data = request.app.state.sessions.get(session_id)
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found",
        )

    session = data["session"]
    await session.apply()

    return _session_to_response(session, data["team_id"])


@router.delete("/api/v1/sessions/{session_id}", status_code=204)
async def delete_session(request: Request, session_id: str) -> None:
    """Delete a configuration session."""
    if session_id not in request.app.state.sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found",
        )

    del request.app.state.sessions[session_id]


@router.post("/api/v1/sessions/{session_id}/agents")
async def add_session_agent(
    request: Request,
    session_id: str,
    body: Dict[str, Any],
) -> Dict[str, Any]:
    """Add an agent to the session configuration."""
    data = request.app.state.sessions.get(session_id)
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found",
        )

    session = data["session"]

    session.add_agent(
        role=body.get("role"),
        type=body.get("type", "llm"),
        prompt=body.get("prompt"),
        **{k: v for k, v in body.items() if k not in ["role", "type", "prompt"]},
    )

    return {"status": "added", "role": body.get("role")}


@router.patch("/api/v1/sessions/{session_id}/agents/{role}")
async def modify_session_agent(
    request: Request,
    session_id: str,
    role: str,
    body: Dict[str, Any],
) -> Dict[str, Any]:
    """Modify an agent in the session configuration."""
    data = request.app.state.sessions.get(session_id)
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found",
        )

    session = data["session"]
    session.modify_agent(role, **body)

    return {"status": "modified", "role": role}


@router.delete("/api/v1/sessions/{session_id}/agents/{role}", status_code=204)
async def remove_session_agent(
    request: Request,
    session_id: str,
    role: str,
) -> None:
    """Remove an agent from the session configuration."""
    data = request.app.state.sessions.get(session_id)
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found",
        )

    session = data["session"]
    session.remove_agent(role)


@router.put("/api/v1/sessions/{session_id}/flow")
async def set_session_flow(
    request: Request,
    session_id: str,
    body: Dict[str, Any],
) -> Dict[str, Any]:
    """Set the execution flow for the session."""
    data = request.app.state.sessions.get(session_id)
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found",
        )

    session = data["session"]
    flow = body.get("flow")

    session.set_flow(flow)

    return {"status": "updated", "flow": flow}


@router.put("/api/v1/sessions/{session_id}/quality")
async def set_session_quality(
    request: Request,
    session_id: str,
    body: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Set the quality level for the session.

    RFC-021: Quality is a design-time parameter that determines agent models/parameters.
    """
    data = request.app.state.sessions.get(session_id)
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found",
        )

    session = data["session"]
    quality = body.get("quality", 50)

    session.set_quality(quality)

    return {"status": "updated", "quality": quality}


def _session_to_response(session, team_id: str) -> SessionResponse:
    """Convert session to response model."""
    task_analysis = None
    if hasattr(session, "task_analysis") and session.task_analysis:
        task_analysis = TaskAnalysisResponse(
            main_goal=session.task_analysis.main_goal,
            input_type=session.task_analysis.input_type,
            output_type=session.task_analysis.output_type,
            sub_tasks=session.task_analysis.sub_tasks,
            complexity=session.task_analysis.complexity,
        )

    suggested_agents = []
    if hasattr(session, "suggested_agents"):
        suggested_agents = [
            AgentSuggestionResponse(
                role=a.role,
                type=a.type,
                purpose=a.purpose,
                prompt_template=a.prompt_template,
                reasoning=a.reasoning,
            )
            for a in session.suggested_agents
        ]

    return SessionResponse(
        session_id=session.session_id,
        team_id=team_id,
        task=session.task,
        state=SessionStateEnum(session.state.value if hasattr(session.state, "value") else str(session.state)),
        quality=session.quality if hasattr(session, "quality") else 50,
        task_analysis=task_analysis,
        suggested_agents=suggested_agents,
        suggested_flow=session.suggested_flow if hasattr(session, "suggested_flow") else None,
        current_agents_count=len(session.current_agents) if hasattr(session, "current_agents") else 0,
        test_runs_count=len(session.test_runs) if hasattr(session, "test_runs") else 0,
        created_at=session.created_at if hasattr(session, "created_at") else datetime.utcnow(),
    )
