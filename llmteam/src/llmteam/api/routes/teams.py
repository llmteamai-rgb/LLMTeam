"""
Teams API Routes.

RFC-020: Teams API Extension
RFC-021: Quality Simplification (quality removed from LLMTeam)

Endpoints for LLMTeam management and execution.
"""

import json
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from llmteam.api.models import (
    TeamCreateRequest,
    TeamUpdateRequest,
    TeamRunRequest,
    TeamStreamRequest,
    TeamResponse,
    TeamListResponse,
    TeamRunResponse,
    CostStatsResponse,
    BudgetSetRequest,
    BudgetResponse,
    TeamStateEnum,
)

router = APIRouter(prefix="/api/v1/teams", tags=["Teams"])


@router.post("", response_model=TeamResponse, status_code=201)
async def create_team(request: Request, body: TeamCreateRequest) -> TeamResponse:
    """
    Create a new team.

    RFC-021: Quality removed from LLMTeam. Use ConfigurationSession for quality control.

    Teams are the primary unit of work in LLMTeam. A team consists of one or more
    agents that work together to accomplish tasks.
    """
    from llmteam import LLMTeam

    if body.team_id in request.app.state.teams:
        raise HTTPException(
            status_code=409,
            detail=f"Team '{body.team_id}' already exists",
        )

    team = LLMTeam(
        team_id=body.team_id,
        max_cost_per_run=body.max_cost_per_run,
        orchestration=body.orchestration,
        enforce_lifecycle=body.enforce_lifecycle,
    )

    request.app.state.teams[body.team_id] = {
        "team": team,
        "created_at": datetime.utcnow(),
        "description": body.description,
    }

    return TeamResponse(
        team_id=team.team_id,
        max_cost_per_run=body.max_cost_per_run,
        orchestration=body.orchestration,
        enforce_lifecycle=body.enforce_lifecycle,
        state=TeamStateEnum(team.state.value if hasattr(team, "state") else "unconfigured"),
        agents_count=len(team._agents),
        description=body.description,
        created_at=request.app.state.teams[body.team_id]["created_at"],
    )


@router.get("", response_model=TeamListResponse)
async def list_teams(request: Request) -> TeamListResponse:
    """List all teams."""
    teams = []
    for team_id, data in request.app.state.teams.items():
        team = data["team"]
        teams.append(
            TeamResponse(
                team_id=team.team_id,
                max_cost_per_run=getattr(team, "_max_cost_per_run", None),
                orchestration=team.is_router_mode if hasattr(team, "is_router_mode") else False,
                enforce_lifecycle=getattr(team, "_enforce_lifecycle", False),
                state=TeamStateEnum(team.state.value if hasattr(team, "state") else "unconfigured"),
                agents_count=len(team._agents),
                description=data.get("description", ""),
                created_at=data["created_at"],
            )
        )

    return TeamListResponse(teams=teams, count=len(teams))


@router.get("/{team_id}", response_model=TeamResponse)
async def get_team(request: Request, team_id: str) -> TeamResponse:
    """Get team information."""
    data = request.app.state.teams.get(team_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found")

    team = data["team"]
    return TeamResponse(
        team_id=team.team_id,
        max_cost_per_run=getattr(team, "_max_cost_per_run", None),
        orchestration=team.is_router_mode if hasattr(team, "is_router_mode") else False,
        enforce_lifecycle=getattr(team, "_enforce_lifecycle", False),
        state=TeamStateEnum(team.state.value if hasattr(team, "state") else "unconfigured"),
        agents_count=len(team._agents),
        description=data.get("description", ""),
        created_at=data["created_at"],
    )


@router.patch("/{team_id}", response_model=TeamResponse)
async def update_team(
    request: Request,
    team_id: str,
    body: TeamUpdateRequest,
) -> TeamResponse:
    """Update team settings."""
    data = request.app.state.teams.get(team_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found")

    team = data["team"]

    if body.max_cost_per_run is not None:
        team._max_cost_per_run = body.max_cost_per_run
    if body.description is not None:
        data["description"] = body.description

    return TeamResponse(
        team_id=team.team_id,
        max_cost_per_run=getattr(team, "_max_cost_per_run", None),
        orchestration=team.is_router_mode if hasattr(team, "is_router_mode") else False,
        enforce_lifecycle=getattr(team, "_enforce_lifecycle", False),
        state=TeamStateEnum(team.state.value if hasattr(team, "state") else "unconfigured"),
        agents_count=len(team._agents),
        description=data.get("description", ""),
        created_at=data["created_at"],
    )


@router.delete("/{team_id}", status_code=204)
async def delete_team(request: Request, team_id: str) -> None:
    """Delete a team."""
    if team_id not in request.app.state.teams:
        raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found")

    del request.app.state.teams[team_id]


@router.post("/{team_id}/run", response_model=TeamRunResponse)
async def run_team(
    request: Request,
    team_id: str,
    body: TeamRunRequest,
) -> TeamRunResponse:
    """
    Run a team (synchronous).

    RFC-021: Quality/importance parameters removed. Agents use their configured models.

    Executes the team pipeline and waits for completion.
    """
    data = request.app.state.teams.get(team_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found")

    team = data["team"]
    run_id = str(uuid.uuid4())

    if body.idempotency_key:
        cached = request.app.state.team_runs.get(body.idempotency_key)
        if cached:
            return cached

    try:
        start_time = datetime.utcnow()
        result = await team.run(input_data=body.input)
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        response = TeamRunResponse(
            run_id=run_id,
            team_id=team_id,
            success=result.success if hasattr(result, "success") else True,
            status=result.status.value if hasattr(result, "status") else "completed",
            output=result.output if hasattr(result, "output") else result.final_output if hasattr(result, "final_output") else None,
            error=result.error if hasattr(result, "error") else None,
            duration_ms=duration_ms,
            cost=result.cost if hasattr(result, "cost") else None,
            tokens_used=result.tokens_used if hasattr(result, "tokens_used") else None,
            agents_executed=result.agents_executed if hasattr(result, "agents_executed") else [],
        )

        if body.idempotency_key:
            request.app.state.team_runs[body.idempotency_key] = response

        return response

    except Exception as e:
        return TeamRunResponse(
            run_id=run_id,
            team_id=team_id,
            success=False,
            status="failed",
            error=str(e),
            duration_ms=0,
        )


@router.post("/{team_id}/stream")
async def stream_team(
    request: Request,
    team_id: str,
    body: TeamStreamRequest,
) -> StreamingResponse:
    """
    Run a team with streaming output (SSE).

    RFC-021: Quality/importance parameters removed.
    """
    data = request.app.state.teams.get(team_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found")

    team = data["team"]

    async def generate_events() -> AsyncGenerator[str, None]:
        run_id = str(uuid.uuid4())

        yield _format_sse({
            "type": "run_started",
            "run_id": run_id,
            "team_id": team_id,
            "timestamp": datetime.utcnow().isoformat(),
        })

        try:
            if not hasattr(team, "stream"):
                result = await team.run(input_data=body.input)
                yield _format_sse({
                    "type": "run_completed",
                    "run_id": run_id,
                    "success": result.success if hasattr(result, "success") else True,
                    "output": result.output if hasattr(result, "output") else None,
                    "timestamp": datetime.utcnow().isoformat(),
                })
                return

            async for event in team.stream(input_data=body.input):
                event_data = {
                    "type": event.type.value if hasattr(event.type, "value") else str(event.type),
                    "run_id": run_id,
                    "timestamp": datetime.utcnow().isoformat(),
                }

                if hasattr(event, "agent_id") and event.agent_id:
                    event_data["agent_id"] = event.agent_id
                if hasattr(event, "data") and event.data:
                    event_data["data"] = event.data
                if hasattr(event, "token") and body.include_tokens:
                    event_data["token"] = event.token
                if hasattr(event, "output") and body.include_agent_outputs:
                    event_data["output"] = event.output

                yield _format_sse(event_data)

        except Exception as e:
            yield _format_sse({
                "type": "error",
                "run_id": run_id,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            })

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _format_sse(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"


@router.get("/{team_id}/cost", response_model=CostStatsResponse)
async def get_cost_stats(request: Request, team_id: str) -> CostStatsResponse:
    """Get cost statistics for a team."""
    data = request.app.state.teams.get(team_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found")

    team = data["team"]

    total_runs = 0
    total_cost = 0.0
    total_tokens = 0
    last_run_cost = None
    last_run_at = None

    if hasattr(team, "_cost_tracker") and team._cost_tracker:
        tracker = team._cost_tracker
        total_runs = tracker.total_runs if hasattr(tracker, "total_runs") else 0
        total_cost = tracker.total_cost if hasattr(tracker, "total_cost") else 0.0
        total_tokens = tracker.total_tokens if hasattr(tracker, "total_tokens") else 0
        if hasattr(tracker, "last_run"):
            last_run = tracker.last_run
            last_run_cost = last_run.cost if hasattr(last_run, "cost") else None
            last_run_at = last_run.timestamp if hasattr(last_run, "timestamp") else None

    budget_limit = getattr(team, "_max_cost_per_run", None)
    budget_remaining = None
    if budget_limit and hasattr(team, "_budget_manager") and team._budget_manager:
        budget_remaining = team._budget_manager.remaining if hasattr(team._budget_manager, "remaining") else None

    return CostStatsResponse(
        team_id=team_id,
        total_runs=total_runs,
        total_cost=total_cost,
        total_tokens=total_tokens,
        average_cost_per_run=total_cost / total_runs if total_runs > 0 else 0,
        budget_limit=budget_limit,
        budget_remaining=budget_remaining,
        last_run_cost=last_run_cost,
        last_run_at=last_run_at,
    )


@router.post("/{team_id}/budget", response_model=BudgetResponse)
async def set_budget(
    request: Request,
    team_id: str,
    body: BudgetSetRequest,
) -> BudgetResponse:
    """Set budget limits for a team."""
    data = request.app.state.teams.get(team_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found")

    team = data["team"]

    if body.max_cost_per_run is not None:
        team._max_cost_per_run = body.max_cost_per_run

    budget_data = request.app.state.team_budgets.get(team_id, {})
    if body.daily_budget is not None:
        budget_data["daily_budget"] = body.daily_budget
    if body.monthly_budget is not None:
        budget_data["monthly_budget"] = body.monthly_budget
    if body.alert_threshold is not None:
        budget_data["alert_threshold"] = body.alert_threshold
    request.app.state.team_budgets[team_id] = budget_data

    return BudgetResponse(
        team_id=team_id,
        max_cost_per_run=getattr(team, "_max_cost_per_run", None),
        daily_budget=budget_data.get("daily_budget"),
        monthly_budget=budget_data.get("monthly_budget"),
        daily_used=budget_data.get("daily_used", 0),
        monthly_used=budget_data.get("monthly_used", 0),
        alert_threshold=budget_data.get("alert_threshold"),
    )
