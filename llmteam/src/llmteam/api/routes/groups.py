"""
Groups API Routes.

RFC-020: Teams API Extension
RFC-021: Quality Simplification (model/temperature instead of quality)

Endpoints for multi-team group orchestration.
"""

from datetime import datetime
from typing import Optional
import uuid

from fastapi import APIRouter, HTTPException, Request

from llmteam.api.models import (
    GroupCreateRequest,
    GroupAddTeamRequest,
    GroupExecuteRequest,
    GroupResponse,
    GroupListResponse,
    GroupTeamResponse,
    GroupExecuteResponse,
    TeamReportResponse,
    GroupRoleEnum,
    TeamRoleEnum,
)

router = APIRouter(prefix="/api/v1/groups", tags=["Groups"])


@router.post("", response_model=GroupResponse, status_code=201)
async def create_group(request: Request, body: GroupCreateRequest) -> GroupResponse:
    """
    Create a new group.

    RFC-021: Uses explicit model/temperature instead of quality.
    """
    from llmteam.orchestration import GroupOrchestrator, GroupRole

    group_id = body.group_id
    if not group_id:
        group_id = f"group_{uuid.uuid4().hex[:8]}"

    if group_id in request.app.state.groups:
        raise HTTPException(
            status_code=409,
            detail=f"Group '{group_id}' already exists",
        )

    group = GroupOrchestrator(
        group_id=group_id,
        role=GroupRole(body.role.value),
        model=body.model,
        temperature=body.temperature,
        max_iterations=body.max_iterations,
    )

    request.app.state.groups[group_id] = {
        "group": group,
        "created_at": datetime.utcnow(),
    }

    return _group_to_response(group)


@router.get("", response_model=GroupListResponse)
async def list_groups(request: Request) -> GroupListResponse:
    """List all groups."""
    groups = []
    for group_id, data in request.app.state.groups.items():
        groups.append(_group_to_response(data["group"]))

    return GroupListResponse(groups=groups, count=len(groups))


@router.get("/{group_id}", response_model=GroupResponse)
async def get_group(request: Request, group_id: str) -> GroupResponse:
    """Get group information."""
    data = request.app.state.groups.get(group_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Group '{group_id}' not found")

    return _group_to_response(data["group"])


@router.delete("/{group_id}", status_code=204)
async def delete_group(request: Request, group_id: str) -> None:
    """Delete a group."""
    if group_id not in request.app.state.groups:
        raise HTTPException(status_code=404, detail=f"Group '{group_id}' not found")

    del request.app.state.groups[group_id]


@router.post("/{group_id}/teams", response_model=GroupResponse)
async def add_team_to_group(
    request: Request,
    group_id: str,
    body: GroupAddTeamRequest,
) -> GroupResponse:
    """Add a team to a group."""
    from llmteam.orchestration import TeamRole

    data = request.app.state.groups.get(group_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Group '{group_id}' not found")

    team_data = request.app.state.teams.get(body.team_id)
    if not team_data:
        raise HTTPException(status_code=404, detail=f"Team '{body.team_id}' not found")

    group = data["group"]
    team = team_data["team"]

    group.add_team(team, role=TeamRole(body.role.value))

    return _group_to_response(group)


@router.delete("/{group_id}/teams/{team_id}", status_code=204)
async def remove_team_from_group(
    request: Request,
    group_id: str,
    team_id: str,
) -> None:
    """Remove a team from a group."""
    data = request.app.state.groups.get(group_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Group '{group_id}' not found")

    group = data["group"]

    if team_id not in group._teams:
        raise HTTPException(
            status_code=404,
            detail=f"Team '{team_id}' not in group",
        )

    del group._teams[team_id]
    if team_id in group._team_roles:
        del group._team_roles[team_id]


@router.post("/{group_id}/execute", response_model=GroupExecuteResponse)
async def execute_group(
    request: Request,
    group_id: str,
    body: GroupExecuteRequest,
) -> GroupExecuteResponse:
    """Execute a group."""
    data = request.app.state.groups.get(group_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Group '{group_id}' not found")

    group = data["group"]
    run_id = str(uuid.uuid4())

    if len(group._teams) == 0:
        raise HTTPException(
            status_code=400,
            detail="Group has no teams. Add teams before executing.",
        )

    try:
        start_time = datetime.utcnow()
        result = await group.execute(body.input)
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        team_reports = []
        if hasattr(result, "report") and result.report:
            for team_report in result.report.team_reports:
                team_reports.append(
                    TeamReportResponse(
                        team_id=team_report.team_id,
                        role=team_report.team_role if hasattr(team_report, "team_role") else "member",
                        success=team_report.success,
                        duration_ms=team_report.duration_ms,
                        output=team_report.output,
                        error=team_report.error,
                    )
                )

        return GroupExecuteResponse(
            run_id=run_id,
            group_id=group_id,
            success=result.success if hasattr(result, "success") else True,
            duration_ms=duration_ms,
            output=result.output if hasattr(result, "output") else None,
            error=result.error if hasattr(result, "error") else None,
            team_reports=team_reports,
            total_cost=result.total_cost if hasattr(result, "total_cost") else None,
        )

    except Exception as e:
        return GroupExecuteResponse(
            run_id=run_id,
            group_id=group_id,
            success=False,
            duration_ms=0,
            error=str(e),
        )


def _group_to_response(group) -> GroupResponse:
    """Convert group to response model."""
    teams = []
    for team_id, team in group._teams.items():
        role = group._team_roles.get(team_id)
        teams.append(
            GroupTeamResponse(
                team_id=team_id,
                role=TeamRoleEnum(role.value if role and hasattr(role, "value") else "member"),
            )
        )

    return GroupResponse(
        group_id=group.group_id,
        role=GroupRoleEnum(group._role.value if hasattr(group._role, "value") else str(group._role)),
        model=getattr(group, "_model", "gpt-4o"),
        temperature=getattr(group, "_temperature", 0.3),
        max_iterations=getattr(group, "_max_iterations", 10),
        teams=teams,
        teams_count=len(teams),
        leader_id=getattr(group, "_leader_id", None),
    )
