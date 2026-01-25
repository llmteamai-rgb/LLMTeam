"""
API Request/Response Models.

Pydantic models for the REST API layer.

RFC-020: Teams API Extension
RFC-021: Quality Simplification (quality only in ConfigurationSession)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

try:
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError(
        "pydantic is required for the API module. "
        "Install with: pip install llmteam[api]"
    )


# ============================================================
# RFC-020: Teams API Enums
# ============================================================


class AgentTypeEnum(str, Enum):
    """Agent types."""
    llm = "llm"
    rag = "rag"
    kag = "kag"


class TeamStateEnum(str, Enum):
    """Team lifecycle states."""
    unconfigured = "unconfigured"
    configuring = "configuring"
    ready = "ready"
    running = "running"
    completed = "completed"
    failed = "failed"


class ImportanceEnum(str, Enum):
    """Task importance levels."""
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class SessionStateEnum(str, Enum):
    """Configuration session states."""
    created = "created"
    analyzing = "analyzing"
    suggesting = "suggesting"
    configuring = "configuring"
    testing = "testing"
    ready = "ready"
    applied = "applied"


class GroupRoleEnum(str, Enum):
    """Group orchestration roles."""
    report_collector = "report_collector"
    coordinator = "coordinator"
    router = "router"
    aggregator = "aggregator"
    arbiter = "arbiter"


class TeamRoleEnum(str, Enum):
    """Team roles within a group."""
    leader = "leader"
    member = "member"
    specialist = "specialist"
    fallback = "fallback"


# ============================================================
# RFC-020: Teams API Request Models
# ============================================================


class TeamCreateRequest(BaseModel):
    """Request to create a team. RFC-021: quality removed."""
    team_id: str = Field(..., description="Unique team identifier")
    max_cost_per_run: Optional[float] = Field(None, description="Budget limit per run in USD")
    orchestration: bool = Field(False, description="Enable ROUTER mode orchestration")
    enforce_lifecycle: bool = Field(False, description="Enforce lifecycle state transitions")
    description: str = Field("", description="Team description")


class TeamUpdateRequest(BaseModel):
    """Request to update team settings. RFC-021: quality removed."""
    max_cost_per_run: Optional[float] = None
    orchestration: Optional[bool] = None
    description: Optional[str] = None


class AgentCreateRequest(BaseModel):
    """Request to add an agent to a team."""
    role: str = Field(..., description="Unique agent role identifier")
    type: AgentTypeEnum = Field(AgentTypeEnum.llm, description="Agent type")
    prompt: str = Field(..., description="Agent prompt template")
    model: Optional[str] = Field(None, description="LLM model override")
    temperature: Optional[float] = Field(None, ge=0, le=2, description="Temperature")
    max_tokens: Optional[int] = Field(None, ge=1, description="Max tokens")
    tools: List[str] = Field(default_factory=list, description="Tool names")
    description: str = Field("", description="Agent description")


class AgentUpdateRequest(BaseModel):
    """Request to update an agent."""
    prompt: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1)
    tools: Optional[List[str]] = None
    description: Optional[str] = None


class TeamRunRequest(BaseModel):
    """Request to run a team. RFC-021: quality/importance removed."""
    input: Dict[str, Any] = Field(..., description="Input data for the team")
    timeout: Optional[float] = Field(None, description="Timeout in seconds")
    idempotency_key: Optional[str] = Field(None, description="Idempotency key")


class TeamStreamRequest(TeamRunRequest):
    """Request to run a team with streaming."""
    include_agent_outputs: bool = Field(True, description="Include agent outputs in events")
    include_tokens: bool = Field(False, description="Include token events")


class ConfigureRequest(BaseModel):
    """Request to start a configuration session. RFC-021: quality remains here."""
    task: str = Field(..., description="Task description for the team")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Optional constraints")
    quality: Optional[int] = Field(None, ge=0, le=100, description="Quality level 0-100")


class SessionTestRequest(BaseModel):
    """Request to test a configuration."""
    input_data: Dict[str, Any] = Field(..., description="Test input data")


class BudgetSetRequest(BaseModel):
    """Request to set budget limits."""
    max_cost_per_run: Optional[float] = None
    daily_budget: Optional[float] = None
    monthly_budget: Optional[float] = None
    alert_threshold: Optional[float] = Field(None, ge=0, le=1, description="Alert at this % of budget")


class GroupCreateRequest(BaseModel):
    """Request to create a group. RFC-021: model/temperature instead of quality."""
    group_id: Optional[str] = Field(None, description="Group ID (auto-generated if empty)")
    role: GroupRoleEnum = Field(GroupRoleEnum.report_collector, description="Group role")
    model: str = Field("gpt-4o", description="Model for routing/arbitration")
    temperature: float = Field(0.3, ge=0, le=2, description="Temperature for routing")
    max_iterations: int = Field(10, ge=1, description="Max execution iterations")


class GroupAddTeamRequest(BaseModel):
    """Request to add a team to a group."""
    team_id: str = Field(..., description="Team ID to add")
    role: TeamRoleEnum = Field(TeamRoleEnum.member, description="Team role in group")


class GroupExecuteRequest(BaseModel):
    """Request to execute a group."""
    input: Dict[str, Any] = Field(..., description="Input data")
    timeout: Optional[float] = Field(None, description="Timeout in seconds")


# ============================================================
# RFC-020: Teams API Response Models
# ============================================================


class AgentResponse(BaseModel):
    """Agent information response."""
    role: str
    type: AgentTypeEnum
    prompt: str
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: List[str] = []
    description: str = ""


class TeamResponse(BaseModel):
    """Team information response. RFC-021: quality removed."""
    team_id: str
    max_cost_per_run: Optional[float] = None
    orchestration: bool
    enforce_lifecycle: bool
    state: TeamStateEnum
    agents_count: int
    description: str = ""
    created_at: datetime


class TeamListResponse(BaseModel):
    """List of teams response."""
    teams: List[TeamResponse]
    count: int


class AgentListResponse(BaseModel):
    """List of agents response."""
    team_id: str
    agents: List[AgentResponse]
    count: int


class TeamRunResponse(BaseModel):
    """Team run result response."""
    run_id: str
    team_id: str
    success: bool
    status: str
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: int = 0
    cost: Optional[float] = None
    tokens_used: Optional[int] = None
    agents_executed: List[str] = []


class CostStatsResponse(BaseModel):
    """Cost statistics response."""
    team_id: str
    total_runs: int
    total_cost: float
    total_tokens: int
    average_cost_per_run: float
    budget_limit: Optional[float] = None
    budget_remaining: Optional[float] = None
    last_run_cost: Optional[float] = None
    last_run_at: Optional[datetime] = None


class BudgetResponse(BaseModel):
    """Budget settings response."""
    team_id: str
    max_cost_per_run: Optional[float] = None
    daily_budget: Optional[float] = None
    monthly_budget: Optional[float] = None
    daily_used: float = 0
    monthly_used: float = 0
    alert_threshold: Optional[float] = None


class TaskAnalysisResponse(BaseModel):
    """Task analysis result."""
    main_goal: str
    input_type: str
    output_type: str
    sub_tasks: List[str]
    complexity: str


class AgentSuggestionResponse(BaseModel):
    """Suggested agent configuration."""
    role: str
    type: str
    purpose: str
    prompt_template: str
    reasoning: str


class SessionSuggestResponse(BaseModel):
    """Session suggestions response."""
    agents: List[AgentSuggestionResponse]
    flow: Optional[str] = None
    reasoning: str


class TestRunResponse(BaseModel):
    """Test run result."""
    test_id: str
    success: bool
    output: Dict[str, Any]
    agent_outputs: Dict[str, Any]
    duration_ms: int
    analysis: str
    issues: List[str]
    recommendations: List[str]
    ready_for_production: bool


class SessionResponse(BaseModel):
    """Configuration session response. RFC-021: quality remains here."""
    session_id: str
    team_id: str
    task: str
    state: SessionStateEnum
    quality: int
    task_analysis: Optional[TaskAnalysisResponse] = None
    suggested_agents: List[AgentSuggestionResponse] = []
    suggested_flow: Optional[str] = None
    current_agents_count: int = 0
    test_runs_count: int = 0
    created_at: datetime


class GroupTeamResponse(BaseModel):
    """Team info within a group."""
    team_id: str
    role: TeamRoleEnum


class GroupResponse(BaseModel):
    """Group information response. RFC-021: model/temperature instead of quality."""
    group_id: str
    role: GroupRoleEnum
    model: str
    temperature: float
    max_iterations: int
    teams: List[GroupTeamResponse]
    teams_count: int
    leader_id: Optional[str] = None


class GroupListResponse(BaseModel):
    """List of groups response."""
    groups: List[GroupResponse]
    count: int


class TeamReportResponse(BaseModel):
    """Team execution report within a group."""
    team_id: str
    role: str
    success: bool
    duration_ms: int
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class GroupExecuteResponse(BaseModel):
    """Group execution result."""
    run_id: str
    group_id: str
    success: bool
    duration_ms: int
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    team_reports: List[TeamReportResponse] = []
    total_cost: Optional[float] = None


# Request Models


class RunRequest(BaseModel):
    """Request to run a segment."""

    input_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Input data for the segment entrypoint",
    )
    idempotency_key: Optional[str] = Field(
        None,
        description="Idempotency key to prevent duplicate runs",
    )
    timeout: Optional[float] = Field(
        None,
        description="Timeout in seconds",
    )


class SegmentCreateRequest(BaseModel):
    """Request to create/update a segment."""

    segment_id: str = Field(..., description="Unique segment identifier")
    name: str = Field(..., description="Human-readable name")
    description: str = Field("", description="Segment description")
    steps: list[dict[str, Any]] = Field(..., description="Step definitions")
    edges: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Edge definitions",
    )
    entrypoint: Optional[str] = Field(
        None,
        description="Entrypoint step ID (defaults to first step)",
    )


class CancelRequest(BaseModel):
    """Request to cancel a run."""

    force: bool = Field(
        False,
        description="Force cancellation without cleanup",
    )


class PauseRequest(BaseModel):
    """Request to pause a run."""

    pass


class ResumeRequest(BaseModel):
    """Request to resume a paused run."""

    snapshot_id: str = Field(..., description="Snapshot ID to resume from")
    input_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional input data overrides",
    )


# Response Models


class RunStatusEnum(str, Enum):
    """Run status values."""

    pending = "pending"
    running = "running"
    paused = "paused"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"
    timeout = "timeout"


class RunResponse(BaseModel):
    """Response for run operations."""

    run_id: str
    segment_id: str
    status: RunStatusEnum


class RunStatusResponse(BaseModel):
    """Detailed run status response."""

    run_id: str
    segment_id: str
    status: RunStatusEnum
    steps_completed: int = 0
    steps_total: int = 0
    current_step: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: int = 0
    output: Optional[dict[str, Any]] = None
    error: Optional[dict[str, Any]] = None
    resumed_from: Optional[str] = None


class SegmentResponse(BaseModel):
    """Response for segment operations."""

    segment_id: str
    name: str
    description: str
    steps_count: int
    edges_count: int
    entrypoint: str


class PauseResponse(BaseModel):
    """Response for pause operation."""

    run_id: str
    status: str
    snapshot_id: Optional[str] = None


class CatalogEntryResponse(BaseModel):
    """Single catalog entry."""

    type_id: str
    name: str
    description: str
    category: str
    supports_parallel: bool
    config_schema: dict[str, Any]
    input_ports: list[dict[str, Any]]
    output_ports: list[dict[str, Any]]


class CatalogResponse(BaseModel):
    """Full catalog response."""

    types: list[CatalogEntryResponse]
    count: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    error_type: str
    detail: Optional[str] = None
