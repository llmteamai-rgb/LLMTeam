"""
API Request/Response Models.

Pydantic models for the REST API layer.
"""

from datetime import datetime
from typing import Any, Optional
from enum import Enum

try:
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError(
        "pydantic is required for the API module. "
        "Install with: pip install llmteam[api]"
    )


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
