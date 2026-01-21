"""
LLMTeam REST API.

FastAPI application for the LLMTeam workflow runtime.

Usage:
    uvicorn llmteam.api:app --reload

    # Or with custom settings
    from llmteam.api import create_app
    app = create_app(runner=my_runner, catalog=my_catalog)
"""

from datetime import datetime
from typing import Any, Optional
import time

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
except ImportError:
    raise ImportError(
        "fastapi is required for the API module. "
        "Install with: pip install llmteam[api]"
    )

# Optional slowapi for rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    SLOWAPI_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False
    Limiter = None

import llmteam
from llmteam.engine import (
    WorkflowDefinition,
    StepDefinition,
    EdgeDefinition,
    ExecutionEngine,
    ExecutionResult,
    StepCatalog,
)
# Backward compatibility aliases
SegmentDefinition = WorkflowDefinition
SegmentRunner = ExecutionEngine
SegmentResult = ExecutionResult
from llmteam.runtime import RuntimeContextFactory, RuntimeContext
from llmteam.api.models import (
    RunRequest,
    SegmentCreateRequest,
    CancelRequest,
    PauseRequest,
    ResumeRequest,
    RunResponse,
    RunStatusResponse,
    RunStatusEnum,
    SegmentResponse,
    PauseResponse,
    CatalogResponse,
    CatalogEntryResponse,
    HealthResponse,
    ErrorResponse,
)


# Global state (for default app instance)
_start_time = time.time()
_segments: dict[str, SegmentDefinition] = {}
_results: dict[str, SegmentResult] = {}
_idempotency_cache: dict[str, str] = {}  # idempotency_key -> run_id


def create_app(
    runner: Optional[SegmentRunner] = None,
    catalog: Optional[StepCatalog] = None,
    runtime_manager: Optional[RuntimeContextFactory] = None,
    enable_cors: bool = True,
    cors_origins: list[str] = None,
    enable_rate_limit: bool = True,
    rate_limit: str = "100/minute",
) -> FastAPI:
    """
    Create a FastAPI application for LLMTeam.

    Args:
        runner: SegmentRunner instance (default: creates new one)
        catalog: StepCatalog instance (default: uses singleton)
        runtime_manager: RuntimeContextFactory instance (default: creates new one)
        enable_cors: Whether to enable CORS middleware
        cors_origins: Allowed CORS origins (default: ["*"])
        enable_rate_limit: Whether to enable rate limiting (requires slowapi)
        rate_limit: Rate limit string (e.g., "100/minute", "10/second")

    Returns:
        Configured FastAPI application
    """
    # OpenAPI tags for logical grouping
    tags_metadata = [
        {
            "name": "Health",
            "description": "Health check and system status endpoints",
        },
        {
            "name": "Catalog",
            "description": "Step types catalog management",
        },
        {
            "name": "Segments",
            "description": "Workflow segment definitions CRUD operations",
        },
        {
            "name": "Runs",
            "description": "Segment execution: start, monitor, pause, resume, cancel",
        },
        {
            "name": "Events",
            "description": "Real-time event streaming via WebSocket",
        },
    ]

    app = FastAPI(
        title="LLMTeam API",
        description="""
## LLMTeam Workflow Runtime API

Enterprise AI Workflow Runtime for building multi-agent LLM pipelines.

### Features
- **Segment Management**: Create, update, and validate workflow segments
- **Execution Control**: Start, pause, resume, and cancel workflow runs
- **Real-time Events**: WebSocket streaming for UI integration
- **Step Catalog**: Registry of built-in and custom step types

### Authentication
API supports JWT, API Key, and OIDC authentication (configure via middleware).

### Rate Limiting
API endpoints are rate-limited. Default: 100 requests/minute per IP.
        """,
        version=llmteam.__version__,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        openapi_tags=tags_metadata,
        license_info={
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0",
        },
        contact={
            "name": "LLMTeam Support",
            "url": "https://github.com/llmteamai-rgb/LLMTeam",
        },
    )

    # Store instances in app state
    app.state.runner = runner or SegmentRunner()
    app.state.catalog = catalog or StepCatalog.instance()
    app.state.runtime_manager = runtime_manager or RuntimeContextFactory()
    app.state.start_time = time.time()
    app.state.segments = {}
    app.state.results = {}
    app.state.idempotency_cache = {}
    app.state.rate_limit = rate_limit

    # Rate limiting (optional, requires slowapi)
    if enable_rate_limit and SLOWAPI_AVAILABLE:
        limiter = Limiter(key_func=get_remote_address, default_limits=[rate_limit])
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    else:
        app.state.limiter = None

    # CORS
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins or ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Register routes
    _register_routes(app, enable_rate_limit and SLOWAPI_AVAILABLE)

    # Register WebSocket routes
    from llmteam.api.websocket import register_websocket_routes
    register_websocket_routes(app)

    return app


def _register_routes(app: FastAPI, use_rate_limit: bool = False) -> None:
    """Register API routes."""

    # Helper to get limiter decorator
    def get_limit_decorator(limit: str = None):
        if use_rate_limit and app.state.limiter:
            return app.state.limiter.limit(limit or app.state.rate_limit)
        return lambda f: f  # No-op decorator

    # Health check (no rate limit - always available)
    @app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version=llmteam.__version__,
            uptime_seconds=time.time() - app.state.start_time,
        )

    # Catalog
    @app.get("/api/v1/catalog", response_model=CatalogResponse, tags=["Catalog"])
    async def get_catalog() -> CatalogResponse:
        """Get step types catalog."""
        catalog: StepCatalog = app.state.catalog
        types = catalog.list_all()

        entries = []
        for type_meta in types:
            entries.append(
                CatalogEntryResponse(
                    type_id=type_meta.type_id,
                    name=type_meta.name,
                    description=type_meta.description,
                    category=type_meta.category.value,
                    supports_parallel=type_meta.supports_parallel,
                    config_schema=type_meta.config_schema,
                    input_ports=[p.to_dict() for p in type_meta.input_ports],
                    output_ports=[p.to_dict() for p in type_meta.output_ports],
                )
            )

        return CatalogResponse(types=entries, count=len(entries))

    # Segments
    @app.post(
        "/api/v1/segments",
        response_model=SegmentResponse,
        tags=["Segments"],
        status_code=201,
    )
    async def create_segment(request: SegmentCreateRequest) -> SegmentResponse:
        """Create or update a segment definition."""
        # Build segment
        steps = [StepDefinition.from_dict(s) for s in request.steps]
        edges = [EdgeDefinition.from_dict(e) for e in request.edges]

        segment = SegmentDefinition(
            segment_id=request.segment_id,
            name=request.name,
            description=request.description,
            steps=steps,
            edges=edges,
            entrypoint=request.entrypoint or (steps[0].step_id if steps else None),
        )

        # Validate
        errors = segment.validate()
        if errors:
            raise HTTPException(
                status_code=400,
                detail={"error": "Validation failed", "errors": errors},
            )

        # Store
        app.state.segments[segment.segment_id] = segment

        return SegmentResponse(
            segment_id=segment.segment_id,
            name=segment.name,
            description=segment.description,
            steps_count=len(segment.steps),
            edges_count=len(segment.edges),
            entrypoint=segment.entrypoint or "",
        )

    @app.get(
        "/api/v1/segments/{segment_id}",
        response_model=SegmentResponse,
        tags=["Segments"],
    )
    async def get_segment(segment_id: str) -> SegmentResponse:
        """Get a segment definition."""
        segment = app.state.segments.get(segment_id)
        if not segment:
            raise HTTPException(status_code=404, detail="Segment not found")

        return SegmentResponse(
            segment_id=segment.segment_id,
            name=segment.name,
            description=segment.description,
            steps_count=len(segment.steps),
            edges_count=len(segment.edges),
            entrypoint=segment.entrypoint or "",
        )

    # Runs
    @app.post(
        "/api/v1/segments/{segment_id}/runs",
        response_model=RunResponse,
        tags=["Runs"],
        status_code=202,
    )
    @get_limit_decorator("30/minute")  # Stricter limit for execution
    async def start_run(
        request_obj: Request,
        segment_id: str,
        request: RunRequest,
        background: BackgroundTasks,
    ) -> RunResponse:
        """Start a segment run."""
        # Check idempotency
        if request.idempotency_key:
            existing_run_id = app.state.idempotency_cache.get(request.idempotency_key)
            if existing_run_id:
                return RunResponse(
                    run_id=existing_run_id,
                    segment_id=segment_id,
                    status=RunStatusEnum.running,
                )

        # Get segment
        segment = app.state.segments.get(segment_id)
        if not segment:
            raise HTTPException(status_code=404, detail="Segment not found")

        # Create runtime
        runtime_manager: RuntimeContextFactory = app.state.runtime_manager
        runtime = runtime_manager.create_runtime(
            tenant_id="default",
            instance_id=None,  # Auto-generate
        )

        # Cache idempotency key
        if request.idempotency_key:
            app.state.idempotency_cache[request.idempotency_key] = runtime.run_id

        # Start run in background
        background.add_task(
            _run_segment,
            app,
            segment,
            runtime,
            request.input_data,
            request.timeout,
        )

        return RunResponse(
            run_id=runtime.run_id,
            segment_id=segment_id,
            status=RunStatusEnum.running,
        )

    @app.get(
        "/api/v1/runs/{run_id}",
        response_model=RunStatusResponse,
        tags=["Runs"],
    )
    async def get_run_status(run_id: str) -> RunStatusResponse:
        """Get run status."""
        runner: SegmentRunner = app.state.runner

        # Check if running
        if runner.is_running(run_id):
            # Get partial info from run state
            run_state = runner._run_state.get(run_id)
            if run_state:
                return RunStatusResponse(
                    run_id=run_id,
                    segment_id=run_state.segment.segment_id,
                    status=RunStatusEnum.running,
                    steps_completed=run_state.result.steps_completed,
                    steps_total=run_state.result.steps_total,
                    current_step=run_state.current_step,
                    started_at=run_state.result.started_at,
                )

        # Check if paused
        if runner.is_paused(run_id):
            return RunStatusResponse(
                run_id=run_id,
                segment_id="",
                status=RunStatusEnum.paused,
            )

        # Check completed results
        result = app.state.results.get(run_id)
        if result:
            return RunStatusResponse(
                run_id=result.run_id,
                segment_id=result.segment_id,
                status=RunStatusEnum(result.status.value),
                steps_completed=result.steps_completed,
                steps_total=result.steps_total,
                current_step=result.current_step,
                started_at=result.started_at,
                completed_at=result.completed_at,
                duration_ms=result.duration_ms,
                output=result.output,
                error=result.error.to_dict() if result.error else None,
                resumed_from=result.resumed_from,
            )

        raise HTTPException(status_code=404, detail="Run not found")

    @app.post(
        "/api/v1/runs/{run_id}/cancel",
        response_model=RunResponse,
        tags=["Runs"],
    )
    async def cancel_run(run_id: str, request: CancelRequest) -> RunResponse:
        """Cancel a running segment."""
        runner: SegmentRunner = app.state.runner

        success = await runner.cancel(run_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Run not found or already completed",
            )

        return RunResponse(
            run_id=run_id,
            segment_id="",
            status=RunStatusEnum.cancelled,
        )

    @app.post(
        "/api/v1/runs/{run_id}/pause",
        response_model=PauseResponse,
        tags=["Runs"],
    )
    async def pause_run(run_id: str, request: PauseRequest) -> PauseResponse:
        """Pause a running segment."""
        runner: SegmentRunner = app.state.runner

        snapshot_id = await runner.pause(run_id)
        if not snapshot_id:
            raise HTTPException(
                status_code=404,
                detail="Run not found or already completed",
            )

        return PauseResponse(
            run_id=run_id,
            status="paused",
            snapshot_id=snapshot_id,
        )

    @app.post(
        "/api/v1/runs/{run_id}/resume",
        response_model=RunResponse,
        tags=["Runs"],
        status_code=202,
    )
    async def resume_run(
        run_id: str,
        request: ResumeRequest,
        background: BackgroundTasks,
    ) -> RunResponse:
        """Resume a paused segment."""
        runner: SegmentRunner = app.state.runner
        runtime_manager: RuntimeContextFactory = app.state.runtime_manager

        # Create new runtime for resumed run
        runtime = runtime_manager.create_runtime(
            tenant_id="default",
            instance_id=None,
        )

        # Start resume in background
        background.add_task(
            _resume_segment,
            app,
            runner,
            request.snapshot_id,
            runtime,
        )

        return RunResponse(
            run_id=runtime.run_id,
            segment_id="",
            status=RunStatusEnum.running,
        )


async def _run_segment(
    app: FastAPI,
    segment: SegmentDefinition,
    runtime: RuntimeContext,
    input_data: dict[str, Any],
    timeout: Optional[float],
) -> None:
    """Background task to run a segment."""
    from datetime import timedelta

    runner: SegmentRunner = app.state.runner

    from llmteam.engine import RunConfig

    config = RunConfig(
        timeout=timedelta(seconds=timeout) if timeout else None,
    )

    result = await runner.run(
        segment=segment,
        runtime=runtime,
        input_data=input_data,
        config=config,
    )

    # Store result
    app.state.results[result.run_id] = result


async def _resume_segment(
    app: FastAPI,
    runner: SegmentRunner,
    snapshot_id: str,
    runtime: RuntimeContext,
) -> None:
    """Background task to resume a segment."""
    try:
        result = await runner.resume(
            snapshot_id=snapshot_id,
            runtime=runtime,
        )
        app.state.results[result.run_id] = result
    except Exception as e:
        # Log error
        print(f"Resume failed: {e}")


# Default app instance
app = create_app()
