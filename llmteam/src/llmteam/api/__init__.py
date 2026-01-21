"""
LLMTeam REST API Module.

Provides a FastAPI-based REST API for the LLMTeam workflow runtime.

Installation:
    pip install llmteam[api]

Usage:
    # Run the default app
    uvicorn llmteam.api:app --reload --port 8000

    # Create a custom app
    from llmteam.api import create_app
    from llmteam.engine import ExecutionEngine, StepCatalog

    engine = ExecutionEngine()
    app = create_app(runner=engine)

Endpoints:
    GET  /api/v1/health              - Health check
    GET  /api/v1/catalog             - Get step types catalog
    POST /api/v1/segments            - Create/update segment
    GET  /api/v1/segments/{id}       - Get segment
    POST /api/v1/segments/{id}/runs  - Start run
    GET  /api/v1/runs/{id}           - Get run status
    POST /api/v1/runs/{id}/cancel    - Cancel run
    POST /api/v1/runs/{id}/pause     - Pause run
    POST /api/v1/runs/{id}/resume    - Resume run
"""

try:
    from llmteam.api.app import app, create_app
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
    from llmteam.api.websocket import (
        ConnectionManager,
        WebSocketEventEmitter,
        connection_manager,
        register_websocket_routes,
    )

    __all__ = [
        # App
        "app",
        "create_app",
        # Request models
        "RunRequest",
        "SegmentCreateRequest",
        "CancelRequest",
        "PauseRequest",
        "ResumeRequest",
        # Response models
        "RunResponse",
        "RunStatusResponse",
        "RunStatusEnum",
        "SegmentResponse",
        "PauseResponse",
        "CatalogResponse",
        "CatalogEntryResponse",
        "HealthResponse",
        "ErrorResponse",
        # WebSocket
        "ConnectionManager",
        "WebSocketEventEmitter",
        "connection_manager",
        "register_websocket_routes",
    ]
except ImportError as e:
    # FastAPI not installed
    def _raise_import_error(*args, **kwargs):
        raise ImportError(
            "FastAPI is required for the API module. "
            "Install with: pip install llmteam[api]"
        )

    app = None
    create_app = _raise_import_error

    __all__ = ["app", "create_app"]
