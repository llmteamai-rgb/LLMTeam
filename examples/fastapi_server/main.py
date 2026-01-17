"""
FastAPI Server Example.

REST API for running LLMTeam segments.

Usage:
    export OPENAI_API_KEY=sk-your-key
    uvicorn main:app --reload
"""

import os
import asyncio
from typing import Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LLMTeam imports
from llmteam.canvas import (
    SegmentDefinition,
    SegmentRunner,
    SegmentStatus,
    validate_segment_dict,
)
from llmteam.runtime import RuntimeContextFactory
from llmteam.events import EventEmitter, WorktrailEvent

# Try to import provider (optional)
try:
    from llmteam.providers import OpenAIProvider
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from llmteam.testing import MockLLMProvider
    HAS_MOCK = True
except ImportError:
    HAS_MOCK = False


# FastAPI app
app = FastAPI(
    title="LLMTeam API",
    description="REST API for running LLMTeam workflow segments",
    version="2.0.3",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global runtime context
runtime_factory = RuntimeContextFactory()

# WebSocket connections for events
ws_connections: list[WebSocket] = []


# Pydantic models
class SegmentInput(BaseModel):
    """Input for running a segment."""
    segment: dict[str, Any]
    input_data: dict[str, Any]
    use_mock: bool = False


class ValidationInput(BaseModel):
    """Input for validating a segment."""
    segment: dict[str, Any]


class RunResult(BaseModel):
    """Result from running a segment."""
    status: str
    output: Optional[dict[str, Any]]
    error: Optional[str]
    duration_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    llm_available: bool
    timestamp: str


# Event broadcaster
async def broadcast_event(event: WorktrailEvent) -> None:
    """Broadcast event to all connected WebSocket clients."""
    if not ws_connections:
        return

    event_data = {
        "event_type": event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type),
        "timestamp": event.timestamp.isoformat(),
        "data": event.data,
    }

    for ws in ws_connections[:]:
        try:
            await ws.send_json(event_data)
        except Exception:
            ws_connections.remove(ws)


# Custom event emitter that broadcasts to WebSockets
class BroadcastingEmitter:
    """Event emitter that broadcasts to WebSocket clients."""

    async def emit(self, event: WorktrailEvent) -> None:
        await broadcast_event(event)


@app.on_event("startup")
async def startup():
    """Initialize runtime context on startup."""
    # Register LLM provider
    if HAS_OPENAI and os.environ.get("OPENAI_API_KEY"):
        provider = OpenAIProvider(model="gpt-4o-mini")
        runtime_factory.register_llm("default", provider)
        runtime_factory.register_llm("openai", provider)
        print("OpenAI provider registered")
    elif HAS_MOCK:
        from llmteam.testing import MockLLMProvider
        mock = MockLLMProvider(responses=["Mock response from LLMTeam API"])
        runtime_factory.register_llm("default", mock)
        runtime_factory.register_llm("mock", mock)
        print("Mock provider registered (no OPENAI_API_KEY)")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    llm_available = bool(os.environ.get("OPENAI_API_KEY")) or HAS_MOCK
    return HealthResponse(
        status="healthy",
        version="2.0.3",
        llm_available=llm_available,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/segments/validate")
async def validate_segment(input: ValidationInput):
    """Validate a segment definition."""
    result = validate_segment_dict(input.segment)
    return {
        "is_valid": result.is_valid,
        "errors": [{"severity": m.severity.value, "message": m.message} for m in result.errors],
        "warnings": [{"severity": m.severity.value, "message": m.message} for m in result.warnings],
    }


@app.post("/segments/run", response_model=RunResult)
async def run_segment(input: SegmentInput):
    """Run a segment with input data."""
    start_time = datetime.now()

    try:
        # Parse segment
        segment = SegmentDefinition.from_dict(input.segment)

        # Create runtime
        runtime = runtime_factory.create_runtime(
            tenant_id="api",
            instance_id=f"run-{datetime.now().isoformat()}",
        )

        # Create runner with broadcasting emitter
        emitter = BroadcastingEmitter()
        runner = SegmentRunner(event_emitter=emitter)

        # Run segment
        result = await runner.run(
            segment=segment,
            input_data=input.input_data,
            runtime=runtime,
        )

        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        return RunResult(
            status=result.status.value if hasattr(result.status, 'value') else str(result.status),
            output=result.output,
            error=str(result.error) if result.error else None,
            duration_ms=duration_ms,
        )

    except Exception as e:
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        return RunResult(
            status="error",
            output=None,
            error=str(e),
            duration_ms=duration_ms,
        )


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """WebSocket endpoint for real-time events."""
    await websocket.accept()
    ws_connections.append(websocket)

    try:
        while True:
            # Keep connection alive, wait for messages
            data = await websocket.receive_text()
            # Echo back for ping/pong
            await websocket.send_json({"type": "pong", "data": data})
    except WebSocketDisconnect:
        ws_connections.remove(websocket)


@app.get("/segments/schema")
async def get_segment_schema():
    """Get JSON Schema for segment definition."""
    return SegmentDefinition.json_schema()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
