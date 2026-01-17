"""
WebSocket support for real-time events.

Provides WebSocket endpoint for streaming Worktrail events to connected clients.
"""

from typing import Any, Optional
import asyncio
import json

try:
    from fastapi import WebSocket, WebSocketDisconnect
except ImportError:
    raise ImportError(
        "fastapi is required for the API module. "
        "Install with: pip install llmteam[api]"
    )

from llmteam.events import WorktrailEvent, EventEmitter


class ConnectionManager:
    """
    Manages WebSocket connections for run event streaming.

    Supports:
    - Multiple clients per run
    - Broadcast to all clients watching a run
    - Automatic cleanup on disconnect
    """

    def __init__(self) -> None:
        # run_id -> set of connected websockets
        self._connections: dict[str, set[WebSocket]] = {}
        # websocket -> run_id (for reverse lookup)
        self._websocket_runs: dict[WebSocket, str] = {}

    async def connect(self, websocket: WebSocket, run_id: str) -> None:
        """
        Accept a WebSocket connection for a run.

        Args:
            websocket: The WebSocket connection
            run_id: The run ID to watch
        """
        await websocket.accept()

        if run_id not in self._connections:
            self._connections[run_id] = set()

        self._connections[run_id].add(websocket)
        self._websocket_runs[websocket] = run_id

    async def disconnect(self, websocket: WebSocket) -> None:
        """
        Handle WebSocket disconnection.

        Args:
            websocket: The disconnected WebSocket
        """
        run_id = self._websocket_runs.pop(websocket, None)
        if run_id and run_id in self._connections:
            self._connections[run_id].discard(websocket)
            # Clean up empty sets
            if not self._connections[run_id]:
                del self._connections[run_id]

    async def broadcast(self, run_id: str, data: dict[str, Any]) -> None:
        """
        Broadcast data to all clients watching a run.

        Args:
            run_id: The run ID
            data: Data to send (will be JSON serialized)
        """
        connections = self._connections.get(run_id, set())

        # Send to all connections, handling failures
        disconnected = []
        for websocket in connections:
            try:
                await websocket.send_json(data)
            except Exception:
                disconnected.append(websocket)

        # Clean up disconnected sockets
        for ws in disconnected:
            await self.disconnect(ws)

    async def broadcast_event(self, event: WorktrailEvent) -> None:
        """
        Broadcast a WorktrailEvent to clients.

        Args:
            event: The event to broadcast
        """
        await self.broadcast(event.run_id, event.to_dict())

    def get_connection_count(self, run_id: str) -> int:
        """Get number of connections for a run."""
        return len(self._connections.get(run_id, set()))

    def get_total_connections(self) -> int:
        """Get total number of connections."""
        return sum(len(conns) for conns in self._connections.values())

    def get_watched_runs(self) -> list[str]:
        """Get list of run IDs being watched."""
        return list(self._connections.keys())


# Global connection manager
connection_manager = ConnectionManager()


class WebSocketEventEmitter(EventEmitter):
    """
    EventEmitter that also broadcasts events to WebSocket clients.

    Use this instead of the base EventEmitter to enable real-time updates.

    Example:
        manager = ConnectionManager()
        emitter = WebSocketEventEmitter(runtime, manager)

        # Events will be sent to both EventStream and WebSocket clients
        emitter.step_started("step_1", "llm_agent", {"input": data})
    """

    def __init__(
        self,
        runtime: Any,
        connection_manager: Optional[ConnectionManager] = None,
    ) -> None:
        """
        Initialize WebSocket-enabled EventEmitter.

        Args:
            runtime: Runtime context
            connection_manager: Connection manager (default: global)
        """
        super().__init__(runtime)
        self._ws_manager = connection_manager or globals()["connection_manager"]

    def _emit(self, event: WorktrailEvent) -> None:
        """Emit event to stream and WebSocket."""
        # Call parent to emit to stream
        super()._emit(event)

        # Also broadcast to WebSocket (in background)
        asyncio.create_task(self._ws_manager.broadcast_event(event))


def register_websocket_routes(app: Any) -> None:
    """
    Register WebSocket routes on a FastAPI app.

    Args:
        app: FastAPI application

    Routes added:
        GET /api/v1/runs/{run_id}/ws - WebSocket endpoint for run events
        GET /api/v1/ws/stats - Get WebSocket connection stats
    """

    @app.websocket("/api/v1/runs/{run_id}/ws")
    async def websocket_run_events(websocket: WebSocket, run_id: str) -> None:
        """
        WebSocket endpoint for streaming run events.

        Connect to receive real-time events for a specific run.
        The connection stays open until the run completes or client disconnects.

        Events are sent as JSON with the following structure:
        {
            "event_id": "...",
            "event_type": "step_started|step_completed|...",
            "run_id": "...",
            "step_id": "...",
            "timestamp": "...",
            "payload": {...}
        }
        """
        await connection_manager.connect(websocket, run_id)
        try:
            while True:
                # Keep connection alive
                # Client can send ping/pong or just wait
                data = await websocket.receive_text()

                # Handle ping
                if data == "ping":
                    await websocket.send_text("pong")

        except WebSocketDisconnect:
            await connection_manager.disconnect(websocket)

    @app.get("/api/v1/ws/stats", tags=["WebSocket"])
    async def websocket_stats() -> dict[str, Any]:
        """Get WebSocket connection statistics."""
        return {
            "total_connections": connection_manager.get_total_connections(),
            "watched_runs": connection_manager.get_watched_runs(),
            "connections_per_run": {
                run_id: connection_manager.get_connection_count(run_id)
                for run_id in connection_manager.get_watched_runs()
            },
        }
