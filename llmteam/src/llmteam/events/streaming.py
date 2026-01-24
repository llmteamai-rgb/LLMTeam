"""
Streaming event types and models.

RFC-011: Streaming Output (EventEmitter + SecureBus integration).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class StreamEventType(str, Enum):
    """Types of streaming events."""

    # Run lifecycle
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"

    # Agent lifecycle
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"

    # Token streaming
    TOKEN = "token"
    CHUNK = "chunk"

    # Progress
    PROGRESS = "progress"

    # Cost (RFC-010 integration)
    COST_UPDATE = "cost_update"


@dataclass
class StreamEvent:
    """
    A single streaming event emitted during team execution.

    Example:
        async for event in team.stream(input_data):
            if event.type == StreamEventType.TOKEN:
                print(event.data["token"], end="")
            elif event.type == StreamEventType.AGENT_COMPLETED:
                print(f"Agent {event.agent_id} done")
    """

    type: StreamEventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    run_id: Optional[str] = None
    agent_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (for SSE/WebSocket transport)."""
        return {
            "type": self.type.value if isinstance(self.type, StreamEventType) else self.type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "run_id": self.run_id,
            "agent_id": self.agent_id,
        }

    def to_sse(self) -> str:
        """Format as Server-Sent Event string."""
        import json

        event_type = self.type.value if isinstance(self.type, StreamEventType) else self.type
        data = json.dumps(self.to_dict())
        return f"event: {event_type}\ndata: {data}\n\n"
