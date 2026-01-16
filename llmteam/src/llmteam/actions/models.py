"""
External Actions - Data Models.

Provides data structures for external action execution:
- ActionType, ActionStatus enums
- ActionConfig configuration
- ActionContext execution context
- ActionResult execution result
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class ActionType(Enum):
    """Type of external action."""

    WEBHOOK = "webhook"
    REST_API = "rest_api"
    GRPC = "grpc"
    DATABASE = "database"
    MESSAGE_QUEUE = "message_queue"
    FUNCTION = "function"


class ActionStatus(Enum):
    """Action execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ActionConfig:
    """Action configuration."""

    name: str
    action_type: ActionType

    # Connection
    url: Optional[str] = None
    method: str = "POST"
    headers: Dict[str, str] = field(default_factory=dict)

    # Timeouts
    timeout_seconds: float = 30.0

    # Retry (uses RateLimiter from v1.7.0)
    retry_count: int = 3

    # Validation
    request_schema: Optional[dict] = None
    response_schema: Optional[dict] = None

    # Security
    auth_type: str = ""  # "bearer", "basic", "api_key"
    auth_config: Dict[str, str] = field(default_factory=dict)


@dataclass
class ActionContext:
    """Action execution context."""

    action_name: str
    run_id: str
    agent_name: str
    tenant_id: str  # From v1.7.0 TenantContext

    # Input
    input_data: Dict[str, Any] = field(default_factory=dict)

    # State
    pipeline_state: Dict[str, Any] = field(default_factory=dict)
    agent_context: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now())
    correlation_id: str = ""


@dataclass
class ActionResult:
    """Action execution result."""

    action_name: str
    status: ActionStatus

    # Response
    response_data: Any = None
    response_code: int = 0

    # Timing
    started_at: datetime = field(default_factory=lambda: datetime.now())
    completed_at: Optional[datetime] = None
    duration_ms: int = 0

    # Errors
    error_message: str = ""
    error_type: str = ""
    retry_count: int = 0

    # Audit (for v1.7.0 AuditTrail)
    audit_metadata: Dict[str, Any] = field(default_factory=dict)


class ActionHandler:
    """Base handler for actions."""

    async def execute(self, context: ActionContext) -> ActionResult:
        """Execute the action."""
        raise NotImplementedError

    async def validate_request(self, context: ActionContext) -> bool:
        """Validate request before execution."""
        return True

    async def validate_response(self, result: ActionResult) -> bool:
        """Validate response after execution."""
        return True
