"""
Webhook Action Handler.

Handles HTTP/REST API calls with:
- Authentication (Bearer, API Key)
- Timeout handling
- Error handling
- Correlation tracking
"""

import asyncio
from datetime import datetime
from typing import Dict

try:
    import aiohttp
except ImportError:
    aiohttp = None

from ..models import ActionConfig, ActionContext, ActionResult, ActionStatus, ActionHandler


class WebhookActionHandler(ActionHandler):
    """Handler for webhook/REST API calls."""

    def __init__(self, config: ActionConfig):
        self.config = config

        if aiohttp is None:
            raise ImportError(
                "aiohttp is required for WebhookActionHandler. "
                "Install with: pip install aiohttp"
            )

    async def execute(self, context: ActionContext) -> ActionResult:
        """Execute webhook call."""
        started_at = datetime.now()

        try:
            async with aiohttp.ClientSession() as session:
                headers = self._prepare_headers(context)

                async with session.request(
                    method=self.config.method,
                    url=self.config.url,
                    json=context.input_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                ) as response:
                    # Try to parse as JSON, fallback to text
                    try:
                        response_data = await response.json()
                    except Exception:
                        response_data = await response.text()

                    completed_at = datetime.now()
                    duration_ms = int((completed_at - started_at).total_seconds() * 1000)

                    return ActionResult(
                        action_name=self.config.name,
                        status=ActionStatus.COMPLETED if response.ok else ActionStatus.FAILED,
                        response_data=response_data,
                        response_code=response.status,
                        started_at=started_at,
                        completed_at=completed_at,
                        duration_ms=duration_ms,
                    )

        except asyncio.TimeoutError:
            completed_at = datetime.now()
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)

            return ActionResult(
                action_name=self.config.name,
                status=ActionStatus.TIMEOUT,
                error_message="Request timeout",
                error_type="TimeoutError",
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
            )

        except Exception as e:
            completed_at = datetime.now()
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)

            return ActionResult(
                action_name=self.config.name,
                status=ActionStatus.FAILED,
                error_message=str(e),
                error_type=type(e).__name__,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
            )

    def _prepare_headers(self, context: ActionContext) -> Dict[str, str]:
        """Prepare request headers with auth and metadata."""
        headers = self.config.headers.copy()
        headers["Content-Type"] = "application/json"
        headers["X-Correlation-ID"] = context.correlation_id
        headers["X-Tenant-ID"] = context.tenant_id

        # Authentication
        if self.config.auth_type == "bearer":
            token = self.config.auth_config.get("token", "")
            headers["Authorization"] = f"Bearer {token}"
        elif self.config.auth_type == "api_key":
            key_header = self.config.auth_config.get("header", "X-API-Key")
            key_value = self.config.auth_config.get("key", "")
            headers[key_header] = key_value
        elif self.config.auth_type == "basic":
            import base64

            username = self.config.auth_config.get("username", "")
            password = self.config.auth_config.get("password", "")
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {credentials}"

        return headers
