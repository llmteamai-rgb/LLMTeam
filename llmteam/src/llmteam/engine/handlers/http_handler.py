"""
HTTP Action Handler.

Executes HTTP requests using the configured HTTP client.
"""

from typing import Any, Optional
import json

from llmteam.runtime import StepContext
from llmteam.observability import get_logger


logger = get_logger(__name__)


class HTTPActionHandler:
    """
    Handler for http_action step type.

    Resolves HTTP client from runtime context and executes HTTP requests.
    """

    def __init__(
        self,
        default_timeout: float = 30.0,
        default_retry_count: int = 3,
    ) -> None:
        """
        Initialize handler.

        Args:
            default_timeout: Default request timeout in seconds
            default_retry_count: Default retry count
        """
        self.default_timeout = default_timeout
        self.default_retry_count = default_retry_count

    async def __call__(
        self,
        ctx: StepContext,
        config: dict[str, Any],
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute HTTP request.

        Args:
            ctx: Step context with runtime resources
            config: Step configuration:
                - client_ref: Reference to HTTP client (required)
                - method: HTTP method (GET, POST, PUT, DELETE, PATCH)
                - path: Request path (required)
                - headers: Additional headers
                - timeout: Request timeout
                - retry_count: Number of retries
            input_data: Request body data

        Returns:
            Dict with 'response' (body) and 'status' (code)
        """
        client_ref = config.get("client_ref", "default")
        method = config.get("method", "POST").upper()
        path = config.get("path", "/")
        headers = config.get("headers", {})
        timeout = config.get("timeout", self.default_timeout)

        logger.debug(f"HTTP Action: {method} {path} via client={client_ref}")

        try:
            # Resolve HTTP client from context
            client = ctx.get_client(client_ref)

            # Get body from input_data
            body = input_data.get("body", input_data)

            # Execute request based on method
            if method == "GET":
                response = await client.get(
                    path,
                    headers=headers,
                    timeout=timeout,
                )
            elif method == "POST":
                response = await client.post(
                    path,
                    json=body,
                    headers=headers,
                    timeout=timeout,
                )
            elif method == "PUT":
                response = await client.put(
                    path,
                    json=body,
                    headers=headers,
                    timeout=timeout,
                )
            elif method == "PATCH":
                response = await client.patch(
                    path,
                    json=body,
                    headers=headers,
                    timeout=timeout,
                )
            elif method == "DELETE":
                response = await client.delete(
                    path,
                    headers=headers,
                    timeout=timeout,
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            # Extract response data
            status_code = getattr(response, "status", 200)

            # Try to get JSON response, fall back to text
            try:
                if hasattr(response, "json"):
                    response_data = await response.json() if callable(response.json) else response.json
                else:
                    response_data = response
            except Exception:
                response_data = str(response)

            logger.debug(f"HTTP Action completed: status={status_code}")

            return {
                "response": response_data,
                "status": status_code,
            }

        except Exception as e:
            logger.error(f"HTTP Action failed: {e}")
            return {
                "response": {"error": str(e)},
                "status": 500,
            }
