"""
External Actions - Handlers.

Provides action handlers for different action types:
- WebhookActionHandler: HTTP/REST API calls
- FunctionActionHandler: Python function calls
"""

from .webhook import WebhookActionHandler
from .function import FunctionActionHandler

__all__ = [
    "WebhookActionHandler",
    "FunctionActionHandler",
]
