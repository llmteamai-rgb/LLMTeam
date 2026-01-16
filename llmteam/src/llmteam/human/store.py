"""
Human Interaction - Storage.

Provides storage for interaction requests and responses.
"""

from typing import Dict, List, Optional

from .models import InteractionRequest, InteractionResponse


class InteractionStore:
    """Base class for interaction storage."""

    async def save(self, request: InteractionRequest) -> None:
        """Save interaction request."""
        raise NotImplementedError

    async def get(self, request_id: str) -> Optional[InteractionRequest]:
        """Get interaction request by ID."""
        raise NotImplementedError

    async def update(self, request: InteractionRequest) -> None:
        """Update interaction request."""
        raise NotImplementedError

    async def list_pending(self, tenant_id: str) -> List[InteractionRequest]:
        """List pending requests for tenant."""
        raise NotImplementedError

    async def save_response(self, response: InteractionResponse) -> None:
        """Save interaction response."""
        raise NotImplementedError

    async def get_response(self, request_id: str) -> Optional[InteractionResponse]:
        """Get response for request."""
        raise NotImplementedError


class MemoryInteractionStore(InteractionStore):
    """In-memory implementation of interaction store."""

    def __init__(self) -> None:
        self._requests: Dict[str, InteractionRequest] = {}
        self._responses: Dict[str, InteractionResponse] = {}

    async def save(self, request: InteractionRequest) -> None:
        """Save interaction request."""
        self._requests[request.request_id] = request

    async def get(self, request_id: str) -> Optional[InteractionRequest]:
        """Get interaction request by ID."""
        return self._requests.get(request_id)

    async def update(self, request: InteractionRequest) -> None:
        """Update interaction request."""
        self._requests[request.request_id] = request

    async def list_pending(self, tenant_id: str) -> List[InteractionRequest]:
        """List pending requests for tenant."""
        from .models import InteractionStatus

        return [
            req
            for req in self._requests.values()
            if req.tenant_id == tenant_id and req.status == InteractionStatus.PENDING
        ]

    async def save_response(self, response: InteractionResponse) -> None:
        """Save interaction response."""
        self._responses[response.request_id] = response

    async def get_response(self, request_id: str) -> Optional[InteractionResponse]:
        """Get response for request."""
        return self._responses.get(request_id)
