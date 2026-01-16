"""
Tests for Human Interaction module (v1.9.0).
"""

import pytest
from datetime import timedelta

from llmteam.human import (
    InteractionType,
    InteractionStatus,
    InteractionPriority,
    InteractionRequest,
    InteractionResponse,
    HumanInteractionManager,
    MemoryInteractionStore,
)


class TestInteractionModels:
    """Tests for interaction data models."""

    def test_interaction_request(self):
        """Test InteractionRequest creation."""
        request = InteractionRequest(
            request_id="req_123",
            interaction_type=InteractionType.APPROVAL,
            run_id="run_123",
            pipeline_id="pipeline_1",
            agent_name="agent_1",
            step_name="approval",
            tenant_id="tenant_1",
            title="Approve deployment",
            description="Deploy to production?",
        )

        assert request.request_id == "req_123"
        assert request.interaction_type == InteractionType.APPROVAL
        assert request.title == "Approve deployment"

    def test_interaction_response(self):
        """Test InteractionResponse creation."""
        response = InteractionResponse(
            request_id="req_123",
            responder_id="user@example.com",
            approved=True,
            comment="Looks good",
        )

        assert response.request_id == "req_123"
        assert response.approved is True


class TestMemoryInteractionStore:
    """Tests for MemoryInteractionStore."""

    @pytest.mark.asyncio
    async def test_save_and_get(self):
        """Test saving and retrieving interaction request."""
        store = MemoryInteractionStore()

        request = InteractionRequest(
            request_id="req_123",
            interaction_type=InteractionType.APPROVAL,
            run_id="run_123",
            pipeline_id="pipeline_1",
            agent_name="agent_1",
            step_name="approval",
            tenant_id="tenant_1",
            title="Test",
            description="Test",
        )

        await store.save(request)
        retrieved = await store.get("req_123")

        assert retrieved is not None
        assert retrieved.request_id == "req_123"


class TestHumanInteractionManager:
    """Tests for HumanInteractionManager."""

    @pytest.fixture
    def manager(self):
        """Create manager with memory store."""
        store = MemoryInteractionStore()
        return HumanInteractionManager(store)

    @pytest.mark.asyncio
    async def test_request_approval(self, manager):
        """Test requesting approval."""
        request = await manager.request_approval(
            title="Approve deployment",
            description="Deploy v1.2.3?",
            run_id="run_123",
            pipeline_id="pipeline_1",
            agent_name="agent_1",
        )

        assert request.interaction_type == InteractionType.APPROVAL
        assert request.title == "Approve deployment"
        assert request.status == InteractionStatus.PENDING

    @pytest.mark.asyncio
    async def test_respond(self, manager):
        """Test responding to interaction request."""
        # Request approval
        request = await manager.request_approval(
            title="Test",
            description="Test",
            run_id="run_123",
            pipeline_id="pipeline_1",
            agent_name="agent_1",
        )

        # Respond
        response = await manager.respond(
            request.request_id,
            responder_id="user@example.com",
            approved=True,
            comment="Approved",
        )

        assert response.request_id == request.request_id
        assert response.approved is True

        # Check request status updated
        updated_request = await manager.store.get(request.request_id)
        assert updated_request.status == InteractionStatus.COMPLETED
