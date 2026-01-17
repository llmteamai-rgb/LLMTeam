"""
Tests for Canvas Handlers.
"""

import pytest
import asyncio

from llmteam.canvas import (
    HumanTaskHandler,
    create_human_task_handler,
)
from llmteam.runtime import RuntimeContext, StepContext
from llmteam.human import (
    HumanInteractionManager,
    MemoryInteractionStore,
    InteractionStatus,
)
from llmteam.tenancy import current_tenant


# === Fixtures ===


@pytest.fixture(autouse=True)
def set_tenant():
    """Set current tenant for tests."""
    token = current_tenant.set("test_tenant")
    yield
    current_tenant.reset(token)


@pytest.fixture
def runtime():
    """Create a runtime context for testing."""
    return RuntimeContext(
        tenant_id="test_tenant",
        instance_id="test_instance",
        run_id="test_run_123",
        segment_id="test_segment",
    )


@pytest.fixture
def step_context(runtime):
    """Create a step context for testing."""
    return runtime.child_context("test_step")


@pytest.fixture
def interaction_store():
    """Create a memory interaction store."""
    return MemoryInteractionStore()


@pytest.fixture
def interaction_manager(interaction_store):
    """Create a human interaction manager."""
    return HumanInteractionManager(interaction_store)


# === Tests for HumanTaskHandler ===


class TestHumanTaskHandler:
    """Tests for HumanTaskHandler."""

    def test_create_handler(self, interaction_manager):
        handler = HumanTaskHandler(manager=interaction_manager)

        assert handler.manager is interaction_manager
        assert handler.timeout_seconds == 86400

    def test_create_handler_custom_timeout(self, interaction_manager):
        handler = HumanTaskHandler(
            manager=interaction_manager, timeout_seconds=3600
        )

        assert handler.timeout_seconds == 3600

    def test_create_handler_without_manager(self):
        handler = HumanTaskHandler()

        assert handler.manager is not None
        assert isinstance(handler.manager, HumanInteractionManager)

    async def test_approval_task_approved(
        self, step_context, interaction_manager, interaction_store
    ):
        handler = HumanTaskHandler(manager=interaction_manager)

        config = {
            "task_type": "approval",
            "title": "Test Approval",
            "description": "Please approve this test",
            "assignee_ref": "test@example.com",
            "timeout_hours": 0.001,  # Very short timeout for testing
        }

        input_data = {"content": "Test content"}

        # Start handler in background
        task = asyncio.create_task(
            handler(step_context, config, input_data)
        )

        # Wait for request to be created
        await asyncio.sleep(0.01)

        # Get pending requests and approve (access store directly)
        pending = await interaction_store.list_pending("test_tenant")
        assert len(pending) == 1

        await interaction_manager.respond(
            pending[0].request_id,
            responder_id="approver@example.com",
            approved=True,
        )

        # Wait for handler to complete
        result = await task

        assert "approved" in result
        assert result["approved"]["data"] == input_data

    async def test_approval_task_rejected(
        self, step_context, interaction_manager, interaction_store
    ):
        handler = HumanTaskHandler(manager=interaction_manager)

        config = {
            "task_type": "approval",
            "title": "Test Approval",
            "timeout_hours": 0.001,
        }

        input_data = {"content": "Test content"}

        # Start handler in background
        task = asyncio.create_task(
            handler(step_context, config, input_data)
        )

        # Wait for request to be created
        await asyncio.sleep(0.01)

        # Get pending requests and reject
        pending = await interaction_store.list_pending("test_tenant")
        assert len(pending) == 1

        await interaction_manager.respond(
            pending[0].request_id,
            responder_id="reviewer@example.com",
            approved=False,
            reason="Not ready",
        )

        # Wait for handler to complete
        result = await task

        assert "rejected" in result
        assert result["rejected"]["reason"] == "Not ready"

    async def test_choice_task(self, step_context, interaction_manager, interaction_store):
        handler = HumanTaskHandler(manager=interaction_manager)

        config = {
            "task_type": "choice",
            "title": "Select Option",
            "choices": ["Option A", "Option B", "Option C"],
            "timeout_hours": 0.001,
        }

        input_data = {"context": "Choose wisely"}

        # Start handler in background
        task = asyncio.create_task(
            handler(step_context, config, input_data)
        )

        # Wait for request to be created
        await asyncio.sleep(0.01)

        # Get pending requests and respond
        pending = await interaction_store.list_pending("test_tenant")
        assert len(pending) == 1

        await interaction_manager.respond(
            pending[0].request_id,
            responder_id="chooser@example.com",
            approved=True,
            selected_option="Option B",
        )

        # Wait for handler to complete
        result = await task

        assert "approved" in result

    async def test_timeout(self, step_context, interaction_manager):
        handler = HumanTaskHandler(manager=interaction_manager)

        config = {
            "task_type": "approval",
            "title": "Test Timeout",
            "timeout_hours": 0.00001,  # Very short (0.036 seconds)
        }

        with pytest.raises(TimeoutError):
            await handler(step_context, config, {})


# === Tests for create_human_task_handler ===


class TestCreateHumanTaskHandler:
    """Tests for create_human_task_handler factory."""

    def test_create_without_manager(self):
        handler = create_human_task_handler()

        assert handler is not None
        assert isinstance(handler, HumanTaskHandler)

    def test_create_with_manager(self, interaction_manager):
        handler = create_human_task_handler(manager=interaction_manager)

        assert handler.manager is interaction_manager
