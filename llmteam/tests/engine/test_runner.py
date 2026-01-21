"""
Tests for Segment Runner.
"""

import pytest
import asyncio
from datetime import timedelta

from llmteam.engine import (
    SegmentDefinition,
    StepDefinition,
    EdgeDefinition,
    StepCatalog,
    StepTypeMetadata,
    StepCategory,
    PortSpec,
    SegmentRunner,
    SegmentStatus,
    SegmentResult,
    RunConfig,
)
from llmteam.runtime import RuntimeContext


# === Mock Handlers ===


async def mock_llm_handler(ctx, config, input_data):
    """Mock LLM handler that returns a simple response."""
    return {"output": f"Processed: {input_data.get('input', 'no input')}"}


async def mock_http_handler(ctx, config, input_data):
    """Mock HTTP handler that returns status."""
    return {"response": {"success": True}, "status": 200}


async def mock_transform_handler(ctx, config, input_data):
    """Mock transform handler."""
    return {"output": input_data}


async def mock_failing_handler(ctx, config, input_data):
    """Mock handler that always fails."""
    raise ValueError("Intentional failure")


async def mock_slow_handler(ctx, config, input_data):
    """Mock handler that takes time."""
    await asyncio.sleep(2)
    return {"output": "slow result"}


async def mock_retryable_handler(ctx, config, input_data):
    """Mock handler that fails first time, succeeds second time."""
    attempt = config.get("_attempt", 0)
    config["_attempt"] = attempt + 1

    if attempt < 1:
        raise ValueError("First attempt failed")
    return {"output": "retry succeeded"}


# === Fixtures ===


@pytest.fixture(autouse=True)
def reset_catalog():
    """Reset catalog singleton before each test."""
    StepCatalog.reset_instance()
    # Register handlers for built-in types
    catalog = StepCatalog.instance()
    catalog._handlers["llm_agent"] = mock_llm_handler
    catalog._handlers["http_action"] = mock_http_handler
    catalog._handlers["transform"] = mock_transform_handler
    yield
    StepCatalog.reset_instance()


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
def simple_segment():
    """Create a simple single-step segment."""
    return SegmentDefinition(
        segment_id="simple_test",
        name="Simple Test",
        entrypoint="step_1",
        steps=[
            StepDefinition(
                step_id="step_1",
                type="llm_agent",
                config={"llm_ref": "test"},
            )
        ],
    )


@pytest.fixture
def multi_step_segment():
    """Create a multi-step segment with edges."""
    return SegmentDefinition(
        segment_id="multi_step",
        name="Multi-Step Test",
        entrypoint="step_1",
        steps=[
            StepDefinition(
                step_id="step_1",
                type="llm_agent",
                config={"llm_ref": "test"},
            ),
            StepDefinition(
                step_id="step_2",
                type="transform",
            ),
        ],
        edges=[
            EdgeDefinition(from_step="step_1", to_step="step_2"),
        ],
    )


# === Tests for SegmentResult ===


class TestSegmentResult:
    """Tests for SegmentResult model."""

    def test_create(self):
        result = SegmentResult(
            run_id="run_123",
            segment_id="seg_456",
            status=SegmentStatus.COMPLETED,
        )

        assert result.run_id == "run_123"
        assert result.segment_id == "seg_456"
        assert result.status == SegmentStatus.COMPLETED

    def test_to_dict(self):
        result = SegmentResult(
            run_id="run_123",
            segment_id="seg_456",
            status=SegmentStatus.COMPLETED,
            output={"result": "success"},
            steps_completed=3,
            steps_total=3,
        )

        data = result.to_dict()

        assert data["run_id"] == "run_123"
        assert data["status"] == "completed"
        assert data["output"] == {"result": "success"}
        assert data["steps_completed"] == 3


# === Tests for RunConfig ===


class TestRunConfig:
    """Tests for RunConfig model."""

    def test_defaults(self):
        config = RunConfig()

        assert config.timeout is None
        assert config.max_retries == 3
        assert config.retry_delay == timedelta(seconds=1)
        assert config.snapshot_interval == 0

    def test_custom(self):
        config = RunConfig(
            timeout=timedelta(seconds=30),
            max_retries=5,
            retry_delay=timedelta(seconds=2),
        )

        assert config.timeout == timedelta(seconds=30)
        assert config.max_retries == 5


# === Tests for SegmentRunner ===


class TestSegmentRunner:
    """Tests for SegmentRunner."""

    async def test_run_simple_segment(self, runtime, simple_segment):
        runner = SegmentRunner()

        result = await runner.run(
            simple_segment, runtime, {"input": "test data"}
        )

        assert result.status == SegmentStatus.COMPLETED
        assert result.steps_completed == 1
        assert result.steps_total == 1
        assert "output" in result.output

    async def test_run_multi_step_segment(self, runtime, multi_step_segment):
        runner = SegmentRunner()

        result = await runner.run(
            multi_step_segment, runtime, {"input": "test data"}
        )

        assert result.status == SegmentStatus.COMPLETED
        assert result.steps_completed == 2
        assert result.steps_total == 2

    async def test_run_with_custom_handler(self, runtime):
        catalog = StepCatalog.instance()

        # Register custom handler
        async def custom_handler(ctx, cfg, inp):
            return {"result": inp.get("value", 0) * 2}

        # Register custom type
        catalog.register(
            StepTypeMetadata(
                type_id="custom",
                version="1.0",
                display_name="Custom",
                description="Custom step",
                category=StepCategory.UTILITY,
            ),
            handler=custom_handler,
        )

        segment = SegmentDefinition(
            segment_id="custom_test",
            name="Custom Test",
            entrypoint="step_1",
            steps=[StepDefinition(step_id="step_1", type="custom")],
        )

        runner = SegmentRunner()
        result = await runner.run(segment, runtime, {"value": 21})

        assert result.status == SegmentStatus.COMPLETED

    async def test_run_failure(self, runtime):
        catalog = StepCatalog.instance()
        catalog._handlers["llm_agent"] = mock_failing_handler

        segment = SegmentDefinition(
            segment_id="fail_test",
            name="Fail Test",
            entrypoint="step_1",
            steps=[
                StepDefinition(step_id="step_1", type="llm_agent", config={})
            ],
        )

        runner = SegmentRunner()
        result = await runner.run(
            segment, runtime, {}, config=RunConfig(max_retries=0)
        )

        assert result.status == SegmentStatus.FAILED
        assert result.error is not None
        assert result.error.error_type == "ValueError"

    async def test_run_with_timeout(self, runtime):
        catalog = StepCatalog.instance()
        catalog._handlers["llm_agent"] = mock_slow_handler

        segment = SegmentDefinition(
            segment_id="timeout_test",
            name="Timeout Test",
            entrypoint="step_1",
            steps=[
                StepDefinition(step_id="step_1", type="llm_agent", config={})
            ],
        )

        runner = SegmentRunner()
        result = await runner.run(
            segment,
            runtime,
            {},
            config=RunConfig(timeout=timedelta(milliseconds=100)),
        )

        assert result.status == SegmentStatus.TIMEOUT
        assert result.error is not None
        assert "timed out" in result.error.error_message.lower()

    async def test_run_with_retry(self, runtime):
        catalog = StepCatalog.instance()

        # Create a handler that tracks attempts
        attempts = []

        async def retry_handler(ctx, cfg, inp):
            attempts.append(1)
            if len(attempts) < 2:
                raise ValueError("First attempt failed")
            return {"output": "success"}

        catalog._handlers["llm_agent"] = retry_handler

        segment = SegmentDefinition(
            segment_id="retry_test",
            name="Retry Test",
            entrypoint="step_1",
            steps=[
                StepDefinition(step_id="step_1", type="llm_agent", config={})
            ],
        )

        runner = SegmentRunner()
        result = await runner.run(
            segment,
            runtime,
            {},
            config=RunConfig(max_retries=3, retry_delay=timedelta(milliseconds=10)),
        )

        assert result.status == SegmentStatus.COMPLETED
        assert len(attempts) == 2  # First failed, second succeeded

    async def test_cancel_running_segment(self, runtime):
        catalog = StepCatalog.instance()
        catalog._handlers["llm_agent"] = mock_slow_handler

        segment = SegmentDefinition(
            segment_id="cancel_test",
            name="Cancel Test",
            entrypoint="step_1",
            steps=[
                StepDefinition(step_id="step_1", type="llm_agent", config={})
            ],
        )

        runner = SegmentRunner()

        # Start segment in background
        task = asyncio.create_task(
            runner.run(segment, runtime, {})
        )

        # Wait a bit then cancel
        await asyncio.sleep(0.1)
        cancelled = await runner.cancel(runtime.run_id)

        # Wait for task to complete
        result = await task

        assert cancelled is True
        assert result.status == SegmentStatus.CANCELLED

    async def test_get_status(self, runtime):
        catalog = StepCatalog.instance()
        catalog._handlers["llm_agent"] = mock_slow_handler

        segment = SegmentDefinition(
            segment_id="status_test",
            name="Status Test",
            entrypoint="step_1",
            steps=[
                StepDefinition(step_id="step_1", type="llm_agent", config={})
            ],
        )

        runner = SegmentRunner()

        # Start segment in background
        task = asyncio.create_task(
            runner.run(segment, runtime, {})
        )

        # Wait a bit and check status
        await asyncio.sleep(0.1)
        status = await runner.get_status(runtime.run_id)
        assert status == SegmentStatus.RUNNING

        # Cancel and wait
        await runner.cancel(runtime.run_id)
        await task

        # After completion, status should be None
        status = await runner.get_status(runtime.run_id)
        assert status is None

    async def test_is_running(self, runtime):
        catalog = StepCatalog.instance()
        catalog._handlers["llm_agent"] = mock_slow_handler

        segment = SegmentDefinition(
            segment_id="running_test",
            name="Running Test",
            entrypoint="step_1",
            steps=[
                StepDefinition(step_id="step_1", type="llm_agent", config={})
            ],
        )

        runner = SegmentRunner()

        assert runner.is_running(runtime.run_id) is False

        # Start segment in background
        task = asyncio.create_task(
            runner.run(segment, runtime, {})
        )

        await asyncio.sleep(0.1)
        assert runner.is_running(runtime.run_id) is True
        assert runtime.run_id in runner.list_running()

        # Cancel and wait
        await runner.cancel(runtime.run_id)
        await task

        assert runner.is_running(runtime.run_id) is False

    async def test_step_callbacks(self, runtime, simple_segment):
        started_steps = []
        completed_steps = []

        async def on_start(step_id, input_data):
            started_steps.append(step_id)

        async def on_complete(step_id, output):
            completed_steps.append(step_id)

        config = RunConfig(
            on_step_start=on_start,
            on_step_complete=on_complete,
        )

        runner = SegmentRunner()
        result = await runner.run(
            simple_segment, runtime, {"input": "test"}, config=config
        )

        assert result.status == SegmentStatus.COMPLETED
        assert "step_1" in started_steps
        assert "step_1" in completed_steps

    async def test_no_handler_error(self, runtime):
        catalog = StepCatalog.instance()
        catalog._handlers.pop("llm_agent", None)  # Remove handler

        segment = SegmentDefinition(
            segment_id="no_handler_test",
            name="No Handler Test",
            entrypoint="step_1",
            steps=[
                StepDefinition(step_id="step_1", type="llm_agent", config={})
            ],
        )

        runner = SegmentRunner()
        result = await runner.run(segment, runtime, {})

        assert result.status == SegmentStatus.FAILED
        assert "No handler" in result.error.error_message

    async def test_timing_info(self, runtime, simple_segment):
        runner = SegmentRunner()
        result = await runner.run(simple_segment, runtime, {"input": "test"})

        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.duration_ms >= 0  # Can be 0 if very fast
        assert result.completed_at >= result.started_at


# === Tests for Edge Navigation ===


class TestEdgeNavigation:
    """Tests for edge-based step navigation."""

    async def test_simple_edge_navigation(self, runtime, multi_step_segment):
        runner = SegmentRunner()

        result = await runner.run(
            multi_step_segment, runtime, {"input": "test"}
        )

        assert result.status == SegmentStatus.COMPLETED
        assert result.steps_completed == 2

    async def test_branching_edges(self, runtime):
        """Test segment with multiple output ports and conditional edges."""
        catalog = StepCatalog.instance()

        async def condition_handler(ctx, cfg, inp):
            return {"true": inp, "false": None}

        catalog._handlers["condition"] = condition_handler

        segment = SegmentDefinition(
            segment_id="branch_test",
            name="Branch Test",
            entrypoint="condition",
            steps=[
                StepDefinition(
                    step_id="condition",
                    type="condition",
                    config={"expression": "value > 0"},
                    output_ports=["true", "false"],
                ),
                StepDefinition(step_id="handler_true", type="transform"),
                StepDefinition(step_id="handler_false", type="transform"),
            ],
            edges=[
                EdgeDefinition(
                    from_step="condition",
                    to_step="handler_true",
                    from_port="true",
                ),
            ],
        )

        runner = SegmentRunner()
        result = await runner.run(segment, runtime, {"value": 10})

        assert result.status == SegmentStatus.COMPLETED

    async def test_no_edges_single_step(self, runtime, simple_segment):
        """Single step with no edges should complete."""
        runner = SegmentRunner()
        result = await runner.run(simple_segment, runtime, {"input": "test"})

        assert result.status == SegmentStatus.COMPLETED
        assert result.steps_completed == 1


# === Tests for Input Gathering ===


class TestInputGathering:
    """Tests for step input gathering from edges."""

    async def test_entrypoint_gets_initial_input(self, runtime, simple_segment):
        runner = SegmentRunner()

        result = await runner.run(
            simple_segment, runtime, {"input": "initial data"}
        )

        assert result.status == SegmentStatus.COMPLETED
        assert "initial data" in str(result.output)

    async def test_subsequent_steps_get_previous_output(self, runtime, multi_step_segment):
        runner = SegmentRunner()

        result = await runner.run(
            multi_step_segment, runtime, {"input": "test data"}
        )

        assert result.status == SegmentStatus.COMPLETED
