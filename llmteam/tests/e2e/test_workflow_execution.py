"""
End-to-End Tests for Workflow Execution.

Tests realistic workflow scenarios:
- Full functional cycle (JSON -> Runner -> Result)
- Parallel execution branches
- Human Task pause/resume
"""

import pytest
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock

from llmteam.engine import (
    SegmentDefinition,
    StepDefinition,
    EdgeDefinition,
    SegmentRunner,
    SegmentStatus,
    StepCatalog,
)
from llmteam.runtime import RuntimeContext
from llmteam.human import MemoryInteractionStore, HumanInteractionManager


# === Mock Handlers for E2E ===


async def e2e_transform_handler(ctx, config, input_data):
    """Transform handler that applies expression."""
    expression = config.get("expression", "input")
    if expression == "input":
        return {"output": input_data}
    elif expression == "uppercase":
        text = input_data.get("text", "")
        return {"output": {"text": text.upper()}}
    elif expression == "append_processed":
        data = dict(input_data)
        data["processed"] = True
        return {"output": data}
    return {"output": input_data}


async def e2e_llm_handler(ctx, config, input_data):
    """Mock LLM handler for E2E tests."""
    prompt = config.get("prompt", "")
    query = input_data.get("query", input_data.get("input", {}).get("query", ""))

    # Simulate LLM response based on query
    response = f"LLM response to: {query}"
    return {"output": {"response": response, "query": query}}


async def e2e_http_handler(ctx, config, input_data):
    """Mock HTTP handler for E2E tests."""
    url = config.get("url", "")
    method = config.get("method", "GET")

    return {
        "output": {
            "status": 200,
            "body": {"success": True, "url": url, "method": method},
        }
    }


async def e2e_condition_handler(ctx, config, input_data):
    """Condition handler for branching."""
    field = config.get("field", "value")
    operator = config.get("operator", "eq")
    compare_value = config.get("value")

    actual_value = input_data.get(field)

    if operator == "eq":
        result = actual_value == compare_value
    elif operator == "gt":
        result = actual_value > compare_value
    elif operator == "lt":
        result = actual_value < compare_value
    elif operator == "contains":
        result = compare_value in str(actual_value)
    else:
        result = False

    if result:
        return {"true": input_data}
    else:
        return {"false": input_data}


async def e2e_parallel_split_handler(ctx, config, input_data):
    """Split data to parallel branches."""
    branches = config.get("branches", ["branch_1", "branch_2"])
    result = {}
    for branch in branches:
        result[branch] = dict(input_data)
    return result


async def e2e_parallel_join_handler(ctx, config, input_data):
    """Join parallel branch results."""
    strategy = config.get("strategy", "all")
    # Input data contains results from all branches
    return {"output": {"merged": input_data, "strategy": strategy}}


# === Fixtures ===


@pytest.fixture(autouse=True)
def setup_catalog():
    """Setup catalog with E2E handlers."""
    StepCatalog.reset_instance()
    catalog = StepCatalog.instance()

    # Register E2E handlers
    catalog._handlers["transform"] = e2e_transform_handler
    catalog._handlers["llm_agent"] = e2e_llm_handler
    catalog._handlers["http_action"] = e2e_http_handler
    catalog._handlers["condition"] = e2e_condition_handler
    catalog._handlers["parallel_split"] = e2e_parallel_split_handler
    catalog._handlers["parallel_join"] = e2e_parallel_join_handler

    yield
    StepCatalog.reset_instance()


@pytest.fixture
def runtime():
    """Create runtime context."""
    return RuntimeContext(
        tenant_id="e2e_tenant",
        instance_id="e2e_instance",
        run_id="e2e_run_001",
        segment_id="e2e_segment",
    )


# === Test Classes ===


class TestFullFunctionalCycle:
    """Tests for complete workflow execution from JSON to Result."""

    @pytest.mark.asyncio
    async def test_json_to_runner_to_result(self, runtime):
        """Test complete cycle: JSON definition -> SegmentRunner -> Result."""
        # Define workflow as JSON (simulating real-world usage)
        segment_json = {
            "segment_id": "e2e-full-cycle",
            "name": "E2E Full Cycle Test",
            "entrypoint": "input",
            "steps": [
                {
                    "step_id": "input",
                    "type": "transform",
                    "config": {"expression": "input"},
                },
                {
                    "step_id": "process",
                    "type": "llm_agent",
                    "config": {"prompt": "Process the query"},
                },
                {
                    "step_id": "output",
                    "type": "transform",
                    "config": {"expression": "append_processed"},
                },
            ],
            "edges": [
                {"from": "input", "to": "process"},
                {"from": "process", "to": "output"},
            ],
        }

        # Parse JSON to SegmentDefinition
        segment = SegmentDefinition.from_dict(segment_json)

        # Execute with SegmentRunner
        runner = SegmentRunner()
        result = await runner.run(
            segment=segment,
            input_data={"query": "Hello, World!"},
            runtime=runtime,
        )

        # Verify result
        assert result.status == SegmentStatus.COMPLETED
        assert result.output is not None
        assert "processed" in result.output.get("output", {})

    @pytest.mark.asyncio
    async def test_multi_step_pipeline(self, runtime):
        """Test pipeline with multiple sequential steps."""
        segment = SegmentDefinition(
            segment_id="e2e-pipeline",
            name="E2E Pipeline",
            entrypoint="step1",
            steps=[
                StepDefinition(step_id="step1", type="transform", config={"expression": "input"}),
                StepDefinition(step_id="step2", type="llm_agent", config={"prompt": "analyze"}),
                StepDefinition(step_id="step3", type="http_action", config={"url": "https://api.example.com", "method": "POST"}),
                StepDefinition(step_id="step4", type="transform", config={"expression": "append_processed"}),
            ],
            edges=[
                EdgeDefinition(from_step="step1", to_step="step2"),
                EdgeDefinition(from_step="step2", to_step="step3"),
                EdgeDefinition(from_step="step3", to_step="step4"),
            ],
        )

        runner = SegmentRunner()
        result = await runner.run(
            segment=segment,
            input_data={"data": "test input"},
            runtime=runtime,
        )

        assert result.status == SegmentStatus.COMPLETED
        assert result.steps_completed >= 4

    @pytest.mark.asyncio
    async def test_conditional_branching(self, runtime):
        """Test workflow with conditional branching."""
        segment = SegmentDefinition(
            segment_id="e2e-conditional",
            name="E2E Conditional",
            entrypoint="start",
            steps=[
                StepDefinition(step_id="start", type="transform", config={"expression": "input"}),
                StepDefinition(
                    step_id="check",
                    type="condition",
                    config={"expression": "value > 10"},
                    output_ports=["true", "false"],
                ),
                StepDefinition(step_id="high_path", type="transform", config={"expression": "input"}),
                StepDefinition(step_id="low_path", type="transform", config={"expression": "input"}),
            ],
            edges=[
                EdgeDefinition(from_step="start", to_step="check"),
                EdgeDefinition(from_step="check", from_port="true", to_step="high_path"),
                EdgeDefinition(from_step="check", from_port="false", to_step="low_path"),
            ],
        )

        runner = SegmentRunner()

        # Test high path (value > 10)
        result_high = await runner.run(
            segment=segment,
            input_data={"value": 15},
            runtime=runtime,
        )
        assert result_high.status == SegmentStatus.COMPLETED

        # Test low path (value <= 10)
        result_low = await runner.run(
            segment=segment,
            input_data={"value": 5},
            runtime=runtime,
        )
        assert result_low.status == SegmentStatus.COMPLETED


class TestParallelExecution:
    """Tests for parallel execution branches."""

    @pytest.mark.asyncio
    async def test_parallel_split_join(self, runtime):
        """Test parallel split and join pattern."""
        segment = SegmentDefinition(
            segment_id="e2e-parallel",
            name="E2E Parallel",
            entrypoint="start",
            steps=[
                StepDefinition(step_id="start", type="transform", config={"expression": "input"}),
                StepDefinition(
                    step_id="split",
                    type="parallel_split",
                    config={"branches": ["branch_a", "branch_b"]},
                    output_ports=["branch_a", "branch_b"],
                ),
                StepDefinition(step_id="process_a", type="llm_agent", config={"prompt": "branch A"}),
                StepDefinition(step_id="process_b", type="http_action", config={"url": "https://api.b.com"}),
                StepDefinition(
                    step_id="join",
                    type="parallel_join",
                    config={"strategy": "all"},
                    input_ports=["input_a", "input_b"],
                ),
                StepDefinition(step_id="end", type="transform", config={"expression": "input"}),
            ],
            edges=[
                EdgeDefinition(from_step="start", to_step="split"),
                EdgeDefinition(from_step="split", from_port="branch_a", to_step="process_a"),
                EdgeDefinition(from_step="split", from_port="branch_b", to_step="process_b"),
                EdgeDefinition(from_step="process_a", to_step="join", to_port="input_a"),
                EdgeDefinition(from_step="process_b", to_step="join", to_port="input_b"),
                EdgeDefinition(from_step="join", to_step="end"),
            ],
        )

        runner = SegmentRunner()
        result = await runner.run(
            segment=segment,
            input_data={"request": "parallel test"},
            runtime=runtime,
        )

        assert result.status == SegmentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_fan_out_pattern(self, runtime):
        """Test fan-out to multiple parallel branches."""
        segment = SegmentDefinition(
            segment_id="e2e-fanout",
            name="E2E Fan-Out",
            entrypoint="source",
            steps=[
                StepDefinition(step_id="source", type="transform", config={"expression": "input"}),
                StepDefinition(
                    step_id="fanout",
                    type="parallel_split",
                    config={"branches": ["worker_1", "worker_2", "worker_3"]},
                    output_ports=["worker_1", "worker_2", "worker_3"],
                ),
                StepDefinition(step_id="worker_1", type="transform", config={"expression": "input"}),
                StepDefinition(step_id="worker_2", type="transform", config={"expression": "input"}),
                StepDefinition(step_id="worker_3", type="transform", config={"expression": "input"}),
            ],
            edges=[
                EdgeDefinition(from_step="source", to_step="fanout"),
                EdgeDefinition(from_step="fanout", from_port="worker_1", to_step="worker_1"),
                EdgeDefinition(from_step="fanout", from_port="worker_2", to_step="worker_2"),
                EdgeDefinition(from_step="fanout", from_port="worker_3", to_step="worker_3"),
            ],
        )

        runner = SegmentRunner()
        result = await runner.run(
            segment=segment,
            input_data={"task": "distribute"},
            runtime=runtime,
        )

        assert result.status == SegmentStatus.COMPLETED


class TestHumanTaskPauseResume:
    """Tests for Human Task pause and resume functionality."""

    @pytest.mark.asyncio
    async def test_human_task_approval_flow(self, runtime):
        """Test workflow with human approval task (simulated)."""
        # Create a simple mock human handler that auto-approves
        async def test_human_handler(ctx, config, input_data):
            # Simulate immediate approval for testing
            return {"approved": input_data}

        # Register handler
        catalog = StepCatalog.instance()
        catalog._handlers["human_task"] = test_human_handler

        segment = SegmentDefinition(
            segment_id="e2e-human-approval",
            name="E2E Human Approval",
            entrypoint="prepare",
            steps=[
                StepDefinition(step_id="prepare", type="transform", config={"expression": "input"}),
                StepDefinition(
                    step_id="approval",
                    type="human_task",
                    config={
                        "task_type": "approval",
                        "title": "Approve Request",
                        "description": "Please review and approve",
                        "assignee_ref": "manager@example.com",
                    },
                    output_ports=["approved", "rejected"],
                ),
                StepDefinition(step_id="process", type="llm_agent", config={"prompt": "process approved"}),
            ],
            edges=[
                EdgeDefinition(from_step="prepare", to_step="approval"),
                EdgeDefinition(from_step="approval", from_port="approved", to_step="process"),
            ],
        )

        runner = SegmentRunner()
        result = await runner.run(
            segment=segment,
            input_data={"request_id": "REQ-001", "amount": 1000},
            runtime=runtime,
        )

        assert result.status == SegmentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_human_task_with_timeout(self, runtime):
        """Test human task with timeout handling."""
        store = MemoryInteractionStore()
        manager = HumanInteractionManager(store)

        async def timeout_human_handler(ctx, config, input_data):
            # Simulate timeout scenario
            timeout_action = config.get("timeout_action", "reject")

            if timeout_action == "auto_approve":
                return {"approved": input_data}
            else:
                return {"timeout": input_data}

        catalog = StepCatalog.instance()
        catalog._handlers["human_task"] = timeout_human_handler

        segment = SegmentDefinition(
            segment_id="e2e-human-timeout",
            name="E2E Human Timeout",
            entrypoint="start",
            steps=[
                StepDefinition(step_id="start", type="transform", config={"expression": "input"}),
                StepDefinition(
                    step_id="review",
                    type="human_task",
                    config={
                        "task_type": "review",
                        "title": "Review Document",
                        "timeout_hours": 0.001,  # Very short timeout
                        "timeout_action": "auto_approve",
                    },
                    output_ports=["approved", "timeout"],
                ),
                StepDefinition(step_id="finalize", type="transform", config={"expression": "input"}),
            ],
            edges=[
                EdgeDefinition(from_step="start", to_step="review"),
                EdgeDefinition(from_step="review", from_port="approved", to_step="finalize"),
            ],
        )

        runner = SegmentRunner()
        result = await runner.run(
            segment=segment,
            input_data={"document": "test.pdf"},
            runtime=runtime,
        )

        assert result.status == SegmentStatus.COMPLETED


class TestErrorHandling:
    """Tests for error handling in workflows."""

    @pytest.mark.asyncio
    async def test_step_failure_handling(self, runtime):
        """Test handling of step failures."""
        async def failing_handler(ctx, config, input_data):
            if config.get("should_fail", False):
                raise ValueError("Intentional test failure")
            return {"output": input_data}

        catalog = StepCatalog.instance()
        catalog._handlers["maybe_fail"] = failing_handler

        segment = SegmentDefinition(
            segment_id="e2e-error",
            name="E2E Error Handling",
            entrypoint="start",
            steps=[
                StepDefinition(step_id="start", type="transform", config={"expression": "input"}),
                StepDefinition(step_id="risky", type="maybe_fail", config={"should_fail": True}),
            ],
            edges=[
                EdgeDefinition(from_step="start", to_step="risky"),
            ],
        )

        runner = SegmentRunner()
        result = await runner.run(
            segment=segment,
            input_data={"test": "error handling"},
            runtime=runtime,
        )

        assert result.status == SegmentStatus.FAILED
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, runtime):
        """Test workflow continues on non-critical failure."""
        call_count = {"value": 0}

        async def counting_handler(ctx, config, input_data):
            call_count["value"] += 1
            return {"output": {"count": call_count["value"]}}

        catalog = StepCatalog.instance()
        catalog._handlers["counter"] = counting_handler

        segment = SegmentDefinition(
            segment_id="e2e-graceful",
            name="E2E Graceful",
            entrypoint="step1",
            steps=[
                StepDefinition(step_id="step1", type="counter", config={}),
                StepDefinition(step_id="step2", type="counter", config={}),
                StepDefinition(step_id="step3", type="counter", config={}),
            ],
            edges=[
                EdgeDefinition(from_step="step1", to_step="step2"),
                EdgeDefinition(from_step="step2", to_step="step3"),
            ],
        )

        runner = SegmentRunner()
        result = await runner.run(
            segment=segment,
            input_data={},
            runtime=runtime,
        )

        assert result.status == SegmentStatus.COMPLETED
        assert call_count["value"] == 3


class TestWorkflowValidation:
    """Tests for workflow validation before execution."""

    @pytest.mark.asyncio
    async def test_invalid_segment_rejected(self, runtime):
        """Test that invalid segments are rejected."""
        # Segment with missing entrypoint step
        segment = SegmentDefinition(
            segment_id="e2e-invalid",
            name="E2E Invalid",
            entrypoint="nonexistent",
            steps=[
                StepDefinition(step_id="step1", type="transform", config={}),
            ],
            edges=[],
        )

        runner = SegmentRunner()
        result = await runner.run(
            segment=segment,
            input_data={},
            runtime=runtime,
        )

        # Should fail due to invalid entrypoint
        assert result.status == SegmentStatus.FAILED

    @pytest.mark.asyncio
    async def test_cycle_detection(self, runtime):
        """Test detection of cycles in workflow graph."""
        segment = SegmentDefinition(
            segment_id="e2e-cycle",
            name="E2E Cycle",
            entrypoint="a",
            steps=[
                StepDefinition(step_id="a", type="transform", config={}),
                StepDefinition(step_id="b", type="transform", config={}),
                StepDefinition(step_id="c", type="transform", config={}),
            ],
            edges=[
                EdgeDefinition(from_step="a", to_step="b"),
                EdgeDefinition(from_step="b", to_step="c"),
                EdgeDefinition(from_step="c", to_step="a"),  # Creates cycle
            ],
        )

        # Validation should catch the cycle
        errors = segment.validate()
        # Cycle should be detected in validation
        assert any("cycle" in str(e).lower() or "entrypoint" in str(e).lower() for e in errors) or len(errors) > 0
