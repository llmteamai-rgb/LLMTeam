"""
Tests for pipeline executor.

Tests cover:
- Parallel execution
- Concurrency control (semaphore)
- Timeout handling
- Statistics collection
- Backpressure
"""

import pytest
import asyncio
from dataclasses import dataclass

from llmteam.execution import (
    PipelineExecutor,
    ExecutorConfig,
    ExecutionMode,
    TaskResult,
    ExecutionStats,
)


# Mock agent for testing
@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str
    delay: float = 0.01
    should_fail: bool = False

    async def process(self, input_data: dict) -> dict:
        """Process input (mock implementation)."""
        await asyncio.sleep(self.delay)

        if self.should_fail:
            raise ValueError(f"{self.name} failed")

        return {
            "agent": self.name,
            "input": input_data,
            "result": f"processed by {self.name}",
        }


class TestExecutorConfig:
    """Tests for ExecutorConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ExecutorConfig()

        assert config.mode == ExecutionMode.ADAPTIVE
        assert config.max_concurrent == 10
        assert config.queue_size == 100
        assert config.task_timeout == 300.0
        assert config.enable_backpressure is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExecutorConfig(
            mode=ExecutionMode.PARALLEL,
            max_concurrent=20,
            task_timeout=600.0,
        )

        assert config.mode == ExecutionMode.PARALLEL
        assert config.max_concurrent == 20
        assert config.task_timeout == 600.0


class TestTaskResult:
    """Tests for TaskResult."""

    def test_successful_result(self):
        """Test successful task result."""
        result = TaskResult(
            task_id="task_1",
            agent_name="test_agent",
            success=True,
            result={"data": "test"},
            duration_ms=100,
        )

        assert result.success is True
        assert result.error is None
        assert result.duration_ms == 100

    def test_failed_result(self):
        """Test failed task result."""
        result = TaskResult(
            task_id="task_1",
            agent_name="test_agent",
            success=False,
            error="timeout",
        )

        assert result.success is False
        assert result.error == "timeout"
        assert result.result is None


class TestExecutionStats:
    """Tests for ExecutionStats."""

    def test_default_stats(self):
        """Test default statistics."""
        stats = ExecutionStats()

        assert stats.total_tasks == 0
        assert stats.completed_tasks == 0
        assert stats.failed_tasks == 0
        assert stats.avg_duration_ms == 0.0

    def test_update_avg_duration(self):
        """Test updating average duration."""
        stats = ExecutionStats(
            completed_tasks=5,
            total_duration_ms=500,
        )

        stats.update_avg_duration()

        assert stats.avg_duration_ms == 100.0


class TestPipelineExecutor:
    """Tests for PipelineExecutor."""

    @pytest.mark.asyncio
    async def test_execute_parallel_success(self):
        """Test successful parallel execution."""
        executor = PipelineExecutor(
            config=ExecutorConfig(max_concurrent=5)
        )

        agents = [
            MockAgent(name=f"agent_{i}")
            for i in range(3)
        ]

        results = await executor.execute_parallel(agents, {"test": "data"})

        assert len(results) == 3
        assert all(r.success for r in results)
        assert all(r.result is not None for r in results)

    @pytest.mark.asyncio
    async def test_execute_parallel_with_failure(self):
        """Test parallel execution with some failures."""
        executor = PipelineExecutor()

        agents = [
            MockAgent(name="agent_1"),
            MockAgent(name="agent_2", should_fail=True),
            MockAgent(name="agent_3"),
        ]

        results = await executor.execute_parallel(agents, {"test": "data"})

        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True

    @pytest.mark.asyncio
    async def test_execute_parallel_timeout(self):
        """Test parallel execution with timeout."""
        executor = PipelineExecutor(
            config=ExecutorConfig(task_timeout=0.05)
        )

        agents = [
            MockAgent(name="agent_1", delay=0.01),
            MockAgent(name="agent_2", delay=0.1),  # Will timeout
        ]

        results = await executor.execute_parallel(agents, {"test": "data"})

        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False
        assert results[1].error == "timeout"

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Test that concurrency is limited by semaphore."""
        max_concurrent = 2
        executor = PipelineExecutor(
            config=ExecutorConfig(max_concurrent=max_concurrent)
        )

        # Track max concurrent execution
        current_running = 0
        max_running = 0

        async def track_concurrent():
            nonlocal current_running, max_running
            current_running += 1
            max_running = max(max_running, current_running)
            await asyncio.sleep(0.05)
            current_running -= 1

        # Create agents that track concurrency
        class TrackingAgent:
            def __init__(self, name):
                self.name = name

            async def process(self, input_data):
                await track_concurrent()
                return {"result": "ok"}

        agents = [TrackingAgent(f"agent_{i}") for i in range(5)]

        await executor.execute_parallel(agents, {"test": "data"})

        # Max concurrent should not exceed limit
        assert max_running <= max_concurrent

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting execution statistics."""
        executor = PipelineExecutor()

        agents = [
            MockAgent(name="agent_1"),
            MockAgent(name="agent_2", should_fail=True),
            MockAgent(name="agent_3"),
        ]

        await executor.execute_parallel(agents, {"test": "data"})

        stats = executor.get_stats()

        assert stats.total_tasks == 3
        assert stats.completed_tasks == 2
        assert stats.failed_tasks == 1
        assert stats.avg_duration_ms > 0

    @pytest.mark.asyncio
    async def test_reset_stats(self):
        """Test resetting statistics."""
        executor = PipelineExecutor()

        agents = [MockAgent(name="agent_1")]
        await executor.execute_parallel(agents, {"test": "data"})

        assert executor.get_stats().total_tasks == 1

        executor.reset_stats()

        assert executor.get_stats().total_tasks == 0
        assert executor.get_stats().completed_tasks == 0

    def test_is_backpressure_disabled(self):
        """Test backpressure when disabled."""
        executor = PipelineExecutor(
            config=ExecutorConfig(enable_backpressure=False)
        )

        assert executor.is_backpressure() is False

    def test_is_backpressure_enabled(self):
        """Test backpressure when enabled."""
        executor = PipelineExecutor(
            config=ExecutorConfig(
                enable_backpressure=True,
                queue_size=100,
                backpressure_threshold=0.8,
            )
        )

        # Initially no backpressure
        assert executor.is_backpressure() is False

    @pytest.mark.asyncio
    async def test_empty_agents_list(self):
        """Test executing with empty agents list."""
        executor = PipelineExecutor()

        results = await executor.execute_parallel([], {"test": "data"})

        assert len(results) == 0
        assert executor.get_stats().total_tasks == 0

    @pytest.mark.asyncio
    async def test_agent_without_name_attribute(self):
        """Test agent without name attribute."""
        class SimpleAgent:
            async def process(self, input_data):
                return {"result": "ok"}

        executor = PipelineExecutor()
        agents = [SimpleAgent()]

        results = await executor.execute_parallel(agents, {"test": "data"})

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].agent_name == "unknown"
