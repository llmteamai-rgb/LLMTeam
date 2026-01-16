"""
Parallel execution for llmteam.

This module provides parallel agent execution with:
- Concurrency control (semaphore-based)
- Timeout handling
- Backpressure management
- Execution statistics

Quick Start:
    from llmteam.execution import PipelineExecutor, ExecutorConfig, ExecutionMode

    # Create executor
    executor = PipelineExecutor(
        config=ExecutorConfig(
            mode=ExecutionMode.PARALLEL,
            max_concurrent=10,
        )
    )

    # Execute agents in parallel
    results = await executor.execute_parallel(agents, input_data)

    # Check stats
    stats = executor.get_stats()
    print(f"Completed: {stats.completed_tasks}/{stats.total_tasks}")
"""

from llmteam.execution.config import (
    ExecutionMode,
    ExecutorConfig,
)

from llmteam.execution.stats import (
    TaskResult,
    ExecutionStats,
)

from llmteam.execution.executor import (
    PipelineExecutor,
)

__all__ = [
    "ExecutionMode",
    "ExecutorConfig",
    "TaskResult",
    "ExecutionStats",
    "PipelineExecutor",
]
