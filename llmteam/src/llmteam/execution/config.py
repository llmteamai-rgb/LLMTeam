"""
Execution configuration for llmteam.

Defines execution modes and configuration options.
"""

from dataclasses import dataclass
from enum import Enum


class ExecutionMode(Enum):
    """
    Execution mode for pipeline executor.

    Attributes:
        SEQUENTIAL: Execute agents one by one
        PARALLEL: Execute agents concurrently
        ADAPTIVE: Automatically choose based on workload
    """

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"


@dataclass
class ExecutorConfig:
    """
    Configuration for pipeline executor.

    Attributes:
        mode: Execution mode (sequential, parallel, adaptive)
        max_concurrent: Maximum number of concurrent tasks
        queue_size: Maximum size of task queue
        task_timeout: Timeout for individual task (seconds)
        total_timeout: Timeout for entire execution (seconds)
        max_retries: Maximum number of retries per task
        retry_delay: Delay between retries (seconds)
        enable_backpressure: Enable backpressure handling
        backpressure_threshold: Queue fullness ratio to trigger backpressure (0.0-1.0)

    Example:
        config = ExecutorConfig(
            mode=ExecutionMode.PARALLEL,
            max_concurrent=10,
            task_timeout=300.0,  # 5 minutes
            total_timeout=3600.0,  # 1 hour
        )
    """

    mode: ExecutionMode = ExecutionMode.ADAPTIVE
    max_concurrent: int = 10
    queue_size: int = 100

    # Timeouts (seconds)
    task_timeout: float = 300.0  # 5 minutes
    total_timeout: float = 3600.0  # 1 hour

    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0

    # Backpressure
    enable_backpressure: bool = True
    backpressure_threshold: float = 0.8  # 80% queue full
