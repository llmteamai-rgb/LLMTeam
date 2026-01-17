"""
Execution statistics for llmteam.

Tracks execution metrics and results.
"""

from dataclasses import dataclass
from typing import Any, Optional, TypedDict


class TaskResultDict(TypedDict):
    """Dictionary representation of TaskResult."""
    task_id: str
    agent_name: str
    success: bool
    result: Any
    error: Optional[str]
    duration_ms: int
    retries: int


@dataclass
class TaskResult:
    """
    Result of a task execution.

    Attributes:
        task_id: Unique task identifier
        agent_name: Name of the agent that executed
        success: Whether the task succeeded
        result: Task result data (if successful)
        error: Error message (if failed)
        duration_ms: Execution duration in milliseconds
        retries: Number of retries attempted
    """

    task_id: str
    agent_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration_ms: int = 0
    retries: int = 0

    def to_dict(self) -> TaskResultDict:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "retries": self.retries,
        }


class ExecutionStatsDict(TypedDict):
    """Dictionary representation of ExecutionStats."""
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    total_duration_ms: int
    avg_duration_ms: float
    current_queue_size: int
    current_running: int
    backpressure_events: int


@dataclass
class ExecutionStats:
    """
    Statistics for executor performance.

    Attributes:
        total_tasks: Total number of tasks submitted
        completed_tasks: Number of successfully completed tasks
        failed_tasks: Number of failed tasks
        total_duration_ms: Total execution time
        avg_duration_ms: Average task duration
        current_queue_size: Current number of queued tasks
        current_running: Current number of running tasks
        backpressure_events: Number of backpressure events triggered
    """

    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0

    total_duration_ms: int = 0
    avg_duration_ms: float = 0.0

    current_queue_size: int = 0
    current_running: int = 0

    backpressure_events: int = 0

    def update_avg_duration(self) -> None:
        """Update average duration based on completed tasks."""
        if self.completed_tasks > 0:
            self.avg_duration_ms = self.total_duration_ms / self.completed_tasks

    def to_dict(self) -> ExecutionStatsDict:
        """Convert to dictionary."""
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "total_duration_ms": self.total_duration_ms,
            "avg_duration_ms": self.avg_duration_ms,
            "current_queue_size": self.current_queue_size,
            "current_running": self.current_running,
            "backpressure_events": self.backpressure_events,
        }
