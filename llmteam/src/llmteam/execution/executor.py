"""
Pipeline executor for llmteam.

Provides parallel execution with concurrency control.
"""

import asyncio
import uuid
from typing import Any, List

from llmteam.observability import get_logger
from llmteam.execution.config import ExecutorConfig
from llmteam.execution.stats import TaskResult, ExecutionStats


logger = get_logger(__name__)


def generate_uuid() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())


class PipelineExecutor:
    """
    Executor for parallel agent execution.

    Features:
    - Semaphore-based concurrency control
    - Task timeout handling
    - Backpressure management
    - Execution statistics

    Example:
        executor = PipelineExecutor(
            config=ExecutorConfig(max_concurrent=10)
        )

        results = await executor.execute_parallel(agents, input_data)

        for result in results:
            if result.success:
                print(f"{result.agent_name}: {result.result}")
            else:
                print(f"{result.agent_name} failed: {result.error}")
    """

    def __init__(self, config: ExecutorConfig = None):
        """
        Initialize pipeline executor.

        Args:
            config: Executor configuration
        """
        self.config = config or ExecutorConfig()

        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.queue_size)
        self._stats = ExecutionStats()
        self._running = False
        
        logger.debug(f"PipelineExecutor initialized with mode={self.config.mode}, max_concurrent={self.config.max_concurrent}")

    async def execute_parallel(
        self,
        agents: List[Any],
        input_data: dict,
    ) -> List[TaskResult]:
        """
        Execute agents in parallel.

        Args:
            agents: List of agents to execute
            input_data: Input data for agents

        Returns:
            List of TaskResult for each agent
        """
        count = len(agents)
        logger.info(f"Executing {count} agents in parallel")
        
        tasks = []

        for agent in agents:
            task = asyncio.create_task(
                self._execute_with_semaphore(agent, input_data)
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(f"Parallel execution completed for {count} agents")
        
        final_results = []
        for r in results:
            if isinstance(r, TaskResult):
                final_results.append(r)
            else:
                # Unexpected error during gather
                error_msg = str(r)
                logger.error(f"Unexpected error in parallel execution: {error_msg}")
                final_results.append(TaskResult(
                    task_id=generate_uuid(),
                    agent_name="unknown",
                    success=False,
                    error=error_msg,
                ))

        return final_results

    async def _execute_with_semaphore(
        self,
        agent: Any,
        input_data: dict,
    ) -> TaskResult:
        """
        Execute agent with concurrency control.

        Args:
            agent: Agent to execute
            input_data: Input data

        Returns:
            TaskResult
        """
        agent_name = getattr(agent, 'name', 'unknown')
        
        async with self._semaphore:
            self._stats.current_running += 1
            start_time = asyncio.get_event_loop().time()
            
            logger.debug(f"Starting task for agent: {agent_name}")

            try:
                result = await asyncio.wait_for(
                    agent.process(input_data),
                    timeout=self.config.task_timeout,
                )

                duration = int((asyncio.get_event_loop().time() - start_time) * 1000)
                self._stats.completed_tasks += 1
                self._stats.total_duration_ms += duration
                self._stats.update_avg_duration()
                
                logger.debug(f"Task completed for agent: {agent_name} ({duration}ms)")

                return TaskResult(
                    task_id=generate_uuid(),
                    agent_name=agent_name,
                    success=True,
                    result=result,
                    duration_ms=duration,
                )

            except asyncio.TimeoutError:
                self._stats.failed_tasks += 1
                logger.warning(f"Task timeout for agent: {agent_name} after {self.config.task_timeout}s")
                return TaskResult(
                    task_id=generate_uuid(),
                    agent_name=agent_name,
                    success=False,
                    error="timeout",
                )

            except Exception as e:
                self._stats.failed_tasks += 1
                logger.error(f"Task failed for agent: {agent_name}: {str(e)}")
                return TaskResult(
                    task_id=generate_uuid(),
                    agent_name=agent_name,
                    success=False,
                    error=str(e),
                )

            finally:
                self._stats.current_running -= 1
                self._stats.total_tasks += 1

    def get_stats(self) -> ExecutionStats:
        """
        Get execution statistics.

        Returns:
            Current ExecutionStats
        """
        return self._stats

    def is_backpressure(self) -> bool:
        """
        Check if backpressure condition is active.

        Returns:
            True if queue is too full, False otherwise
        """
        if not self.config.enable_backpressure:
            return False

        queue_ratio = self._queue.qsize() / self.config.queue_size
        is_bp = queue_ratio >= self.config.backpressure_threshold
        
        if is_bp:
            self._stats.backpressure_events += 1
            logger.warning(f"Backpressure triggered: queue utilization {queue_ratio:.2%}")
            
        return is_bp

    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self._stats = ExecutionStats()
        logger.debug("Execution statistics reset")
