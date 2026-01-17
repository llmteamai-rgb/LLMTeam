"""
Segment Runner.

This module provides the SegmentRunner for executing workflow segments.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional
import asyncio

from llmteam.events import (
    EventEmitter,
    EventStream,
    ErrorInfo,
    WorktrailEvent,
)
from llmteam.runtime import RuntimeContext, StepContext
from llmteam.canvas.models import SegmentDefinition, EdgeDefinition
from llmteam.canvas.catalog import StepCatalog


class SegmentStatus(Enum):
    """Segment execution status."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class SegmentResult:
    """Result of segment execution."""

    run_id: str
    segment_id: str
    status: SegmentStatus

    # Output
    output: dict[str, Any] = field(default_factory=dict)

    # Error (if failed)
    error: Optional[ErrorInfo] = None

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: int = 0

    # Steps info
    steps_completed: int = 0
    steps_total: int = 0
    current_step: Optional[str] = None

    # Events
    events: list[WorktrailEvent] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "segment_id": self.segment_id,
            "status": self.status.value,
            "output": self.output,
            "error": self.error.to_dict() if self.error else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "current_step": self.current_step,
        }


@dataclass
class RunConfig:
    """Run configuration."""

    timeout: Optional[timedelta] = None
    max_retries: int = 3
    retry_delay: timedelta = field(default_factory=lambda: timedelta(seconds=1))

    # Callbacks
    on_step_start: Optional[Callable] = None
    on_step_complete: Optional[Callable] = None
    on_step_error: Optional[Callable] = None
    on_cancel: Optional[Callable] = None

    # Persistence
    snapshot_interval: int = 0  # 0 = disabled, N = every N steps


class SegmentRunner:
    """
    Unified segment execution entry point.

    Used by KorpOS to execute segments as sub-workflows.
    """

    def __init__(
        self,
        catalog: Optional[StepCatalog] = None,
        event_stream: Optional[EventStream] = None,
    ) -> None:
        self.catalog = catalog or StepCatalog.instance()
        self.event_stream = event_stream

        self._running: dict[str, asyncio.Task] = {}
        self._cancelled: set[str] = set()

    async def run(
        self,
        segment: SegmentDefinition,
        runtime: RuntimeContext,
        input_data: dict[str, Any],
        *,
        config: Optional[RunConfig] = None,
    ) -> SegmentResult:
        """
        Execute segment.

        Args:
            segment: Segment definition (from JSON)
            runtime: Runtime context with resources
            input_data: Input data for entrypoint
            config: Run configuration

        Returns:
            SegmentResult with output or error
        """
        config = config or RunConfig()
        run_id = runtime.run_id

        # Create emitter
        emitter = EventEmitter(runtime)

        # Initialize result
        result = SegmentResult(
            run_id=run_id,
            segment_id=segment.segment_id,
            status=SegmentStatus.RUNNING,
            started_at=datetime.now(),
            steps_total=len(segment.steps),
        )

        # Emit start event
        emitter.segment_started({"input": input_data})

        try:
            # Create task
            task = asyncio.create_task(
                self._execute_segment(
                    segment, runtime, input_data, emitter, result, config
                )
            )
            self._running[run_id] = task

            # Apply timeout
            if config.timeout:
                output = await asyncio.wait_for(task, config.timeout.total_seconds())
            else:
                output = await task

            # Success
            result.status = SegmentStatus.COMPLETED
            result.output = output
            result.completed_at = datetime.now()
            result.duration_ms = int(
                (result.completed_at - result.started_at).total_seconds() * 1000
            )

            emitter.segment_completed(result.duration_ms, {"output": output})

        except asyncio.CancelledError:
            result.status = SegmentStatus.CANCELLED
            result.completed_at = datetime.now()

            if config.on_cancel:
                await config.on_cancel(result)

        except asyncio.TimeoutError:
            result.status = SegmentStatus.TIMEOUT
            result.completed_at = datetime.now()
            result.error = ErrorInfo(
                error_type="TimeoutError",
                error_message=f"Segment timed out after {config.timeout}",
                recoverable=True,
            )
            emitter.segment_failed(result.error)

        except Exception as e:
            result.status = SegmentStatus.FAILED
            result.completed_at = datetime.now()
            result.error = ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e),
                recoverable=False,
            )
            emitter.segment_failed(result.error)

        finally:
            self._running.pop(run_id, None)
            self._cancelled.discard(run_id)

        return result

    async def cancel(self, run_id: str) -> bool:
        """
        Cancel running segment.

        Returns True if cancelled, False if not found.
        """
        task = self._running.get(run_id)
        if not task:
            return False

        self._cancelled.add(run_id)
        task.cancel()
        return True

    async def get_status(self, run_id: str) -> Optional[SegmentStatus]:
        """Get status of a run."""
        if run_id in self._running:
            if run_id in self._cancelled:
                return SegmentStatus.CANCELLED
            return SegmentStatus.RUNNING
        return None

    def is_running(self, run_id: str) -> bool:
        """Check if run is active."""
        return run_id in self._running

    def list_running(self) -> list[str]:
        """List all running segment run IDs."""
        return list(self._running.keys())

    async def _execute_segment(
        self,
        segment: SegmentDefinition,
        runtime: RuntimeContext,
        input_data: dict,
        emitter: EventEmitter,
        result: SegmentResult,
        config: RunConfig,
    ) -> dict:
        """Execute segment steps."""

        # Build execution graph
        step_map = {s.step_id: s for s in segment.steps}
        edge_map = self._build_edge_map(segment.edges)

        # State
        step_outputs: dict[str, Any] = {}
        current_step_id: Optional[str] = segment.entrypoint

        while current_step_id:
            # Check cancellation
            if runtime.run_id in self._cancelled:
                raise asyncio.CancelledError()

            step_def = step_map[current_step_id]
            result.current_step = current_step_id

            # Create step context
            step_ctx = runtime.child_context(current_step_id)

            # Get handler
            handler = self.catalog.get_handler(step_def.type)
            if not handler:
                raise ValueError(f"No handler for step type: {step_def.type}")

            # Gather input from edges
            step_input = self._gather_step_input(
                current_step_id,
                edge_map,
                step_outputs,
                input_data if current_step_id == segment.entrypoint else None,
            )

            # Callback before step
            if config.on_step_start:
                await config.on_step_start(current_step_id, step_input)

            # Emit step started
            emitter.step_started(current_step_id, step_def.type, {"input": step_input})
            step_start = datetime.now()

            # Execute with retry
            try:
                output = await self._execute_step_with_retry(
                    handler,
                    step_ctx,
                    step_def.config,
                    step_input,
                    config,
                    emitter,
                    current_step_id,
                    step_def.type,
                )

                step_duration = int(
                    (datetime.now() - step_start).total_seconds() * 1000
                )
                step_outputs[current_step_id] = output
                result.steps_completed += 1

                emitter.step_completed(
                    current_step_id, step_def.type, step_duration, {"output": output}
                )

                if config.on_step_complete:
                    await config.on_step_complete(current_step_id, output)

            except Exception as e:
                error = ErrorInfo.from_exception(e, recoverable=False)
                emitter.step_failed(current_step_id, step_def.type, error)

                if config.on_step_error:
                    await config.on_step_error(current_step_id, e)

                raise

            # Determine next step
            current_step_id = self._get_next_step(
                current_step_id,
                edge_map,
                output,
            )

        # Return final output (from last executed step)
        if step_outputs:
            # Get output from the last step that was executed
            last_step_id = list(step_outputs.keys())[-1]
            return step_outputs[last_step_id]
        return {}

    async def _execute_step_with_retry(
        self,
        handler: Callable,
        ctx: StepContext,
        config: dict,
        input_data: dict,
        run_config: RunConfig,
        emitter: EventEmitter,
        step_id: str,
        step_type: str,
    ) -> Any:
        """Execute step with retry logic."""
        last_error: Optional[Exception] = None

        for attempt in range(run_config.max_retries + 1):
            try:
                return await handler(ctx, config, input_data)
            except Exception as e:
                last_error = e
                if attempt < run_config.max_retries:
                    error = ErrorInfo.from_exception(e, recoverable=True)
                    emitter.step_retrying(
                        step_id,
                        step_type,
                        attempt=attempt + 1,
                        max_attempts=run_config.max_retries + 1,
                        error=error,
                    )
                    await asyncio.sleep(run_config.retry_delay.total_seconds())

        if last_error:
            raise last_error
        raise RuntimeError("Unexpected: no error after failed retries")

    def _build_edge_map(
        self, edges: list[EdgeDefinition]
    ) -> dict[str, list[EdgeDefinition]]:
        """Build map of outgoing edges for each step."""
        edge_map: dict[str, list[EdgeDefinition]] = {}
        for edge in edges:
            if edge.from_step not in edge_map:
                edge_map[edge.from_step] = []
            edge_map[edge.from_step].append(edge)
        return edge_map

    def _gather_step_input(
        self,
        step_id: str,
        edge_map: dict[str, list[EdgeDefinition]],
        step_outputs: dict[str, Any],
        initial_input: Optional[dict] = None,
    ) -> dict:
        """Gather input for step from incoming edges."""
        if initial_input:
            return initial_input

        # Find incoming edges
        inputs: dict[str, Any] = {}
        for from_step, edges in edge_map.items():
            for edge in edges:
                if edge.to_step == step_id:
                    output = step_outputs.get(from_step, {})
                    if isinstance(output, dict):
                        inputs[edge.to_port] = output.get(edge.from_port, output)
                    else:
                        inputs[edge.to_port] = output

        return inputs

    def _get_next_step(
        self,
        current_step: str,
        edge_map: dict[str, list[EdgeDefinition]],
        output: Any,
    ) -> Optional[str]:
        """Determine next step based on edges and output."""
        edges = edge_map.get(current_step, [])

        if not edges:
            return None

        # Evaluate conditions
        for edge in edges:
            if edge.condition:
                # Simple condition evaluation
                if self._evaluate_condition(edge.condition, output):
                    return edge.to_step
            else:
                # No condition - take this edge
                return edge.to_step

        return None

    def _evaluate_condition(self, condition: str, output: Any) -> bool:
        """
        Evaluate a simple condition expression.

        Supports basic expressions like:
        - "output.status == 'success'"
        - "output.score > 0.5"
        - "true" / "false"
        """
        # Simple boolean conditions
        if condition.lower() == "true":
            return True
        if condition.lower() == "false":
            return False

        # For more complex conditions, we would need a proper expression evaluator
        # For now, support checking output port presence
        if isinstance(output, dict):
            # Check if the output has the condition as a key with truthy value
            if condition in output:
                return bool(output[condition])

        return True  # Default to true if can't evaluate
