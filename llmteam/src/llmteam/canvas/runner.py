"""
Segment Runner.

This module provides the SegmentRunner for executing workflow segments.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional, Protocol
import asyncio
import hashlib
import json
import uuid

from llmteam.events import (
    EventEmitter,
    EventStream,
    ErrorInfo,
    WorktrailEvent,
)
from llmteam.runtime import RuntimeContext, StepContext
from llmteam.canvas.models import SegmentDefinition, EdgeDefinition
from llmteam.canvas.catalog import StepCatalog
from llmteam.canvas.exceptions import InvalidConditionError, CanvasError


def _generate_id(prefix: str) -> str:
    """Generate a unique ID with prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class SegmentSnapshotStore(Protocol):
    """Protocol for segment snapshot storage."""

    async def save(self, snapshot: "SegmentSnapshot") -> None:
        """Save snapshot."""
        ...

    async def load(self, snapshot_id: str) -> Optional["SegmentSnapshot"]:
        """Load snapshot by ID."""
        ...


@dataclass
class SegmentSnapshot:
    """
    Snapshot of segment execution state for pause/resume.

    Contains all information needed to resume a paused segment.
    """

    snapshot_id: str
    run_id: str
    segment_id: str
    tenant_id: str

    # Execution state
    current_step: Optional[str]
    completed_steps: list[str]
    step_outputs: dict[str, Any]

    # Input data (for entrypoint)
    input_data: dict[str, Any]

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    original_started_at: Optional[datetime] = None

    # Integrity
    checksum: str = ""

    def compute_checksum(self) -> str:
        """Compute checksum for integrity verification."""
        data = {
            "run_id": self.run_id,
            "segment_id": self.segment_id,
            "current_step": self.current_step,
            "completed_steps": self.completed_steps,
            "step_outputs": json.dumps(self.step_outputs, sort_keys=True, default=str),
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def verify(self) -> bool:
        """Verify integrity."""
        return self.checksum == self.compute_checksum()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "run_id": self.run_id,
            "segment_id": self.segment_id,
            "tenant_id": self.tenant_id,
            "current_step": self.current_step,
            "completed_steps": self.completed_steps,
            "step_outputs": self.step_outputs,
            "input_data": self.input_data,
            "created_at": self.created_at.isoformat(),
            "original_started_at": (
                self.original_started_at.isoformat() if self.original_started_at else None
            ),
            "checksum": self.checksum,
        }


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

    # Resume info
    resumed_from: Optional[str] = None  # snapshot_id if resumed

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
            "resumed_from": self.resumed_from,
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


@dataclass
class _RunState:
    """Internal state for a running segment."""

    segment: SegmentDefinition
    runtime: RuntimeContext
    input_data: dict[str, Any]
    config: RunConfig
    result: SegmentResult
    emitter: EventEmitter

    # Execution state (updated during run)
    step_outputs: dict[str, Any] = field(default_factory=dict)
    current_step: Optional[str] = None
    completed_steps: list[str] = field(default_factory=list)


class SegmentRunner:
    """
    Unified segment execution entry point.

    Used by KorpOS to execute segments as sub-workflows.
    Supports pause/resume for long-running workflows.
    """

    def __init__(
        self,
        catalog: Optional[StepCatalog] = None,
        event_stream: Optional[EventStream] = None,
        snapshot_store: Optional[SegmentSnapshotStore] = None,
    ) -> None:
        self.catalog = catalog or StepCatalog.instance()
        self.event_stream = event_stream
        self._snapshot_store = snapshot_store

        self._running: dict[str, asyncio.Task] = {}
        self._cancelled: set[str] = set()
        self._paused: set[str] = set()
        self._run_state: dict[str, _RunState] = {}
        self._snapshots: dict[str, SegmentSnapshot] = {}
        self._segments: dict[str, SegmentDefinition] = {}  # Cache for resume

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

        # Cache segment for resume
        self._segments[segment.segment_id] = segment

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

        # Track run state for pause/resume
        run_state = _RunState(
            segment=segment,
            runtime=runtime,
            input_data=input_data,
            config=config,
            result=result,
            emitter=emitter,
            current_step=segment.entrypoint,
        )
        self._run_state[run_id] = run_state

        # Emit start event
        emitter.segment_started({"input": input_data})

        try:
            # Create task
            task = asyncio.create_task(
                self._execute_segment(
                    segment, runtime, input_data, emitter, result, config, run_state
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
            # Keep run_state if paused (for snapshot creation)
            if run_id not in self._paused:
                self._run_state.pop(run_id, None)

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
        if run_id in self._paused:
            return SegmentStatus.PAUSED
        if run_id in self._running:
            if run_id in self._cancelled:
                return SegmentStatus.CANCELLED
            return SegmentStatus.RUNNING
        return None

    def is_running(self, run_id: str) -> bool:
        """Check if run is active."""
        return run_id in self._running

    def is_paused(self, run_id: str) -> bool:
        """Check if run is paused."""
        return run_id in self._paused

    def list_running(self) -> list[str]:
        """List all running segment run IDs."""
        return list(self._running.keys())

    def list_paused(self) -> list[str]:
        """List all paused segment run IDs."""
        return list(self._paused)

    async def pause(self, run_id: str) -> Optional[str]:
        """
        Pause a running segment and create a snapshot.

        Args:
            run_id: The run ID to pause

        Returns:
            snapshot_id if paused successfully, None if run not found
        """
        if run_id not in self._running:
            return None

        run_state = self._run_state.get(run_id)
        if not run_state:
            return None

        # Mark as paused (execution loop will check this)
        self._paused.add(run_id)

        # Cancel the task to stop execution
        task = self._running.get(run_id)
        if task:
            task.cancel()

        # Wait briefly for task to process cancellation
        try:
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

        # Create snapshot from current state
        snapshot = SegmentSnapshot(
            snapshot_id=_generate_id("snap"),
            run_id=run_id,
            segment_id=run_state.segment.segment_id,
            tenant_id=run_state.runtime.tenant_id,
            current_step=run_state.current_step,
            completed_steps=list(run_state.completed_steps),
            step_outputs=dict(run_state.step_outputs),
            input_data=run_state.input_data,
            created_at=datetime.now(),
            original_started_at=run_state.result.started_at,
        )
        snapshot.checksum = snapshot.compute_checksum()

        # Save snapshot
        self._snapshots[snapshot.snapshot_id] = snapshot
        if self._snapshot_store:
            await self._snapshot_store.save(snapshot)

        # Update result status
        run_state.result.status = SegmentStatus.PAUSED
        run_state.result.completed_at = datetime.now()

        # Clean up
        self._running.pop(run_id, None)
        self._run_state.pop(run_id, None)

        return snapshot.snapshot_id

    async def resume(
        self,
        snapshot_id: str,
        runtime: RuntimeContext,
        *,
        config: Optional[RunConfig] = None,
    ) -> SegmentResult:
        """
        Resume a segment from a snapshot.

        Args:
            snapshot_id: ID of the snapshot to resume from
            runtime: Runtime context (can be different from original)
            config: Optional new run configuration

        Returns:
            SegmentResult with resumed execution

        Raises:
            CanvasError: If snapshot not found or invalid
        """
        # Load snapshot
        snapshot = self._snapshots.get(snapshot_id)
        if not snapshot and self._snapshot_store:
            snapshot = await self._snapshot_store.load(snapshot_id)

        if not snapshot:
            raise CanvasError(f"Snapshot {snapshot_id} not found")

        # Verify integrity
        if not snapshot.verify():
            raise CanvasError(f"Snapshot {snapshot_id} failed integrity check")

        # Load segment definition
        segment = self._segments.get(snapshot.segment_id)
        if not segment:
            raise CanvasError(
                f"Segment {snapshot.segment_id} not found. "
                "Segment must be cached from a previous run."
            )

        # Clean up paused state for old run_id
        self._paused.discard(snapshot.run_id)

        # Use provided config or create default
        config = config or RunConfig()
        run_id = runtime.run_id

        # Create emitter
        emitter = EventEmitter(runtime)

        # Initialize result (resumed)
        result = SegmentResult(
            run_id=run_id,
            segment_id=segment.segment_id,
            status=SegmentStatus.RUNNING,
            started_at=datetime.now(),
            steps_total=len(segment.steps),
            steps_completed=len(snapshot.completed_steps),
            current_step=snapshot.current_step,
            resumed_from=snapshot_id,
        )

        # Track run state
        run_state = _RunState(
            segment=segment,
            runtime=runtime,
            input_data=snapshot.input_data,
            config=config,
            result=result,
            emitter=emitter,
            step_outputs=dict(snapshot.step_outputs),
            current_step=snapshot.current_step,
            completed_steps=list(snapshot.completed_steps),
        )
        self._run_state[run_id] = run_state

        # Emit resume event
        emitter.segment_started({
            "input": snapshot.input_data,
            "resumed_from": snapshot_id,
            "resumed_step": snapshot.current_step,
        })

        try:
            # Create task
            task = asyncio.create_task(
                self._execute_segment_from_state(
                    segment, runtime, emitter, result, config, run_state
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
            # Check if this was a pause
            if run_id in self._paused:
                result.status = SegmentStatus.PAUSED
            else:
                result.status = SegmentStatus.CANCELLED
            result.completed_at = datetime.now()

            if result.status == SegmentStatus.CANCELLED and config.on_cancel:
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
            if run_id not in self._paused:
                self._run_state.pop(run_id, None)

        return result

    async def get_snapshot(self, snapshot_id: str) -> Optional[SegmentSnapshot]:
        """Get a snapshot by ID."""
        snapshot = self._snapshots.get(snapshot_id)
        if not snapshot and self._snapshot_store:
            snapshot = await self._snapshot_store.load(snapshot_id)
        return snapshot

    async def _execute_segment(
        self,
        segment: SegmentDefinition,
        runtime: RuntimeContext,
        input_data: dict,
        emitter: EventEmitter,
        result: SegmentResult,
        config: RunConfig,
        run_state: _RunState,
    ) -> dict:
        """Execute segment steps from the beginning."""
        # Build execution graph
        step_map = {s.step_id: s for s in segment.steps}
        edge_map = self._build_edge_map(segment.edges)

        # Execute from entrypoint
        return await self._execute_steps(
            segment=segment,
            runtime=runtime,
            input_data=input_data,
            emitter=emitter,
            result=result,
            config=config,
            run_state=run_state,
            step_map=step_map,
            edge_map=edge_map,
            start_step=segment.entrypoint,
            step_outputs={},
            completed_steps=[],
        )

    async def _execute_segment_from_state(
        self,
        segment: SegmentDefinition,
        runtime: RuntimeContext,
        emitter: EventEmitter,
        result: SegmentResult,
        config: RunConfig,
        run_state: _RunState,
    ) -> dict:
        """Execute segment steps from a restored state (resume)."""
        # Build execution graph
        step_map = {s.step_id: s for s in segment.steps}
        edge_map = self._build_edge_map(segment.edges)

        # Execute from current step with restored state
        return await self._execute_steps(
            segment=segment,
            runtime=runtime,
            input_data=run_state.input_data,
            emitter=emitter,
            result=result,
            config=config,
            run_state=run_state,
            step_map=step_map,
            edge_map=edge_map,
            start_step=run_state.current_step,
            step_outputs=run_state.step_outputs,
            completed_steps=run_state.completed_steps,
        )

    async def _execute_steps(
        self,
        segment: SegmentDefinition,
        runtime: RuntimeContext,
        input_data: dict,
        emitter: EventEmitter,
        result: SegmentResult,
        config: RunConfig,
        run_state: _RunState,
        step_map: dict,
        edge_map: dict,
        start_step: Optional[str],
        step_outputs: dict[str, Any],
        completed_steps: list[str],
    ) -> dict:
        """Core step execution loop."""
        current_step_id = start_step

        while current_step_id:
            # Check cancellation or pause
            if runtime.run_id in self._cancelled or runtime.run_id in self._paused:
                # Update run_state before raising
                run_state.current_step = current_step_id
                run_state.step_outputs = step_outputs
                run_state.completed_steps = completed_steps
                raise asyncio.CancelledError()

            step_def = step_map[current_step_id]
            result.current_step = current_step_id
            run_state.current_step = current_step_id

            # Skip if already completed (for resume)
            if current_step_id in completed_steps:
                # Get next step based on saved output
                output = step_outputs.get(current_step_id, {})
                current_step_id = self._get_next_step(current_step_id, edge_map, output)
                continue

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
                completed_steps.append(current_step_id)
                result.steps_completed += 1

                # Update run_state for pause/resume
                run_state.step_outputs = step_outputs
                run_state.completed_steps = completed_steps

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
            run_state.current_step = current_step_id

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
        - "true" / "false" - Boolean literals
        - "<key>" - Check if key exists in output dict with truthy value

        Raises:
            InvalidConditionError: If condition cannot be evaluated
        """
        # Validate condition is not empty
        if not condition or not condition.strip():
            raise InvalidConditionError(condition, "Condition cannot be empty")

        condition = condition.strip()

        # Simple boolean conditions
        if condition.lower() == "true":
            return True
        if condition.lower() == "false":
            return False

        # Check output port presence (condition is a key name)
        if isinstance(output, dict):
            # Check if the output has the condition as a key with truthy value
            if condition in output:
                return bool(output[condition])
            # Key not found in output
            raise InvalidConditionError(
                condition,
                f"Key '{condition}' not found in output. Available keys: {list(output.keys())}"
            )

        # Cannot evaluate condition against non-dict output
        raise InvalidConditionError(
            condition,
            f"Cannot evaluate condition against output of type {type(output).__name__}. "
            "Expected dict or use 'true'/'false' literal."
        )
