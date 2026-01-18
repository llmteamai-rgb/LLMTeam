"""
Subworkflow Handler.

Executes nested workflow segments within a parent workflow.

Usage:
    {
        "step_id": "run_subprocess",
        "type": "subworkflow",
        "config": {
            "segment_id": "child-workflow",
            "segment": {...},  # Or inline segment definition
            "input_mapping": {"child_input": "parent.field"},
            "output_mapping": {"parent_output": "child.result"},
            "timeout_seconds": 60,
            "inherit_context": true
        }
    }
"""

from typing import Any, Optional
from dataclasses import dataclass, field

from llmteam.runtime import StepContext
from llmteam.observability import get_logger


logger = get_logger(__name__)


@dataclass
class SubworkflowConfig:
    """Configuration for subworkflow handler."""

    # Segment ID to execute (for registered segments)
    segment_id: Optional[str] = None

    # Inline segment definition
    segment: Optional[dict[str, Any]] = None

    # Input mapping: {child_field: parent_path}
    input_mapping: dict[str, str] = field(default_factory=dict)

    # Output mapping: {parent_field: child_path}
    output_mapping: dict[str, str] = field(default_factory=dict)

    # Timeout for subworkflow execution
    timeout_seconds: float = 300.0

    # Whether to inherit parent context (stores, clients, etc.)
    inherit_context: bool = True

    # Whether to propagate errors or wrap them
    propagate_errors: bool = True

    # Maximum depth of nested subworkflows
    max_depth: int = 10


class SubworkflowHandler:
    """
    Handler for executing nested workflow segments.

    Allows composing complex workflows from smaller, reusable segments.

    Step Type: "subworkflow"

    Config:
        segment_id: ID of registered segment to execute
        segment: Inline segment definition (alternative to segment_id)
        input_mapping: Map parent data to child input
        output_mapping: Map child output to parent result
        timeout_seconds: Execution timeout
        inherit_context: Whether child inherits parent's runtime context
        propagate_errors: Whether to propagate child errors
        max_depth: Maximum nesting depth

    Input:
        Any data that will be mapped to child segment input

    Output:
        output: Mapped output from child segment
        status: Child segment execution status
        duration_ms: Execution duration

    Usage in segment JSON:
        {
            "step_id": "process_order",
            "type": "subworkflow",
            "config": {
                "segment_id": "order-processing-v2",
                "input_mapping": {
                    "order": "parent.order_data"
                },
                "output_mapping": {
                    "result": "child.processed_order"
                }
            }
        }
    """

    STEP_TYPE = "subworkflow"
    DISPLAY_NAME = "Subworkflow"
    DESCRIPTION = "Execute a nested workflow segment"
    CATEGORY = "orchestration"

    def __init__(self) -> None:
        """Initialize handler."""
        self._segment_cache: dict[str, Any] = {}

    async def __call__(
        self,
        ctx: StepContext,
        config: dict[str, Any],
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute subworkflow.

        Args:
            ctx: Step context
            config: Step configuration
            input_data: Input data to pass to subworkflow

        Returns:
            Dict with output, status, and duration_ms
        """
        subconfig = self._parse_config(config)

        logger.debug(
            f"Subworkflow: executing segment_id={subconfig.segment_id}"
        )

        # Check nesting depth
        current_depth = input_data.get("_subworkflow_depth", 0)
        if current_depth >= subconfig.max_depth:
            raise ValueError(
                f"Maximum subworkflow depth ({subconfig.max_depth}) exceeded"
            )

        # Get segment definition
        segment_def = await self._get_segment(ctx, subconfig)
        if segment_def is None:
            raise ValueError(
                f"Segment not found: {subconfig.segment_id or 'inline'}"
            )

        # Map input data
        child_input = self._map_input(input_data, subconfig.input_mapping)
        child_input["_subworkflow_depth"] = current_depth + 1

        # Execute subworkflow
        try:
            result = await self._execute_segment(
                ctx=ctx,
                segment=segment_def,
                input_data=child_input,
                config=subconfig,
            )

            # Map output
            output = self._map_output(result.get("output", {}), subconfig.output_mapping)

            return {
                "output": output,
                "status": result.get("status", "completed"),
                "duration_ms": result.get("duration_ms", 0),
                "steps_executed": result.get("steps_executed", 0),
            }

        except Exception as e:
            logger.error(f"Subworkflow execution failed: {e}")
            if subconfig.propagate_errors:
                raise
            return {
                "output": {},
                "status": "failed",
                "error": str(e),
            }

    def _parse_config(self, config: dict) -> SubworkflowConfig:
        """Parse configuration dict into SubworkflowConfig."""
        return SubworkflowConfig(
            segment_id=config.get("segment_id"),
            segment=config.get("segment"),
            input_mapping=config.get("input_mapping", {}),
            output_mapping=config.get("output_mapping", {}),
            timeout_seconds=config.get("timeout_seconds", 300.0),
            inherit_context=config.get("inherit_context", True),
            propagate_errors=config.get("propagate_errors", True),
            max_depth=config.get("max_depth", 10),
        )

    async def _get_segment(
        self,
        ctx: StepContext,
        config: SubworkflowConfig,
    ) -> Optional[dict[str, Any]]:
        """Get segment definition by ID or from inline config."""
        # Inline segment takes precedence
        if config.segment:
            return config.segment

        # Look up by segment_id
        if config.segment_id:
            # Check cache first
            if config.segment_id in self._segment_cache:
                return self._segment_cache[config.segment_id]

            # Try to get from segment store if available
            try:
                store = ctx.get_store("segments")
                if store:
                    segment = await store.get(config.segment_id)
                    if segment:
                        self._segment_cache[config.segment_id] = segment
                        return segment
            except Exception:
                pass

        return None

    async def _execute_segment(
        self,
        ctx: StepContext,
        segment: dict[str, Any],
        input_data: dict[str, Any],
        config: SubworkflowConfig,
    ) -> dict[str, Any]:
        """Execute a segment and return results."""
        try:
            from llmteam.canvas import SegmentDefinition, SegmentRunner, RunConfig

            # Parse segment
            segment_def = SegmentDefinition.from_dict(segment)

            # Create child runtime
            if config.inherit_context:
                runtime = ctx.runtime
            else:
                from llmteam.runtime import RuntimeContextManager
                manager = RuntimeContextManager()
                runtime = manager.create_runtime(
                    tenant_id=ctx.tenant_id,
                    instance_id=f"{ctx.instance_id}-sub-{segment_def.segment_id}",
                )

            # Run segment
            runner = SegmentRunner()
            run_config = RunConfig(timeout_seconds=config.timeout_seconds)

            result = await runner.run(
                segment=segment_def,
                input_data=input_data,
                runtime=runtime,
                config=run_config,
            )

            return {
                "status": result.status.value,
                "output": result.output,
                "steps_executed": result.steps_executed,
                "duration_ms": result.duration_ms,
            }

        except ImportError as e:
            raise ImportError(
                f"Canvas module not available for subworkflow execution: {e}"
            )

    def _map_input(
        self,
        data: dict[str, Any],
        mapping: dict[str, str],
    ) -> dict[str, Any]:
        """Map parent data to child input using mapping rules."""
        if not mapping:
            return data.copy()

        result = {}
        for child_field, parent_path in mapping.items():
            value = self._get_nested_value(data, parent_path)
            result[child_field] = value

        return result

    def _map_output(
        self,
        data: dict[str, Any],
        mapping: dict[str, str],
    ) -> dict[str, Any]:
        """Map child output to parent result using mapping rules."""
        if not mapping:
            return data

        result = {}
        for parent_field, child_path in mapping.items():
            value = self._get_nested_value(data, child_path)
            result[parent_field] = value

        return result

    def _get_nested_value(self, data: Any, path: str) -> Any:
        """Get value from nested dict using dot notation."""
        if not path:
            return data

        # Handle special prefixes
        if path.startswith("parent."):
            path = path[7:]
        elif path.startswith("child."):
            path = path[6:]

        current = data
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current
