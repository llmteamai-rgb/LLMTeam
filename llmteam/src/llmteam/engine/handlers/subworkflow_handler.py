"""
Subworkflow Handler.

Executes a nested workflow as a single step.
This allows for modular workflow design and reuse.
"""

from typing import Any, Optional

from llmteam.engine import WorkflowDefinition, ExecutionEngine
from llmteam.runtime import StepContext
from llmteam.observability import get_logger

# Backward compatibility aliases
SegmentDefinition = WorkflowDefinition
SegmentRunner = ExecutionEngine

logger = get_logger(__name__)


class SubworkflowHandler:
    """
    Subworkflow execution handler.
    
    Configuration:
        segment_id (str): ID of the segment to run (optional if segment_ref provided).
        segment_ref (str): Reference/Alias to segment.
        isolated (bool): If True, runs in a new trace/run_id context. Default: False.
        input_mapping (dict): Map parent input fields to subworkflow input.
        output_mapping (dict): Map subworkflow output fields to parent output.
        
    Input:
        <dynamic> - Mapped to subworkflow input
    
    Output:
        <dynamic> - Mapped from subworkflow output
    
    Usage:
        {
            "step_id": "process_order",
            "type": "subworkflow",
            "config": {
                "segment_ref": "order_processing_v2",
                "input_mapping": {
                    "order": "current_order",
                    "customer": "customer_data"
                },
                "output_mapping": {
                    "result": "processing_result"
                }
            }
        }
    """
    
    STEP_TYPE = "subworkflow"
    DISPLAY_NAME = "Subworkflow"
    DESCRIPTION = "Execute a nested workflow"
    CATEGORY = "flow_control"
    
    def __init__(self, segment_registry: Optional[dict] = None):
        """
        Initialize handler.
        
        Args:
            segment_registry: Optional registry of segment definitions
        """
        self._registry = segment_registry or {}
        self._runner = SegmentRunner()
    
    def register_segment(self, segment_id: str, segment: SegmentDefinition) -> None:
        """Register a segment for use as subworkflow."""
        self._registry[segment_id] = segment
    
    async def __call__(
        self,
        ctx: StepContext,
        config: dict[str, Any],
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute subworkflow."""
        segment_id = config.get("segment_id")
        segment_ref = config.get("segment_ref")
        input_mapping = config.get("input_mapping", {})
        output_mapping = config.get("output_mapping", {})
        isolated = config.get("isolated", False)
        
        # Resolve segment
        segment = self._resolve_segment(segment_id, segment_ref, ctx)
        if not segment:
            raise ValueError(
                f"Subworkflow segment not found: {segment_id or segment_ref}"
            )
        
        logger.info(
            f"Executing subworkflow: {segment.segment_id}",
            extra={"parent_step": ctx.step_id}
        )
        
        # Map input
        subworkflow_input = self._map_input(input_data, input_mapping)
        
        # Create child runtime
        # If isolated, we create a new run structure, otherwise just a child context
        if isolated:
            # We assume ctx.runtime has a method to spawn a new isolated context 
            # or we manually construct one if supported.
            # ideally: ctx.runtime.create_isolated_child(...)
            # For now, we use child_context which maintains parent linkage but separate ID
            child_runtime = ctx.runtime.child_context(f"sub_{segment.segment_id}")
        else:
            child_runtime = ctx.runtime.child_context(segment.segment_id)
        
        # Execute subworkflow
        try:
            result = await self._runner.run(
                segment=segment,
                runtime=child_runtime,
                input_data=subworkflow_input,
            )
            
            if result.status.value != "completed":
                return {
                    "error": {
                        "type": "SubworkflowFailed",
                        "message": f"Subworkflow {segment.segment_id} failed: {result.status.value}",
                        "details": result.error,
                    }
                }
            
            # Map output
            # result.output is typically {"output": ...} or custom dict
            return self._map_output(result.output, output_mapping)
            
        except Exception as e:
            logger.error(f"Subworkflow failed: {e}")
            return {
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                }
            }
    
    def _resolve_segment(
        self,
        segment_id: Optional[str],
        segment_ref: Optional[str],
        ctx: StepContext,
    ) -> Optional[SegmentDefinition]:
        """Resolve segment from ID, ref, or runtime."""
        # 1. Try internal registry
        if segment_id and segment_id in self._registry:
            return self._registry[segment_id]
        
        if segment_ref and segment_ref in self._registry:
            return self._registry[segment_ref]
        
        # 2. Try runtime segment registry if available
        # This assumes RuntimeContext has a 'segments' lookup
        if hasattr(ctx.runtime, "segments"):
            registry = getattr(ctx.runtime, "segments", {})
            seg = registry.get(segment_id or segment_ref)
            if seg:
                return seg
        
        return None
    
    def _map_input(
        self,
        input_data: dict[str, Any],
        mapping: dict[str, str],
    ) -> dict[str, Any]:
        """Map parent input to subworkflow input."""
        if not mapping:
            return dict(input_data)
        
        result = {}
        for child_key, parent_key in mapping.items():
            if parent_key in input_data:
                result[child_key] = input_data[parent_key]
        
        return result
    
    def _map_output(
        self,
        output_data: dict[str, Any],
        mapping: dict[str, str],
    ) -> dict[str, Any]:
        """Map subworkflow output to parent output."""
        if not mapping:
            return output_data
        
        result = {}
        for parent_key, child_key in mapping.items():
            # Support nested lookup if subworkflow returns {"output": {"field": ...}}
            if child_key in output_data:
                result[parent_key] = output_data[child_key]
            elif "output" in output_data and isinstance(output_data["output"], dict):
                if child_key in output_data["output"]:
                    result[parent_key] = output_data["output"][child_key]
        
        # If mapping didn't produce anything but we have output, maybe keep original?
        # Typically we execute mapping strictly if provided.
        # Fallback to returning result, if empty, maybe merge defaults?
        return result
