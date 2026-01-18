"""
Switch Handler.

Implements multi-way branching based on value matching or case expressions.
Alternative to nested "condition" steps.
"""

from typing import Any, Optional, List

from llmteam.runtime import StepContext
from llmteam.observability import get_logger

logger = get_logger(__name__)


class SwitchHandler:
    """
    Switch/Case execution handler.
    
    Configuration:
        target (str): The value to switch on (optional if using case expressions).
        cases (list[dict]): List of cases.
            - value: Exact match value
            - expression: Python expression (e.g., 'x > 10')
            - port: Output port name (default: case index or 'case_N')
        default_port (str): Port to use if no case matches (default: 'default').
        
    Input:
        <any> - Data to evaluate
    
    Output:
        <any> - Passed to the selected output port
    
    Usage:
        {
            "step_id": "route_request",
            "type": "switch",
            "config": {
                "target": "{category}",
                "cases": [
                    {"value": "support", "port": "support_team"},
                    {"value": "sales", "port": "sales_team"},
                    {"expression": "'urgent' in value", "port": "urgent_queue"}
                ],
                "default_port": "general_inbox"
            }
        }
    """
    
    STEP_TYPE = "switch"
    DISPLAY_NAME = "Switch"
    DESCRIPTION = "Multi-way branching logic"
    CATEGORY = "flow_control"
    
    async def __call__(
        self,
        ctx: StepContext,
        config: dict[str, Any],
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute switch logic."""
        target_value = config.get("target")
        cases = config.get("cases", [])
        default_port = config.get("default_port", "default")
        
        # Determine the value to switch on
        # If target is not provided, we might be switching on the whole input
        # But typically target is resolved string from input
        
        # Iterate cases
        selected_port = default_port
        
        for case in cases:
            if self._matches(case, target_value, input_data):
                selected_port = case.get("port")
                logger.debug(f"Switch matched case: {case}")
                break
        
        logger.info(f"Switch routed to port: {selected_port}")
        
        # Return input data to selected port
        # In llmteam canvas, specific port output is handled by returning
        # a dict where the key matches the port name.
        return {selected_port: input_data}
    
    def _matches(self, case: dict, target_value: Any, input_data: dict) -> bool:
        """Check if case matches."""
        # Exact value match
        if "value" in case:
            return str(case["value"]) == str(target_value)
        
        # Expression match
        if "expression" in case:
            # Safe evaluation context
            eval_ctx = {"value": target_value, "input": input_data}
            try:
                return bool(eval(case["expression"], {"__builtins__": {}}, eval_ctx))
            except Exception as e:
                logger.warning(f"Switch expression error: {e}")
                return False
                
        return False
