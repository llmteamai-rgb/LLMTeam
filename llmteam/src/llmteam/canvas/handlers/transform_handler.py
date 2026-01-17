"""
Transform Handler.

Transforms data using expressions or field mappings.
"""

from typing import Any, Optional
import json
import re

from llmteam.runtime import StepContext
from llmteam.observability import get_logger


logger = get_logger(__name__)


class TransformHandler:
    """
    Handler for transform step type.

    Applies field mappings or expressions to transform data.
    """

    def __init__(self) -> None:
        """Initialize handler."""
        pass

    async def __call__(
        self,
        ctx: StepContext,
        config: dict[str, Any],
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute data transformation.

        Args:
            ctx: Step context
            config: Step configuration:
                - expression: JSONPath-like expression (e.g., "data.items", "input")
                - mapping: Field mapping dict (e.g., {"output_field": "input.field"})
                - default: Default value if expression fails
            input_data: Input data to transform

        Returns:
            Dict with 'output' containing transformed data
        """
        expression = config.get("expression")
        mapping = config.get("mapping")
        default_value = config.get("default")

        logger.debug(f"Transform: expression={expression}, mapping={mapping}")

        try:
            if mapping:
                # Apply field mapping
                result = self._apply_mapping(input_data, mapping)
            elif expression:
                # Apply expression
                result = self._evaluate_expression(input_data, expression)
                # Use default if result is None and default is provided
                if result is None and default_value is not None:
                    result = default_value
            else:
                # Pass-through
                result = input_data

            logger.debug(f"Transform completed: result type={type(result).__name__}")

            return {
                "output": result,
            }

        except Exception as e:
            logger.error(f"Transform failed: {e}")
            if default_value is not None:
                return {"output": default_value}
            return {"output": input_data}

    def _apply_mapping(
        self,
        data: dict[str, Any],
        mapping: dict[str, str],
    ) -> dict[str, Any]:
        """
        Apply field mapping to data.

        Args:
            data: Source data
            mapping: Field mapping {output_field: input_path}

        Returns:
            Mapped data dict
        """
        result = {}
        for output_field, input_path in mapping.items():
            value = self._get_nested_value(data, input_path)
            result[output_field] = value
        return result

    def _evaluate_expression(
        self,
        data: Any,
        expression: str,
    ) -> Any:
        """
        Evaluate a simple expression on data.

        Supported expressions:
        - "input" / "data" - return input as-is
        - "output" - return input as-is (for passthrough)
        - "field.subfield" - nested field access
        - "items[0]" - array indexing
        - "keys()" - get dict keys
        - "values()" - get dict values
        - "len()" - get length
        - "json()" - serialize to JSON string

        Args:
            data: Input data
            expression: Expression string

        Returns:
            Evaluated result
        """
        expression = expression.strip()

        # Simple passthrough expressions
        if expression in ("input", "data", "output", "."):
            return data

        # Function expressions
        if expression == "keys()":
            return list(data.keys()) if isinstance(data, dict) else []
        if expression == "values()":
            return list(data.values()) if isinstance(data, dict) else []
        if expression == "len()":
            return len(data) if hasattr(data, "__len__") else 0
        if expression == "json()":
            return json.dumps(data, default=str)

        # Field access with dot notation
        if "." in expression or "[" in expression:
            return self._get_nested_value(data, expression)

        # Simple field access
        if isinstance(data, dict) and expression in data:
            return data[expression]

        # If expression matches input port name, get from that port
        if isinstance(data, dict):
            for key in data:
                if key == expression:
                    return data[key]

        return data

    def _get_nested_value(
        self,
        data: Any,
        path: str,
    ) -> Any:
        """
        Get nested value using dot notation and array indexing.

        Args:
            data: Source data
            path: Path like "field.subfield" or "items[0].name"

        Returns:
            Value at path or None
        """
        current = data

        # Split by dots, handling array notation
        parts = re.split(r'\.(?![^\[]*\])', path)

        for part in parts:
            if not part:
                continue

            # Handle array indexing
            array_match = re.match(r'(\w+)\[(\d+)\]', part)
            if array_match:
                field = array_match.group(1)
                index = int(array_match.group(2))

                if isinstance(current, dict) and field in current:
                    current = current[field]
                    if isinstance(current, list) and 0 <= index < len(current):
                        current = current[index]
                    else:
                        return None
                else:
                    return None
            else:
                # Simple field access
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None

        return current
