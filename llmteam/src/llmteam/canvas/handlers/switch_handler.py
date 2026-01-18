"""
Switch Handler.

Multi-way branching based on value matching (like switch/case statement).

Usage:
    {
        "step_id": "route_by_type",
        "type": "switch",
        "config": {
            "expression": "request.type",
            "cases": {
                "order": "process_order",
                "refund": "process_refund",
                "inquiry": "handle_inquiry"
            },
            "default": "unknown_handler"
        }
    }
"""

from typing import Any, Optional
from dataclasses import dataclass, field

from llmteam.runtime import StepContext
from llmteam.observability import get_logger


logger = get_logger(__name__)


@dataclass
class SwitchConfig:
    """Configuration for switch handler."""

    # Expression to evaluate (field path or literal)
    expression: str = ""

    # Case mappings: {value: output_port}
    cases: dict[str, str] = field(default_factory=dict)

    # Default output port when no case matches
    default: Optional[str] = None

    # Whether to use strict equality (vs. string coercion)
    strict: bool = False

    # Whether multiple cases can match (routes to all matching)
    multi_match: bool = False


class SwitchHandler:
    """
    Handler for multi-way branching based on value matching.

    Similar to switch/case statements in programming languages.
    Routes execution to different output ports based on expression value.

    Step Type: "switch"

    Config:
        expression: Field path or expression to evaluate
        cases: Mapping of values to output port names
        default: Default output port when no case matches
        strict: Use strict equality (type-sensitive)
        multi_match: Allow routing to multiple matching cases

    Input:
        Any data containing the field referenced in expression

    Output:
        Routes to the matching case output port with input data
        Each case becomes an output port name

    Usage in segment JSON:
        {
            "step_id": "route_request",
            "type": "switch",
            "config": {
                "expression": "action",
                "cases": {
                    "create": "create_handler",
                    "update": "update_handler",
                    "delete": "delete_handler"
                },
                "default": "error_handler"
            }
        }
    """

    STEP_TYPE = "switch"
    DISPLAY_NAME = "Switch"
    DESCRIPTION = "Multi-way branching based on value matching"
    CATEGORY = "flow_control"

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
        Evaluate switch expression and route to matching case.

        Args:
            ctx: Step context
            config: Step configuration
            input_data: Input data for evaluation

        Returns:
            Dict with matching case port(s) as keys
        """
        switch_config = self._parse_config(config)

        logger.debug(f"Switch: evaluating '{switch_config.expression}'")

        # Get the value to match
        value = self._evaluate_expression(switch_config.expression, input_data)

        logger.debug(f"Switch: value = {value!r}")

        # Find matching case(s)
        matches = self._find_matches(value, switch_config)

        if not matches:
            # Use default if no matches
            if switch_config.default:
                logger.debug(f"Switch: using default '{switch_config.default}'")
                return {switch_config.default: input_data}
            else:
                # No match and no default - return empty
                logger.warning(f"Switch: no match for value {value!r}, no default")
                return {"_no_match": input_data}

        # Build output
        result = {}
        for port in matches:
            result[port] = input_data

        logger.debug(f"Switch: routing to {list(result.keys())}")
        return result

    def _parse_config(self, config: dict) -> SwitchConfig:
        """Parse configuration dict into SwitchConfig."""
        return SwitchConfig(
            expression=config.get("expression", ""),
            cases=config.get("cases", {}),
            default=config.get("default"),
            strict=config.get("strict", False),
            multi_match=config.get("multi_match", False),
        )

    def _evaluate_expression(
        self,
        expression: str,
        data: dict[str, Any],
    ) -> Any:
        """
        Evaluate expression and return the value.

        Supports:
        - Simple field access: "field"
        - Nested field access: "field.subfield"
        - Array indexing: "items[0]"
        """
        expression = expression.strip()

        if not expression:
            return None

        # Handle literal values
        if expression.startswith('"') and expression.endswith('"'):
            return expression[1:-1]
        if expression.startswith("'") and expression.endswith("'"):
            return expression[1:-1]

        # Navigate nested path
        current = data
        parts = self._parse_path(expression)

        for part in parts:
            if isinstance(part, int):
                # Array index
                if isinstance(current, (list, tuple)) and 0 <= part < len(current):
                    current = current[part]
                else:
                    return None
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current

    def _parse_path(self, path: str) -> list[str | int]:
        """Parse a path like 'field.subfield[0].name' into parts."""
        parts: list[str | int] = []
        current = ""

        i = 0
        while i < len(path):
            char = path[i]

            if char == ".":
                if current:
                    parts.append(current)
                    current = ""
            elif char == "[":
                if current:
                    parts.append(current)
                    current = ""
                # Find closing bracket
                j = path.find("]", i)
                if j > i:
                    index_str = path[i + 1 : j]
                    try:
                        parts.append(int(index_str))
                    except ValueError:
                        parts.append(index_str)
                    i = j
            else:
                current += char

            i += 1

        if current:
            parts.append(current)

        return parts

    def _find_matches(
        self,
        value: Any,
        config: SwitchConfig,
    ) -> list[str]:
        """Find matching case(s) for the given value."""
        matches = []

        for case_value, port in config.cases.items():
            if self._matches(value, case_value, config.strict):
                matches.append(port)
                if not config.multi_match:
                    break

        return matches

    def _matches(
        self,
        actual: Any,
        expected: str,
        strict: bool,
    ) -> bool:
        """
        Check if actual value matches expected case value.

        Args:
            actual: The actual value from expression evaluation
            expected: The case value (always a string in config)
            strict: Whether to use strict type comparison
        """
        if actual is None:
            return expected.lower() in ("none", "null", "")

        if strict:
            # Strict: must match type and value
            if isinstance(actual, bool):
                return expected.lower() == str(actual).lower()
            elif isinstance(actual, int):
                try:
                    return actual == int(expected)
                except ValueError:
                    return False
            elif isinstance(actual, float):
                try:
                    return actual == float(expected)
                except ValueError:
                    return False
            else:
                return str(actual) == expected
        else:
            # Loose: string comparison after coercion
            actual_str = str(actual).lower().strip()
            expected_str = expected.lower().strip()
            return actual_str == expected_str

    def get_output_ports(self, config: dict[str, Any]) -> list[str]:
        """
        Get all possible output ports for this step.

        Used for validation and edge suggestions.
        """
        ports = list(config.get("cases", {}).values())
        default = config.get("default")
        if default and default not in ports:
            ports.append(default)
        return ports
