"""
Tool definitions and parameter types.

RFC-013: Tool/Function Calling (basic types, per-agent).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type


class ParamType(str, Enum):
    """Supported parameter types for tools."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"

    @classmethod
    def from_python_type(cls, python_type) -> "ParamType":
        """Convert Python type annotation to ParamType."""
        import typing

        origin = getattr(python_type, "__origin__", None)

        if python_type is str:
            return cls.STRING
        elif python_type is int:
            return cls.INTEGER
        elif python_type is float:
            return cls.FLOAT
        elif python_type is bool:
            return cls.BOOLEAN
        elif origin is list or python_type is list:
            return cls.LIST
        elif origin is dict or python_type is dict:
            return cls.DICT
        elif origin is typing.Union:
            # Optional[T] → get inner type
            args = python_type.__args__
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                return cls.from_python_type(non_none[0])
        return cls.STRING  # Default fallback


@dataclass
class ToolParameter:
    """
    A single parameter of a tool.

    Args:
        name: Parameter name
        type: Parameter type
        description: Parameter description
        required: Whether the parameter is required
        default: Default value (None means no default)
    """

    name: str
    type: ParamType = ParamType.STRING
    description: str = ""
    required: bool = True
    default: Any = None

    def validate(self, value: Any) -> Any:
        """
        Validate and coerce value to expected type.

        Returns:
            Coerced value

        Raises:
            TypeError: If value cannot be coerced
        """
        if value is None:
            if not self.required:
                return self.default
            raise TypeError(f"Parameter '{self.name}' is required")

        try:
            if self.type == ParamType.STRING:
                return str(value)
            elif self.type == ParamType.INTEGER:
                return int(value)
            elif self.type == ParamType.FLOAT:
                return float(value)
            elif self.type == ParamType.BOOLEAN:
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes")
                return bool(value)
            elif self.type == ParamType.LIST:
                if isinstance(value, list):
                    return value
                raise TypeError(f"Expected list, got {type(value).__name__}")
            elif self.type == ParamType.DICT:
                if isinstance(value, dict):
                    return value
                raise TypeError(f"Expected dict, got {type(value).__name__}")
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Parameter '{self.name}': cannot convert {type(value).__name__} "
                f"to {self.type.value}: {e}"
            )

        return value

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (for LLM function calling schema)."""
        result = {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "required": self.required,
        }
        if self.default is not None:
            result["default"] = self.default
        return result


@dataclass
class ToolDefinition:
    """
    Definition of a callable tool.

    Args:
        name: Tool name (used as function name in LLM calls)
        description: Tool description (sent to LLM)
        parameters: List of parameters
        handler: The actual function to call
    """

    name: str
    description: str = ""
    parameters: List[ToolParameter] = field(default_factory=list)
    handler: Optional[Callable] = None

    def to_schema(self) -> Dict[str, Any]:
        """
        Convert to OpenAI-compatible function calling schema.

        Returns:
            Dict matching the function calling format.
        """
        properties = {}
        required = []

        for param in self.parameters:
            prop = {"type": param.type.value, "description": param.description}
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def validate_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and coerce arguments.

        Args:
            args: Raw arguments dict

        Returns:
            Validated and coerced arguments

        Raises:
            TypeError: If validation fails
        """
        validated = {}
        for param in self.parameters:
            value = args.get(param.name)
            validated[param.name] = param.validate(value)
        return validated

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
        }


@dataclass
class ToolResult:
    """
    Result of a tool execution.

    Args:
        tool_name: Name of the tool that was called
        output: Tool output (any serializable value)
        success: Whether execution succeeded
        error: Error message if failed
    """

    tool_name: str
    output: Any = None
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "tool_name": self.tool_name,
            "output": self.output,
            "success": self.success,
            "error": self.error,
        }
