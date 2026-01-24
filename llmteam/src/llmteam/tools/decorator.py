"""
Tool decorator for easy function-to-tool conversion.

RFC-013: Tool/Function Calling.
"""

import inspect
from typing import Callable, Optional

from llmteam.tools.definition import ParamType, ToolDefinition, ToolParameter


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable:
    """
    Decorator to convert a function into a ToolDefinition.

    Automatically infers parameter types from type annotations.
    Uses docstring as description if not provided.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)

    Returns:
        Decorated function with .tool_definition attribute

    Example:
        @tool(description="Get the current weather")
        def get_weather(city: str, units: str = "celsius") -> str:
            return f"Weather in {city}: 20°{units[0].upper()}"

        # Access the ToolDefinition:
        print(get_weather.tool_definition.name)  # "get_weather"
        print(get_weather.tool_definition.parameters)  # [city, units]
    """

    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "").strip()

        # Inspect function signature for parameters
        sig = inspect.signature(func)
        hints = func.__annotations__ if hasattr(func, "__annotations__") else {}

        parameters = []
        for param_name, param in sig.parameters.items():
            # Skip 'self' and 'cls'
            if param_name in ("self", "cls"):
                continue

            # Determine type
            python_type = hints.get(param_name, str)
            # Skip return annotation
            if param_name == "return":
                continue

            param_type = ParamType.from_python_type(python_type)

            # Determine if required (has no default)
            required = param.default is inspect.Parameter.empty
            default = None if required else param.default

            parameters.append(
                ToolParameter(
                    name=param_name,
                    type=param_type,
                    description="",
                    required=required,
                    default=default,
                )
            )

        # Create ToolDefinition
        tool_def = ToolDefinition(
            name=tool_name,
            description=tool_desc,
            parameters=parameters,
            handler=func,
        )

        # Attach to function
        func.tool_definition = tool_def
        return func

    return decorator
