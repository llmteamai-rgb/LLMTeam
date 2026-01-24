"""
Tool executor for safe tool execution.

RFC-013: Tool/Function Calling.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional

from llmteam.tools.definition import ToolDefinition, ToolResult


class ToolExecutor:
    """
    Executes tools safely with validation.

    Manages a registry of tools and handles execution
    with argument validation and error handling.

    Example:
        executor = ToolExecutor()
        executor.register(get_weather_tool)

        result = await executor.execute("get_weather", {"city": "London"})
        print(result.output)  # "Weather in London: 20°C"
    """

    def __init__(
        self,
        tools: Optional[List[ToolDefinition]] = None,
        timeout: float = 30.0,
        on_call: Optional[Callable] = None,
    ):
        """
        Initialize executor.

        Args:
            tools: Initial list of tools to register
            timeout: Execution timeout in seconds
            on_call: Callback on each tool call (tool_name, args)
        """
        self._tools: Dict[str, ToolDefinition] = {}
        self._timeout = timeout
        self._on_call = on_call
        self._call_history: List[Dict[str, Any]] = []

        if tools:
            for t in tools:
                self.register(t)

    def register(self, tool: ToolDefinition) -> None:
        """
        Register a tool.

        Args:
            tool: ToolDefinition to register

        Raises:
            ValueError: If tool with same name already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """
        Unregister a tool by name.

        Returns:
            True if removed, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List registered tool names."""
        return list(self._tools.keys())

    def get_schemas(self) -> List[Dict[str, Any]]:
        """
        Get OpenAI-compatible function schemas for all registered tools.

        Returns:
            List of function calling schemas
        """
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(
        self,
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """
        Execute a tool by name with arguments.

        Validates arguments, handles sync/async functions,
        and enforces timeout.

        Args:
            tool_name: Name of the tool to execute
            args: Arguments to pass to the tool

        Returns:
            ToolResult with output or error
        """
        args = args or {}

        # Find tool
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool '{tool_name}' not found",
            )

        if tool.handler is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool '{tool_name}' has no handler",
            )

        # Validate arguments
        try:
            validated_args = tool.validate_args(args)
        except TypeError as e:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
            )

        # Notify callback
        if self._on_call:
            self._on_call(tool_name, validated_args)

        # Execute with timeout
        try:
            if asyncio.iscoroutinefunction(tool.handler):
                output = await asyncio.wait_for(
                    tool.handler(**validated_args),
                    timeout=self._timeout,
                )
            else:
                # Run sync function in executor to avoid blocking
                loop = asyncio.get_event_loop()
                output = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: tool.handler(**validated_args)),
                    timeout=self._timeout,
                )

            result = ToolResult(
                tool_name=tool_name,
                output=output,
                success=True,
            )

        except asyncio.TimeoutError:
            result = ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool '{tool_name}' timed out after {self._timeout}s",
            )
        except Exception as e:
            result = ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"{type(e).__name__}: {e}",
            )

        # Record in history
        self._call_history.append({
            "tool_name": tool_name,
            "args": args,
            "result": result.to_dict(),
        })

        return result

    @property
    def call_history(self) -> List[Dict[str, Any]]:
        """Get call history."""
        return self._call_history.copy()

    def clear_history(self) -> None:
        """Clear call history."""
        self._call_history.clear()
