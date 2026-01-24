"""
Tools module for LLMTeam.

RFC-013: Tool/Function Calling (basic types, per-agent).
"""

from llmteam.tools.definition import (
    ParamType,
    ToolParameter,
    ToolDefinition,
    ToolResult,
)
from llmteam.tools.decorator import tool
from llmteam.tools.executor import ToolExecutor

__all__ = [
    "ParamType",
    "ToolParameter",
    "ToolDefinition",
    "ToolResult",
    "tool",
    "ToolExecutor",
]
