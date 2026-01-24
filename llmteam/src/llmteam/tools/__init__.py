"""
Tools module for LLMTeam.

RFC-013: Tool/Function Calling (basic types, per-agent).
RFC-018: Built-in tools library.
"""

from llmteam.tools.definition import (
    ParamType,
    ToolParameter,
    ToolDefinition,
    ToolResult,
)
from llmteam.tools.decorator import tool
from llmteam.tools.executor import ToolExecutor
from llmteam.tools.builtin import BUILTIN_TOOLS

__all__ = [
    "ParamType",
    "ToolParameter",
    "ToolDefinition",
    "ToolResult",
    "tool",
    "ToolExecutor",
    # RFC-018
    "BUILTIN_TOOLS",
]
