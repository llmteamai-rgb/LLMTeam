"""
Built-in tools library.

RFC-018: Ready-to-use tools for common agent tasks.

Usage:
    from llmteam.tools.builtin import web_search, http_fetch, json_extract

    team.add_agent({
        "type": "llm",
        "role": "researcher",
        "tools": [web_search, http_fetch],
    })
"""

import json as json_module
from typing import Any, Dict, List, Optional

from llmteam.tools.decorator import tool


@tool(name="web_search", description="Search the web for information. Returns a list of results with titles and snippets.")
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for information.

    This is a placeholder that returns simulated results.
    Override with a real implementation by replacing the handler:
        web_search.tool_definition.handler = my_real_search_fn

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        JSON string with search results
    """
    results = [
        {
            "title": f"Result {i} for: {query}",
            "snippet": f"Information about {query} (result {i})",
            "url": f"https://example.com/result/{i}",
        }
        for i in range(1, min(max_results, 5) + 1)
    ]
    return json_module.dumps(results, indent=2)


@tool(name="http_fetch", description="Fetch content from a URL. Returns the response body as text.")
def http_fetch(url: str, method: str = "GET") -> str:
    """
    Fetch content from a URL.

    This is a placeholder. For real HTTP calls, replace the handler
    with an async implementation using aiohttp.

    Args:
        url: URL to fetch
        method: HTTP method (GET, POST, etc.)

    Returns:
        Response body as text
    """
    return json_module.dumps({
        "status": 200,
        "url": url,
        "method": method,
        "body": f"[Fetched content from {url}]",
    })


@tool(name="json_extract", description="Extract a value from a JSON string using a dot-notation path (e.g., 'data.items.0.name').")
def json_extract(json_string: str, path: str) -> str:
    """
    Extract a value from JSON using dot-notation path.

    Args:
        json_string: JSON string to parse
        path: Dot-notation path (e.g., "data.items.0.name")

    Returns:
        Extracted value as string
    """
    try:
        data = json_module.loads(json_string)
    except json_module.JSONDecodeError as e:
        return f"Error: Invalid JSON - {e}"

    parts = path.split(".")
    current = data
    for part in parts:
        try:
            if isinstance(current, list):
                current = current[int(part)]
            elif isinstance(current, dict):
                current = current[part]
            else:
                return f"Error: Cannot traverse into {type(current).__name__} at '{part}'"
        except (KeyError, IndexError, ValueError) as e:
            return f"Error: Path '{path}' not found - {e}"

    if isinstance(current, (dict, list)):
        return json_module.dumps(current, indent=2)
    return str(current)


@tool(name="text_summarize", description="Summarize a text by extracting the first N sentences.")
def text_summarize(text: str, max_sentences: int = 3) -> str:
    """
    Simple text summarization by extracting first N sentences.

    For LLM-based summarization, use an LLM agent instead.

    Args:
        text: Text to summarize
        max_sentences: Maximum number of sentences to extract

    Returns:
        Summarized text
    """
    # Simple sentence splitting
    sentences = []
    current = ""
    for char in text:
        current += char
        if char in ".!?" and len(current.strip()) > 10:
            sentences.append(current.strip())
            current = ""
            if len(sentences) >= max_sentences:
                break

    if current.strip() and len(sentences) < max_sentences:
        sentences.append(current.strip())

    result = " ".join(sentences) if sentences else text[:500]
    return result[:500] if len(result) > 500 else result


@tool(name="code_eval", description="Safely evaluate a simple Python expression (arithmetic, string ops, list comprehensions). No imports or side effects allowed.")
def code_eval(expression: str) -> str:
    """
    Safely evaluate a Python expression.

    Only allows: arithmetic, string operations, list/dict comprehensions.
    No imports, no assignments, no function definitions, no side effects.

    Args:
        expression: Python expression to evaluate

    Returns:
        Result as string
    """
    # Safety checks
    forbidden = [
        "import", "__", "exec", "eval", "open", "os.", "sys.",
        "subprocess", "shutil", "pathlib", "file", "write",
        "delete", "remove", "rmdir", "unlink",
    ]
    expr_lower = expression.lower()
    for word in forbidden:
        if word in expr_lower:
            return f"Error: '{word}' is not allowed in expressions"

    try:
        # Restricted globals - only safe builtins
        safe_builtins = {
            "abs": abs, "len": len, "min": min, "max": max,
            "sum": sum, "round": round, "sorted": sorted,
            "list": list, "dict": dict, "set": set, "tuple": tuple,
            "str": str, "int": int, "float": float, "bool": bool,
            "range": range, "enumerate": enumerate, "zip": zip,
            "map": map, "filter": filter,
            "True": True, "False": False, "None": None,
        }
        result = eval(expression, {"__builtins__": safe_builtins}, {})
        return str(result)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# Export all built-in tool definitions for easy access
BUILTIN_TOOLS = [
    web_search.tool_definition,
    http_fetch.tool_definition,
    json_extract.tool_definition,
    text_summarize.tool_definition,
    code_eval.tool_definition,
]
