"""
Tests for RFC-018: Built-in Tools Library.
"""

import json
import pytest

from llmteam.tools.builtin import (
    web_search,
    http_fetch,
    json_extract,
    text_summarize,
    code_eval,
    BUILTIN_TOOLS,
)
from llmteam.tools import ToolDefinition


class TestWebSearch:
    """Tests for web_search built-in tool."""

    def test_has_tool_definition(self):
        assert hasattr(web_search, "tool_definition")
        assert isinstance(web_search.tool_definition, ToolDefinition)

    def test_name(self):
        assert web_search.tool_definition.name == "web_search"

    def test_returns_json_results(self):
        result = web_search("AI trends")
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) > 0
        assert "title" in data[0]
        assert "snippet" in data[0]
        assert "url" in data[0]

    def test_max_results(self):
        result = web_search("test", max_results=2)
        data = json.loads(result)
        assert len(data) == 2

    def test_max_results_cap(self):
        result = web_search("test", max_results=100)
        data = json.loads(result)
        assert len(data) <= 5

    def test_query_in_results(self):
        result = web_search("quantum computing")
        assert "quantum computing" in result


class TestHttpFetch:
    """Tests for http_fetch built-in tool."""

    def test_has_tool_definition(self):
        assert hasattr(http_fetch, "tool_definition")
        assert http_fetch.tool_definition.name == "http_fetch"

    def test_returns_json(self):
        result = http_fetch("https://example.com")
        data = json.loads(result)
        assert data["status"] == 200
        assert data["url"] == "https://example.com"
        assert data["method"] == "GET"

    def test_custom_method(self):
        result = http_fetch("https://api.example.com", method="POST")
        data = json.loads(result)
        assert data["method"] == "POST"


class TestJsonExtract:
    """Tests for json_extract built-in tool."""

    def test_has_tool_definition(self):
        assert json_extract.tool_definition.name == "json_extract"

    def test_simple_path(self):
        data = json.dumps({"name": "Alice", "age": 30})
        result = json_extract(data, "name")
        assert result == "Alice"

    def test_nested_path(self):
        data = json.dumps({"user": {"name": "Bob", "email": "bob@x.com"}})
        result = json_extract(data, "user.name")
        assert result == "Bob"

    def test_array_index(self):
        data = json.dumps({"items": ["a", "b", "c"]})
        result = json_extract(data, "items.1")
        assert result == "b"

    def test_nested_object_returns_json(self):
        data = json.dumps({"config": {"debug": True, "port": 8080}})
        result = json_extract(data, "config")
        parsed = json.loads(result)
        assert parsed["debug"] is True

    def test_invalid_json(self):
        result = json_extract("not json", "key")
        assert "Error" in result

    def test_missing_path(self):
        data = json.dumps({"a": 1})
        result = json_extract(data, "b")
        assert "Error" in result

    def test_deep_nested(self):
        data = json.dumps({"a": {"b": {"c": "deep"}}})
        result = json_extract(data, "a.b.c")
        assert result == "deep"


class TestTextSummarize:
    """Tests for text_summarize built-in tool."""

    def test_has_tool_definition(self):
        assert text_summarize.tool_definition.name == "text_summarize"

    def test_extracts_sentences(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = text_summarize(text, max_sentences=2)
        assert "First sentence." in result
        assert "Second sentence." in result
        assert "Third" not in result

    def test_short_text_unchanged(self):
        text = "Short text."
        result = text_summarize(text)
        assert "Short text." in result

    def test_default_three_sentences(self):
        text = "One sentence here. Two sentence here. Three sentence here. Four sentence here."
        result = text_summarize(text)
        # Should have at most 3 sentences
        assert "Four" not in result

    def test_long_text_truncated(self):
        text = "A" * 1000  # No sentence boundaries
        result = text_summarize(text)
        assert len(result) <= 500


class TestCodeEval:
    """Tests for code_eval built-in tool."""

    def test_has_tool_definition(self):
        assert code_eval.tool_definition.name == "code_eval"

    def test_arithmetic(self):
        result = code_eval("2 + 3 * 4")
        assert result == "14"

    def test_string_ops(self):
        result = code_eval("'hello'.upper()")
        assert result == "HELLO"

    def test_list_comprehension(self):
        result = code_eval("[x*2 for x in range(5)]")
        assert result == "[0, 2, 4, 6, 8]"

    def test_builtin_functions(self):
        result = code_eval("sum([1, 2, 3, 4, 5])")
        assert result == "15"

    def test_blocks_import(self):
        result = code_eval("import os")
        assert "Error" in result
        assert "import" in result

    def test_blocks_dunder(self):
        result = code_eval("__builtins__")
        assert "Error" in result

    def test_blocks_exec(self):
        result = code_eval("exec('print(1)')")
        assert "Error" in result

    def test_blocks_os(self):
        result = code_eval("os.system('ls')")
        assert "Error" in result

    def test_division_by_zero(self):
        result = code_eval("1/0")
        assert "Error" in result
        assert "ZeroDivisionError" in result

    def test_syntax_error(self):
        result = code_eval("def f(:")
        assert "Error" in result


class TestBuiltinToolsList:
    """Tests for BUILTIN_TOOLS list."""

    def test_contains_all_tools(self):
        assert len(BUILTIN_TOOLS) == 5

    def test_all_are_tool_definitions(self):
        for td in BUILTIN_TOOLS:
            assert isinstance(td, ToolDefinition)

    def test_tool_names(self):
        names = [t.name for t in BUILTIN_TOOLS]
        assert "web_search" in names
        assert "http_fetch" in names
        assert "json_extract" in names
        assert "text_summarize" in names
        assert "code_eval" in names

    def test_all_have_handlers(self):
        for td in BUILTIN_TOOLS:
            assert td.handler is not None

    def test_all_have_descriptions(self):
        for td in BUILTIN_TOOLS:
            assert td.description != ""

    def test_schemas_are_valid(self):
        for td in BUILTIN_TOOLS:
            schema = td.to_schema()
            assert schema["type"] == "function"
            assert "function" in schema
            assert "name" in schema["function"]
