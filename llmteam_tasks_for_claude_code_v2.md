# ЗАДАЧИ ДЛЯ CLAUDE CODE: llmteam v2.0.0 (обновлённые)

**Проект:** llmteam  
**Итерация:** 2  
**Цель:** Устранить оставшиеся 2 блокера  
**Базовая директория:** `/path/to/llmteam/`

---

## СТАТУС ПРЕДЫДУЩИХ ЗАДАЧ

| Задача | Статус |
|--------|--------|
| ~~TASK-001: Версия wheel~~ | ✅ Выполнено |
| ~~TASK-005: Condition evaluator~~ | ⚠️ Частично (работает, но ограничен) |
| TASK-002: LLMAgentHandler | ❌ Требуется |
| TASK-003: HTTPActionHandler | ❌ Требуется |
| TASK-004: TransformHandler | ❌ Требуется |
| TASK-006: Parallel execution | ❌ Требуется |

---

## БЛОКЕР 1: HANDLERS

### TASK-002: Реализовать LLMAgentHandler

**Файл для создания:** `src/llmteam/canvas/handlers/llm_handler.py`

**Контекст — прочитать:**
- `src/llmteam/canvas/handlers.py` — пример HumanTaskHandler
- `src/llmteam/runtime/protocols.py` — протокол LLMProvider
- `src/llmteam/runtime/context.py` — StepContext

**Код:**

```python
"""LLM Agent step handler."""

from typing import Any, Dict

from llmteam.runtime import StepContext


class LLMAgentHandler:
    """
    Handler for llm_agent step type.
    
    Config schema (from catalog):
        llm_ref: str - Reference to registered LLM provider (required)
        prompt_template: str - Prompt with {input} placeholders
        system_prompt: str - Optional system prompt
        temperature: float - Default 0.7
        max_tokens: int - Default 1000
    """
    
    async def __call__(
        self,
        ctx: StepContext,
        config: Dict[str, Any],
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute LLM completion.
        
        Args:
            ctx: Step execution context with runtime access
            config: Step configuration from SegmentDefinition
            input_data: Input from previous step(s)
            
        Returns:
            Dict with 'output' key containing LLM response
            
        Raises:
            ValueError: If LLM provider not found
        """
        # Get LLM provider from runtime
        llm_ref = config.get("llm_ref", "default")
        llm = ctx.runtime.llms.get(llm_ref)
        
        if llm is None:
            raise ValueError(
                f"LLM provider '{llm_ref}' not registered. "
                f"Available: {list(ctx.runtime.llms.list_keys())}"
            )
        
        # Build prompt from template
        prompt_template = config.get("prompt_template", "{input}")
        
        # Flatten input_data for formatting
        format_vars = {"input": input_data.get("input", "")}
        format_vars.update(input_data)
        
        try:
            prompt = prompt_template.format(**format_vars)
        except KeyError as e:
            # Missing variable - use raw template
            prompt = prompt_template
        
        # Build kwargs for LLM call
        llm_kwargs = {}
        if "system_prompt" in config:
            llm_kwargs["system"] = config["system_prompt"]
        if "temperature" in config:
            llm_kwargs["temperature"] = config["temperature"]
        if "max_tokens" in config:
            llm_kwargs["max_tokens"] = config["max_tokens"]
        
        # Call LLM
        try:
            result = await llm.complete(prompt, **llm_kwargs)
            return {"output": result}
        except Exception as e:
            return {
                "output": None,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                }
            }
```

**Тест:** `tests/canvas/test_llm_handler.py`

```python
"""Tests for LLMAgentHandler."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from llmteam.canvas.handlers.llm_handler import LLMAgentHandler
from llmteam.runtime import RuntimeContext


@pytest.fixture
def mock_runtime():
    """Create runtime with mock LLM."""
    runtime = MagicMock(spec=RuntimeContext)
    
    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value="Generated response")
    
    runtime.llms.get = MagicMock(return_value=mock_llm)
    runtime.llms.list_keys = MagicMock(return_value=["default", "openai"])
    
    return runtime, mock_llm


@pytest.fixture
def step_context(mock_runtime):
    """Create step context."""
    runtime, _ = mock_runtime
    ctx = MagicMock()
    ctx.runtime = runtime
    return ctx


class TestLLMAgentHandler:
    
    @pytest.mark.asyncio
    async def test_basic_completion(self, step_context, mock_runtime):
        _, mock_llm = mock_runtime
        handler = LLMAgentHandler()
        
        config = {"llm_ref": "openai"}
        input_data = {"input": "Hello, world!"}
        
        result = await handler(step_context, config, input_data)
        
        assert result["output"] == "Generated response"
        mock_llm.complete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_prompt_template(self, step_context, mock_runtime):
        _, mock_llm = mock_runtime
        handler = LLMAgentHandler()
        
        config = {
            "llm_ref": "openai",
            "prompt_template": "Translate to French: {input}",
        }
        input_data = {"input": "Hello"}
        
        await handler(step_context, config, input_data)
        
        call_args = mock_llm.complete.call_args
        assert "Translate to French: Hello" in call_args[0]
    
    @pytest.mark.asyncio
    async def test_missing_provider(self, step_context, mock_runtime):
        runtime, _ = mock_runtime
        runtime.llms.get = MagicMock(return_value=None)
        
        handler = LLMAgentHandler()
        
        with pytest.raises(ValueError, match="not registered"):
            await handler(step_context, {"llm_ref": "missing"}, {})
    
    @pytest.mark.asyncio
    async def test_llm_error_handling(self, step_context, mock_runtime):
        _, mock_llm = mock_runtime
        mock_llm.complete = AsyncMock(side_effect=Exception("API Error"))
        
        handler = LLMAgentHandler()
        
        result = await handler(step_context, {"llm_ref": "openai"}, {})
        
        assert result["output"] is None
        assert result["error"]["type"] == "Exception"
        assert "API Error" in result["error"]["message"]
```

---

### TASK-003: Реализовать HTTPActionHandler

**Файл:** `src/llmteam/canvas/handlers/http_handler.py`

**Код:**

```python
"""HTTP Action step handler."""

from typing import Any, Dict, Optional
import aiohttp

from llmteam.runtime import StepContext


class HTTPActionHandler:
    """
    Handler for http_action step type.
    
    Config schema (from catalog):
        client_ref: str - Reference to HTTP client (required)
        path: str - Request path (required)
        method: str - HTTP method, default "POST"
        headers: dict - Additional headers
        retry_count: int - Retries, default 3
    """
    
    def __init__(self, default_timeout: int = 30):
        self.default_timeout = default_timeout
    
    async def __call__(
        self,
        ctx: StepContext,
        config: Dict[str, Any],
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute HTTP request.
        
        Returns:
            Dict with 'response' and 'status' keys
        """
        client_ref = config.get("client_ref")
        path = config.get("path", "")
        method = config.get("method", "POST").upper()
        headers = config.get("headers", {})
        timeout_sec = config.get("timeout", self.default_timeout)
        
        # Try to get registered client
        client = None
        base_url = ""
        
        if client_ref and hasattr(ctx.runtime, "clients"):
            client = ctx.runtime.clients.get(client_ref)
            if client and hasattr(client, "base_url"):
                base_url = client.base_url
        
        # Build full URL
        if path.startswith("http://") or path.startswith("https://"):
            url = path
        else:
            url = f"{base_url.rstrip('/')}/{path.lstrip('/')}" if base_url else path
        
        # Substitute placeholders in URL
        try:
            url = url.format(**input_data)
        except KeyError:
            pass  # Keep URL as-is if placeholders don't match
        
        # Execute request
        timeout = aiohttp.ClientTimeout(total=timeout_sec)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            request_kwargs: Dict[str, Any] = {
                "method": method,
                "url": url,
                "headers": headers,
            }
            
            # Add body for methods that support it
            if method in ("POST", "PUT", "PATCH"):
                body = input_data.get("body", input_data.get("input"))
                if body is not None:
                    if isinstance(body, dict):
                        request_kwargs["json"] = body
                    else:
                        request_kwargs["data"] = str(body)
            
            # Add query params for GET
            if method == "GET" and "params" in input_data:
                request_kwargs["params"] = input_data["params"]
            
            try:
                async with session.request(**request_kwargs) as response:
                    # Try to parse as JSON, fallback to text
                    try:
                        response_body = await response.json()
                    except aiohttp.ContentTypeError:
                        response_body = await response.text()
                    
                    return {
                        "response": response_body,
                        "status": response.status,
                        "headers": dict(response.headers),
                    }
            except aiohttp.ClientError as e:
                return {
                    "response": None,
                    "status": 0,
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                    }
                }
```

**Тест:** `tests/canvas/test_http_handler.py`

```python
"""Tests for HTTPActionHandler."""

import pytest
from unittest.mock import MagicMock
from aioresponses import aioresponses

from llmteam.canvas.handlers.http_handler import HTTPActionHandler


@pytest.fixture
def step_context():
    ctx = MagicMock()
    ctx.runtime.clients.get = MagicMock(return_value=None)
    return ctx


class TestHTTPActionHandler:
    
    @pytest.mark.asyncio
    async def test_post_json(self, step_context):
        handler = HTTPActionHandler()
        
        config = {
            "path": "https://api.example.com/data",
            "method": "POST",
        }
        input_data = {"body": {"key": "value"}}
        
        with aioresponses() as mocked:
            mocked.post(
                "https://api.example.com/data",
                payload={"result": "ok"},
                status=200,
            )
            
            result = await handler(step_context, config, input_data)
        
        assert result["status"] == 200
        assert result["response"]["result"] == "ok"
    
    @pytest.mark.asyncio
    async def test_get_request(self, step_context):
        handler = HTTPActionHandler()
        
        config = {
            "path": "https://api.example.com/users/123",
            "method": "GET",
        }
        
        with aioresponses() as mocked:
            mocked.get(
                "https://api.example.com/users/123",
                payload={"name": "John"},
            )
            
            result = await handler(step_context, config, {})
        
        assert result["response"]["name"] == "John"
    
    @pytest.mark.asyncio
    async def test_url_template(self, step_context):
        handler = HTTPActionHandler()
        
        config = {
            "path": "https://api.example.com/users/{user_id}",
            "method": "GET",
        }
        input_data = {"user_id": "456"}
        
        with aioresponses() as mocked:
            mocked.get(
                "https://api.example.com/users/456",
                payload={"id": "456"},
            )
            
            result = await handler(step_context, config, input_data)
        
        assert result["response"]["id"] == "456"
    
    @pytest.mark.asyncio
    async def test_connection_error(self, step_context):
        handler = HTTPActionHandler(default_timeout=1)
        
        config = {"path": "https://nonexistent.invalid/api"}
        
        result = await handler(step_context, config, {})
        
        assert result["status"] == 0
        assert "error" in result
```

**Зависимость для тестов:**
```bash
pip install aioresponses --break-system-packages
```

---

### TASK-004: Реализовать TransformHandler

**Файл:** `src/llmteam/canvas/handlers/transform_handler.py`

**Код:**

```python
"""Transform step handler."""

import json
from typing import Any, Dict, List

from llmteam.runtime import StepContext


class TransformHandler:
    """
    Handler for transform step type.
    
    Config schema:
        mapping: dict - Field mapping {target: source_path}
        expression: str - Simple transform expression
    
    Supports:
        - Field mapping with dot notation
        - Simple built-in expressions
    """
    
    EXPRESSIONS = {
        "json_dumps": lambda x: json.dumps(x),
        "json_loads": lambda x: json.loads(x) if isinstance(x, str) else x,
        "upper": lambda x: x.upper() if isinstance(x, str) else x,
        "lower": lambda x: x.lower() if isinstance(x, str) else x,
        "strip": lambda x: x.strip() if isinstance(x, str) else x,
        "keys": lambda x: list(x.keys()) if isinstance(x, dict) else [],
        "values": lambda x: list(x.values()) if isinstance(x, dict) else [],
        "length": lambda x: len(x) if hasattr(x, "__len__") else 0,
        "first": lambda x: x[0] if x and hasattr(x, "__getitem__") else None,
        "last": lambda x: x[-1] if x and hasattr(x, "__getitem__") else None,
        "flatten": lambda x: [item for sublist in x for item in sublist] if isinstance(x, list) else x,
        "unique": lambda x: list(set(x)) if isinstance(x, list) else x,
        "sort": lambda x: sorted(x) if isinstance(x, list) else x,
        "reverse": lambda x: list(reversed(x)) if isinstance(x, list) else x,
        "passthrough": lambda x: x,
    }
    
    async def __call__(
        self,
        ctx: StepContext,
        config: Dict[str, Any],
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Transform input data.
        
        Priority:
        1. mapping - field mapping
        2. expression - built-in transform
        3. passthrough - return as-is
        """
        # Mode 1: Field mapping
        if "mapping" in config:
            result = self._apply_mapping(config["mapping"], input_data)
            return {"output": result}
        
        # Mode 2: Expression
        if "expression" in config:
            result = self._apply_expression(config["expression"], input_data)
            return {"output": result}
        
        # Mode 3: Passthrough
        return {"output": input_data.get("input", input_data)}
    
    def _apply_mapping(
        self,
        mapping: Dict[str, str],
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply field mapping.
        
        mapping: {"new_name": "old.nested.path"}
        """
        result = {}
        for target, source_path in mapping.items():
            value = self._get_nested(data, source_path)
            self._set_nested(result, target, value)
        return result
    
    def _get_nested(self, data: Any, path: str) -> Any:
        """Get value by dot-notation path."""
        if not path:
            return data
        
        keys = path.split(".")
        value = data
        
        for key in keys:
            if value is None:
                return None
            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, list) and key.isdigit():
                idx = int(key)
                value = value[idx] if 0 <= idx < len(value) else None
            else:
                return None
        
        return value
    
    def _set_nested(self, data: Dict, path: str, value: Any) -> None:
        """Set value by dot-notation path."""
        keys = path.split(".")
        
        for key in keys[:-1]:
            if key not in data:
                data[key] = {}
            data = data[key]
        
        data[keys[-1]] = value
    
    def _apply_expression(
        self,
        expression: str,
        data: Dict[str, Any],
    ) -> Any:
        """Apply built-in expression."""
        input_val = data.get("input", data)
        
        expr_func = self.EXPRESSIONS.get(expression.lower())
        if expr_func:
            try:
                return expr_func(input_val)
            except Exception:
                return input_val
        
        return input_val
```

**Тест:** `tests/canvas/test_transform_handler.py`

```python
"""Tests for TransformHandler."""

import pytest
from unittest.mock import MagicMock

from llmteam.canvas.handlers.transform_handler import TransformHandler


@pytest.fixture
def step_context():
    return MagicMock()


class TestTransformHandler:
    
    @pytest.mark.asyncio
    async def test_passthrough(self, step_context):
        handler = TransformHandler()
        
        result = await handler(step_context, {}, {"input": "hello"})
        
        assert result["output"] == "hello"
    
    @pytest.mark.asyncio
    async def test_mapping_simple(self, step_context):
        handler = TransformHandler()
        
        config = {"mapping": {"name": "user.name", "id": "user.id"}}
        input_data = {"user": {"name": "John", "id": 123}}
        
        result = await handler(step_context, config, input_data)
        
        assert result["output"]["name"] == "John"
        assert result["output"]["id"] == 123
    
    @pytest.mark.asyncio
    async def test_mapping_nested_target(self, step_context):
        handler = TransformHandler()
        
        config = {"mapping": {"result.value": "data"}}
        input_data = {"data": 42}
        
        result = await handler(step_context, config, input_data)
        
        assert result["output"]["result"]["value"] == 42
    
    @pytest.mark.asyncio
    async def test_expression_upper(self, step_context):
        handler = TransformHandler()
        
        config = {"expression": "upper"}
        input_data = {"input": "hello"}
        
        result = await handler(step_context, config, input_data)
        
        assert result["output"] == "HELLO"
    
    @pytest.mark.asyncio
    async def test_expression_keys(self, step_context):
        handler = TransformHandler()
        
        config = {"expression": "keys"}
        input_data = {"input": {"a": 1, "b": 2}}
        
        result = await handler(step_context, config, input_data)
        
        assert set(result["output"]) == {"a", "b"}
    
    @pytest.mark.asyncio
    async def test_expression_json_dumps(self, step_context):
        handler = TransformHandler()
        
        config = {"expression": "json_dumps"}
        input_data = {"input": {"key": "value"}}
        
        result = await handler(step_context, config, input_data)
        
        assert result["output"] == '{"key": "value"}'
    
    @pytest.mark.asyncio
    async def test_array_index_in_path(self, step_context):
        handler = TransformHandler()
        
        config = {"mapping": {"first_item": "items.0"}}
        input_data = {"items": ["a", "b", "c"]}
        
        result = await handler(step_context, config, input_data)
        
        assert result["output"]["first_item"] == "a"
```

---

### TASK-005: Реализовать ConditionHandler

**Файл:** `src/llmteam/canvas/handlers/condition_handler.py`

**Код:**

```python
"""Condition step handler."""

from typing import Any, Dict

from llmteam.runtime import StepContext


class ConditionHandler:
    """
    Handler for condition step type.
    
    Config schema:
        expression: str - Condition to evaluate
    
    Outputs to 'true' or 'false' port based on evaluation.
    
    Supported expressions:
        - "true" / "false" - literals
        - "key" - check if key exists and is truthy
        - "key == value" - equality check
        - "key != value" - inequality check
        - "key > value" / "key < value" - numeric comparison
        - "key in list" - membership check
    """
    
    async def __call__(
        self,
        ctx: StepContext,
        config: Dict[str, Any],
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate condition.
        
        Returns:
            Dict with 'true' or 'false' key containing input_data
        """
        expression = config.get("expression", "true").strip()
        
        result = self._evaluate(expression, input_data)
        
        if result:
            return {"true": input_data, "result": True}
        else:
            return {"false": input_data, "result": False}
    
    def _evaluate(self, expression: str, data: Dict[str, Any]) -> bool:
        """Evaluate expression against data."""
        expr = expression.strip()
        
        # Literals
        if expr.lower() == "true":
            return True
        if expr.lower() == "false":
            return False
        
        # Flatten data for easier access
        flat_data = self._flatten(data)
        
        # Equality: key == value
        if "==" in expr:
            return self._eval_comparison(expr, "==", flat_data)
        
        # Inequality: key != value
        if "!=" in expr:
            return self._eval_comparison(expr, "!=", flat_data)
        
        # Greater than: key > value
        if " > " in expr:
            return self._eval_numeric(expr, ">", flat_data)
        
        # Less than: key < value
        if " < " in expr:
            return self._eval_numeric(expr, "<", flat_data)
        
        # Greater or equal: key >= value
        if ">=" in expr:
            return self._eval_numeric(expr, ">=", flat_data)
        
        # Less or equal: key <= value
        if "<=" in expr:
            return self._eval_numeric(expr, "<=", flat_data)
        
        # Membership: key in list
        if " in " in expr:
            return self._eval_membership(expr, flat_data)
        
        # Simple truthy check
        return bool(flat_data.get(expr))
    
    def _flatten(self, data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested dict for easier access."""
        result = dict(data)  # Keep original keys
        
        # Also add nested access
        if "output" in data and isinstance(data["output"], dict):
            result.update(data["output"])
        if "input" in data and isinstance(data["input"], dict):
            result.update(data["input"])
        
        return result
    
    def _eval_comparison(self, expr: str, op: str, data: Dict) -> bool:
        """Evaluate equality/inequality."""
        parts = expr.split(op)
        if len(parts) != 2:
            return False
        
        key = parts[0].strip()
        value = parts[1].strip().strip("'\"")
        
        actual = data.get(key)
        
        # Try numeric comparison
        try:
            value = float(value) if "." in value else int(value)
        except ValueError:
            pass
        
        if op == "==":
            return actual == value
        else:  # !=
            return actual != value
    
    def _eval_numeric(self, expr: str, op: str, data: Dict) -> bool:
        """Evaluate numeric comparison."""
        parts = expr.split(op)
        if len(parts) != 2:
            return False
        
        key = parts[0].strip()
        try:
            threshold = float(parts[1].strip())
        except ValueError:
            return False
        
        actual = data.get(key)
        if actual is None:
            return False
        
        try:
            actual = float(actual)
        except (ValueError, TypeError):
            return False
        
        if op == ">":
            return actual > threshold
        elif op == "<":
            return actual < threshold
        elif op == ">=":
            return actual >= threshold
        elif op == "<=":
            return actual <= threshold
        
        return False
    
    def _eval_membership(self, expr: str, data: Dict) -> bool:
        """Evaluate membership check."""
        parts = expr.split(" in ")
        if len(parts) != 2:
            return False
        
        value = parts[0].strip().strip("'\"")
        key = parts[1].strip()
        
        container = data.get(key, [])
        return value in container
```

**Тест:** `tests/canvas/test_condition_handler.py`

```python
"""Tests for ConditionHandler."""

import pytest
from unittest.mock import MagicMock

from llmteam.canvas.handlers.condition_handler import ConditionHandler


@pytest.fixture
def step_context():
    return MagicMock()


class TestConditionHandler:
    
    @pytest.mark.asyncio
    async def test_literal_true(self, step_context):
        handler = ConditionHandler()
        
        result = await handler(step_context, {"expression": "true"}, {})
        
        assert "true" in result
        assert result["result"] is True
    
    @pytest.mark.asyncio
    async def test_literal_false(self, step_context):
        handler = ConditionHandler()
        
        result = await handler(step_context, {"expression": "false"}, {})
        
        assert "false" in result
        assert result["result"] is False
    
    @pytest.mark.asyncio
    async def test_equality_string(self, step_context):
        handler = ConditionHandler()
        
        config = {"expression": "status == 'success'"}
        input_data = {"status": "success"}
        
        result = await handler(step_context, config, input_data)
        
        assert result["result"] is True
    
    @pytest.mark.asyncio
    async def test_equality_number(self, step_context):
        handler = ConditionHandler()
        
        config = {"expression": "code == 200"}
        input_data = {"code": 200}
        
        result = await handler(step_context, config, input_data)
        
        assert result["result"] is True
    
    @pytest.mark.asyncio
    async def test_inequality(self, step_context):
        handler = ConditionHandler()
        
        config = {"expression": "status != 'error'"}
        input_data = {"status": "success"}
        
        result = await handler(step_context, config, input_data)
        
        assert result["result"] is True
    
    @pytest.mark.asyncio
    async def test_greater_than(self, step_context):
        handler = ConditionHandler()
        
        config = {"expression": "count > 10"}
        input_data = {"count": 15}
        
        result = await handler(step_context, config, input_data)
        
        assert result["result"] is True
    
    @pytest.mark.asyncio
    async def test_less_than(self, step_context):
        handler = ConditionHandler()
        
        config = {"expression": "score < 50"}
        input_data = {"score": 30}
        
        result = await handler(step_context, config, input_data)
        
        assert result["result"] is True
    
    @pytest.mark.asyncio
    async def test_truthy_check(self, step_context):
        handler = ConditionHandler()
        
        config = {"expression": "approved"}
        input_data = {"approved": True}
        
        result = await handler(step_context, config, input_data)
        
        assert result["result"] is True
    
    @pytest.mark.asyncio
    async def test_falsy_check(self, step_context):
        handler = ConditionHandler()
        
        config = {"expression": "approved"}
        input_data = {"approved": False}
        
        result = await handler(step_context, config, input_data)
        
        assert result["result"] is False
    
    @pytest.mark.asyncio
    async def test_nested_data_access(self, step_context):
        handler = ConditionHandler()
        
        config = {"expression": "status == 'ok'"}
        input_data = {"output": {"status": "ok"}}
        
        result = await handler(step_context, config, input_data)
        
        assert result["result"] is True
    
    @pytest.mark.asyncio
    async def test_membership(self, step_context):
        handler = ConditionHandler()
        
        config = {"expression": "admin in roles"}
        input_data = {"roles": ["user", "admin", "editor"]}
        
        result = await handler(step_context, config, input_data)
        
        assert result["result"] is True
```

---

## БЛОКЕР 2: PARALLEL EXECUTION

### TASK-006: Реализовать ParallelSplitHandler и ParallelJoinHandler

**Файл:** `src/llmteam/canvas/handlers/parallel_handler.py`

**Код:**

```python
"""Parallel execution handlers."""

from typing import Any, Dict

from llmteam.runtime import StepContext


class ParallelSplitHandler:
    """
    Handler for parallel_split step type.
    
    This handler is a marker - actual parallel execution
    is handled by SegmentRunner.
    
    Passes input to all branches.
    """
    
    async def __call__(
        self,
        ctx: StepContext,
        config: Dict[str, Any],
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Prepare data for parallel branches.
        
        Returns input with metadata for runner.
        """
        num_branches = config.get("branches", 2)
        
        # Create output for each branch port
        result = {
            "_parallel": True,
            "_branch_count": num_branches,
        }
        
        # Copy input to each branch output port
        for i in range(1, num_branches + 1):
            result[f"branch_{i}"] = input_data.get("input", input_data)
        
        return result


class ParallelJoinHandler:
    """
    Handler for parallel_join step type.
    
    Collects results from parallel branches.
    
    Config:
        merge_strategy: "all" | "any" | "first"
    """
    
    async def __call__(
        self,
        ctx: StepContext,
        config: Dict[str, Any],
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge branch results.
        
        input_data contains results from all branches:
        {"branch_1": result1, "branch_2": result2, ...}
        """
        strategy = config.get("merge_strategy", "all")
        
        # Collect branch results
        branches = {}
        for key, value in input_data.items():
            if key.startswith("branch_"):
                branches[key] = value
        
        if strategy == "all":
            # Return all results as array
            return {
                "output": list(branches.values()),
                "branches": branches,
            }
        
        elif strategy == "any":
            # Return first non-None result
            for key, value in branches.items():
                if value is not None:
                    return {
                        "output": value,
                        "from_branch": key,
                    }
            return {"output": None}
        
        elif strategy == "first":
            # Return first result (by branch order)
            if branches:
                sorted_keys = sorted(branches.keys())
                first_key = sorted_keys[0]
                return {
                    "output": branches[first_key],
                    "from_branch": first_key,
                }
            return {"output": None}
        
        return {"output": list(branches.values())}
```

---

### TASK-007: Модифицировать SegmentRunner для parallel execution

**Файл:** `src/llmteam/canvas/runner.py`

**Изменения в `_execute_steps()`:**

Найти цикл `while current_step_id:` (около строки 675) и добавить обработку parallel_split:

```python
# Добавить импорт вверху файла
import asyncio
from typing import Set

# В методе _execute_steps(), внутри цикла while, ПЕРЕД получением handler:

while current_step_id:
    # ... existing cancellation check ...
    
    step_def = step_map[current_step_id]
    
    # === НОВЫЙ КОД: Обработка parallel_split ===
    if step_def.type == "parallel_split":
        parallel_result = await self._execute_parallel_branches(
            step_def=step_def,
            step_map=step_map,
            edge_map=edge_map,
            runtime=runtime,
            input_data=self._gather_step_input(
                current_step_id, edge_map, step_outputs,
                input_data if current_step_id == segment.entrypoint else None,
            ),
            config=config,
            emitter=emitter,
        )
        
        step_outputs[current_step_id] = parallel_result
        completed_steps.append(current_step_id)
        result.steps_completed += 1
        
        # Find join step and continue from there
        join_step_id = self._find_join_step(current_step_id, edge_map, step_map)
        if join_step_id:
            # Pass branch results to join
            step_outputs[join_step_id] = parallel_result
            current_step_id = join_step_id
        else:
            current_step_id = None
        continue
    # === КОНЕЦ НОВОГО КОДА ===
    
    # ... rest of existing step execution ...
```

**Добавить новые методы в класс SegmentRunner:**

```python
async def _execute_parallel_branches(
    self,
    step_def,
    step_map: dict,
    edge_map: dict,
    runtime: RuntimeContext,
    input_data: dict,
    config: RunConfig,
    emitter: EventEmitter,
) -> dict:
    """
    Execute parallel branches from split step.
    
    Returns dict with branch results.
    """
    # Find branch entry steps from edges
    branch_edges = edge_map.get(step_def.step_id, [])
    branch_step_ids = [edge.to_step for edge in branch_edges]
    
    if not branch_step_ids:
        return {"output": input_data}
    
    emitter.emit_custom("parallel_split_started", {
        "step_id": step_def.step_id,
        "branches": branch_step_ids,
    })
    
    # Find the join step
    join_step_id = self._find_join_step(step_def.step_id, edge_map, step_map)
    
    async def execute_branch(branch_start_id: str, branch_index: int) -> tuple:
        """Execute a single branch."""
        branch_outputs = {}
        current_id = branch_start_id
        branch_input = input_data
        
        while current_id and current_id != join_step_id:
            branch_step = step_map.get(current_id)
            if not branch_step:
                break
            
            # Get handler
            handler = self.catalog.get_handler(branch_step.type)
            if not handler:
                raise ValueError(f"No handler for step type: {branch_step.type}")
            
            # Create context
            step_ctx = runtime.child_context(current_id)
            
            # Execute
            try:
                output = await handler(step_ctx, branch_step.config, branch_input)
                branch_outputs[current_id] = output
                branch_input = output
            except Exception as e:
                return (f"branch_{branch_index + 1}", {"error": str(e)})
            
            # Next step in branch
            next_edges = [e for e in edge_map.get(current_id, []) if e.to_step != join_step_id]
            current_id = next_edges[0].to_step if next_edges else None
        
        return (f"branch_{branch_index + 1}", branch_input)
    
    # Execute all branches in parallel
    tasks = [
        execute_branch(branch_id, idx)
        for idx, branch_id in enumerate(branch_step_ids)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Collect results
    branch_results = {}
    for result in results:
        if isinstance(result, Exception):
            branch_results["error"] = str(result)
        else:
            key, value = result
            branch_results[key] = value
    
    emitter.emit_custom("parallel_split_completed", {
        "step_id": step_def.step_id,
        "results": branch_results,
    })
    
    return branch_results


def _find_join_step(
    self,
    split_step_id: str,
    edge_map: dict,
    step_map: dict,
) -> Optional[str]:
    """Find the parallel_join step for a given split."""
    visited: Set[str] = set()
    queue = []
    
    # Start from edges leaving split
    for edge in edge_map.get(split_step_id, []):
        queue.append(edge.to_step)
    
    while queue:
        step_id = queue.pop(0)
        if step_id in visited:
            continue
        visited.add(step_id)
        
        step = step_map.get(step_id)
        if step and step.type == "parallel_join":
            return step_id
        
        # Continue traversal
        for edge in edge_map.get(step_id, []):
            if edge.to_step not in visited:
                queue.append(edge.to_step)
    
    return None
```

---

### TASK-008: Создать __init__.py для handlers

**Файл:** `src/llmteam/canvas/handlers/__init__.py`

```python
"""
Canvas Step Handlers.

Built-in handlers for step types.
"""

from llmteam.canvas.handlers.llm_handler import LLMAgentHandler
from llmteam.canvas.handlers.http_handler import HTTPActionHandler
from llmteam.canvas.handlers.transform_handler import TransformHandler
from llmteam.canvas.handlers.condition_handler import ConditionHandler
from llmteam.canvas.handlers.parallel_handler import (
    ParallelSplitHandler,
    ParallelJoinHandler,
)

__all__ = [
    "LLMAgentHandler",
    "HTTPActionHandler", 
    "TransformHandler",
    "ConditionHandler",
    "ParallelSplitHandler",
    "ParallelJoinHandler",
]
```

---

### TASK-009: Зарегистрировать handlers в catalog

**Файл:** `src/llmteam/canvas/catalog.py`

В конце метода `_register_builtin_types()` добавить:

```python
def _register_builtin_types(self) -> None:
    """Register built-in step types."""
    
    # ... existing metadata registration ...
    
    # === ДОБАВИТЬ В КОНЕЦ МЕТОДА ===
    
    # Register handlers
    from llmteam.canvas.handlers import (
        LLMAgentHandler,
        HTTPActionHandler,
        TransformHandler,
        ConditionHandler,
        ParallelSplitHandler,
        ParallelJoinHandler,
    )
    
    self._handlers["llm_agent"] = LLMAgentHandler()
    self._handlers["http_action"] = HTTPActionHandler()
    self._handlers["transform"] = TransformHandler()
    self._handlers["condition"] = ConditionHandler()
    self._handlers["parallel_split"] = ParallelSplitHandler()
    self._handlers["parallel_join"] = ParallelJoinHandler()
    
    # human_task handler is created on-demand with manager
```

---

### TASK-010: Обновить тесты runner

**Файл:** `tests/canvas/test_runner.py`

Убрать mock handlers из fixture и добавить тест parallel execution:

```python
# ИЗМЕНИТЬ fixture:
@pytest.fixture(autouse=True)
def reset_catalog():
    """Reset catalog singleton before each test."""
    StepCatalog.reset_instance()
    # Handlers теперь регистрируются автоматически в _register_builtin_types()
    yield
    StepCatalog.reset_instance()


# ДОБАВИТЬ тест:
class TestParallelExecution:
    """Tests for parallel branch execution."""
    
    @pytest.mark.asyncio
    async def test_parallel_branches_execute_concurrently(self, runtime):
        """Verify branches run in parallel, not sequentially."""
        import time
        
        execution_times = {}
        
        async def timed_handler(ctx, config, input_data):
            start = time.time()
            await asyncio.sleep(0.1)  # Simulate work
            execution_times[ctx.step_id] = {
                "start": start,
                "end": time.time(),
            }
            return {"output": ctx.step_id}
        
        catalog = StepCatalog.instance()
        catalog._handlers["timed_step"] = timed_handler
        
        segment = SegmentDefinition(
            segment_id="parallel_test",
            name="Parallel Test",
            entrypoint="split",
            steps=[
                StepDefinition(step_id="split", type="parallel_split", config={"branches": 2}),
                StepDefinition(step_id="branch_a", type="timed_step", config={}),
                StepDefinition(step_id="branch_b", type="timed_step", config={}),
                StepDefinition(step_id="join", type="parallel_join", config={}),
            ],
            edges=[
                EdgeDefinition(from_step="split", to_step="branch_a"),
                EdgeDefinition(from_step="split", to_step="branch_b"),
                EdgeDefinition(from_step="branch_a", to_step="join"),
                EdgeDefinition(from_step="branch_b", to_step="join"),
            ],
        )
        
        runner = SegmentRunner()
        start_time = time.time()
        result = await runner.run(segment, runtime, {"input": "test"})
        total_time = time.time() - start_time
        
        # If sequential: ~0.2s (0.1 + 0.1)
        # If parallel: ~0.1s
        assert total_time < 0.15, f"Expected parallel execution, got {total_time:.2f}s"
        assert result.status == SegmentStatus.COMPLETED
```

---

## ПОРЯДОК ВЫПОЛНЕНИЯ

```
TASK-002 (LLM handler)
    ↓
TASK-003 (HTTP handler)
    ↓
TASK-004 (Transform handler)
    ↓
TASK-005 (Condition handler)
    ↓
TASK-006 (Parallel handlers)
    ↓
TASK-007 (Runner modification) ← требует внимания
    ↓
TASK-008 (handlers __init__)
    ↓
TASK-009 (catalog registration)
    ↓
TASK-010 (update tests)
    ↓
Финальный прогон тестов
```

---

## КОМАНДЫ ПРОВЕРКИ

```bash
# После каждого handler:
python -c "from llmteam.canvas.handlers.llm_handler import LLMAgentHandler; print('OK')"

# После всех handlers:
python -c "from llmteam.canvas import StepCatalog; c = StepCatalog.instance(); print(list(c._handlers.keys()))"

# Тесты:
pytest tests/canvas/ -v

# Полный прогон:
pytest tests/ -v --tb=short
```

---

*Задачи для Claude Code*  
*Версия: 2*  
*Дата: 17.01.2026*
