"""
Mock implementations for testing.

Provides deterministic mocks for LLM providers, HTTP clients, stores, etc.
"""

import re
from typing import Any, AsyncIterator, Optional, Callable, Union
from dataclasses import dataclass, field


@dataclass
class MockLLMProvider:
    """
    Mock LLM provider for testing.

    Supports:
    - Fixed responses (list of strings)
    - Pattern-based responses (dict mapping regex to response)
    - Custom response function

    Usage:
        # Fixed responses (cycles through list)
        mock = MockLLMProvider(responses=["Hello!", "World!"])

        # Pattern matching
        mock = MockLLMProvider(patterns={
            r".*weather.*": "It's sunny!",
            r".*time.*": "It's 12:00 PM",
        })

        # Custom function
        mock = MockLLMProvider(response_fn=lambda prompt: f"Echo: {prompt}")
    """

    responses: list[str] = field(default_factory=lambda: ["Mock response"])
    patterns: dict[str, str] = field(default_factory=dict)
    response_fn: Optional[Callable[[str], str]] = None

    # Call tracking
    calls: list[dict[str, Any]] = field(default_factory=list)
    _response_index: int = field(default=0, repr=False)

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        """Generate mock completion."""
        self.calls.append({"prompt": prompt, "kwargs": kwargs})

        # Custom function takes priority
        if self.response_fn:
            return self.response_fn(prompt)

        # Pattern matching
        for pattern, response in self.patterns.items():
            if re.search(pattern, prompt, re.IGNORECASE):
                return response

        # Cycle through fixed responses
        response = self.responses[self._response_index % len(self.responses)]
        self._response_index += 1
        return response

    async def complete_with_messages(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Generate mock completion from messages."""
        # Extract last user message as prompt
        prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break
        return await self.complete(prompt, messages=messages, **kwargs)

    async def stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Stream mock completion word by word."""
        response = await self.complete(prompt, **kwargs)
        for word in response.split():
            yield word + " "

    def reset(self) -> None:
        """Reset call tracking and response index."""
        self.calls.clear()
        self._response_index = 0

    def assert_called(self) -> None:
        """Assert that the provider was called at least once."""
        assert len(self.calls) > 0, "MockLLMProvider was not called"

    def assert_called_with(self, prompt_contains: str) -> None:
        """Assert that the provider was called with a prompt containing text."""
        for call in self.calls:
            if prompt_contains in call["prompt"]:
                return
        raise AssertionError(
            f"MockLLMProvider was not called with prompt containing '{prompt_contains}'"
        )

    def assert_call_count(self, expected: int) -> None:
        """Assert the number of calls."""
        actual = len(self.calls)
        assert actual == expected, f"Expected {expected} calls, got {actual}"


@dataclass
class MockHTTPClient:
    """
    Mock HTTP client for testing.

    Usage:
        mock = MockHTTPClient(responses={
            ("GET", "/api/users"): {"users": [{"id": 1}]},
            ("POST", "/api/users"): {"id": 2, "created": True},
        })
    """

    responses: dict[tuple[str, str], Any] = field(default_factory=dict)
    default_response: Any = field(default_factory=lambda: {"status": "ok"})
    calls: list[dict[str, Any]] = field(default_factory=list)
    raise_on_unknown: bool = False

    async def request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Make a mock request."""
        self.calls.append({"method": method, "path": path, "kwargs": kwargs})

        key = (method.upper(), path)
        if key in self.responses:
            response = self.responses[key]
            # Support callable responses
            if callable(response):
                return response(method, path, **kwargs)
            return response

        if self.raise_on_unknown:
            raise KeyError(f"No mock response for {method} {path}")

        return self.default_response

    async def get(self, path: str, **kwargs: Any) -> Any:
        """Make a GET request."""
        return await self.request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs: Any) -> Any:
        """Make a POST request."""
        return await self.request("POST", path, **kwargs)

    async def put(self, path: str, **kwargs: Any) -> Any:
        """Make a PUT request."""
        return await self.request("PUT", path, **kwargs)

    async def delete(self, path: str, **kwargs: Any) -> Any:
        """Make a DELETE request."""
        return await self.request("DELETE", path, **kwargs)

    def reset(self) -> None:
        """Reset call tracking."""
        self.calls.clear()


@dataclass
class MockStore:
    """
    Mock key-value store for testing.

    Usage:
        store = MockStore(initial_data={"key1": "value1"})
        await store.set("key2", "value2")
        value = await store.get("key1")
    """

    initial_data: dict[str, Any] = field(default_factory=dict)
    _data: dict[str, Any] = field(default_factory=dict, repr=False)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._data = dict(self.initial_data)

    async def get(self, key: str) -> Any:
        """Get value by key."""
        self.calls.append({"operation": "get", "key": key})
        return self._data.get(key)

    async def set(self, key: str, value: Any) -> None:
        """Set value by key."""
        self.calls.append({"operation": "set", "key": key, "value": value})
        self._data[key] = value

    async def delete(self, key: str) -> None:
        """Delete value by key."""
        self.calls.append({"operation": "delete", "key": key})
        self._data.pop(key, None)

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._data

    def reset(self) -> None:
        """Reset to initial state."""
        self._data = dict(self.initial_data)
        self.calls.clear()


@dataclass
class MockSecretsProvider:
    """
    Mock secrets provider for testing.

    Usage:
        secrets = MockSecretsProvider(secrets={
            "api_key": "test-key-123",
            "db_password": "secret",
        })
        key = await secrets.get_secret("api_key")
    """

    secrets: dict[str, str] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)
    raise_on_missing: bool = True

    async def get_secret(self, secret_id: str) -> str:
        """Get secret value."""
        self.calls.append(secret_id)

        if secret_id in self.secrets:
            return self.secrets[secret_id]

        if self.raise_on_missing:
            raise KeyError(f"Secret not found: {secret_id}")

        return ""

    def reset(self) -> None:
        """Reset call tracking."""
        self.calls.clear()


@dataclass
class MockEventEmitter:
    """
    Mock event emitter for testing.

    Captures all emitted events for verification.

    Usage:
        emitter = MockEventEmitter()
        await emitter.emit(event)
        assert emitter.events[0].event_type == EventType.STEP_STARTED
    """

    events: list[Any] = field(default_factory=list)
    handlers: dict[str, list[Callable]] = field(default_factory=dict)

    async def emit(self, event: Any) -> None:
        """Emit an event."""
        self.events.append(event)

        # Call registered handlers
        event_type = getattr(event, "event_type", None)
        if event_type and event_type in self.handlers:
            for handler in self.handlers[event_type]:
                if callable(handler):
                    result = handler(event)
                    if hasattr(result, "__await__"):
                        await result

    def on(self, event_type: str, handler: Callable) -> None:
        """Register an event handler."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    def get_events_by_type(self, event_type: Any) -> list[Any]:
        """Get all events of a specific type."""
        return [e for e in self.events if getattr(e, "event_type", None) == event_type]

    def assert_event_emitted(self, event_type: Any) -> None:
        """Assert that an event of the given type was emitted."""
        events = self.get_events_by_type(event_type)
        assert len(events) > 0, f"No events of type {event_type} were emitted"

    def reset(self) -> None:
        """Reset events and handlers."""
        self.events.clear()
        self.handlers.clear()
