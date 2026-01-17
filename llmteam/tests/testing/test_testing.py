"""
Tests for Testing Utilities module.

Tests the mock providers and test runners.
"""

import pytest
from llmteam.testing import (
    MockLLMProvider,
    MockHTTPClient,
    MockStore,
    MockSecretsProvider,
    MockEventEmitter,
    SegmentTestRunner,
    TestRunConfig,
    TestResult,
    StepTestHarness,
    HandlerTestCase,
)


class TestMockLLMProvider:
    """Tests for MockLLMProvider."""

    async def test_fixed_responses(self):
        mock = MockLLMProvider(responses=["Hello!", "World!"])

        response1 = await mock.complete("test1")
        assert response1 == "Hello!"

        response2 = await mock.complete("test2")
        assert response2 == "World!"

        # Cycles back
        response3 = await mock.complete("test3")
        assert response3 == "Hello!"

    async def test_pattern_matching(self):
        mock = MockLLMProvider(patterns={
            r".*weather.*": "It's sunny!",
            r".*time.*": "It's 12:00 PM",
        })

        response = await mock.complete("What's the weather like?")
        assert response == "It's sunny!"

        response = await mock.complete("What time is it?")
        assert response == "It's 12:00 PM"

    async def test_custom_function(self):
        mock = MockLLMProvider(response_fn=lambda p: f"Echo: {p}")

        response = await mock.complete("Hello")
        assert response == "Echo: Hello"

    async def test_call_tracking(self):
        mock = MockLLMProvider()

        await mock.complete("test prompt", temperature=0.5)

        assert len(mock.calls) == 1
        assert mock.calls[0]["prompt"] == "test prompt"
        assert mock.calls[0]["kwargs"]["temperature"] == 0.5

    async def test_assert_called(self):
        mock = MockLLMProvider()

        with pytest.raises(AssertionError):
            mock.assert_called()

        await mock.complete("test")
        mock.assert_called()  # Should not raise

    async def test_assert_called_with(self):
        mock = MockLLMProvider()
        await mock.complete("Hello world")

        mock.assert_called_with("world")

        with pytest.raises(AssertionError):
            mock.assert_called_with("xyz")

    async def test_assert_call_count(self):
        mock = MockLLMProvider()

        await mock.complete("1")
        await mock.complete("2")
        await mock.complete("3")

        mock.assert_call_count(3)

        with pytest.raises(AssertionError):
            mock.assert_call_count(5)

    async def test_reset(self):
        mock = MockLLMProvider(responses=["A", "B"])

        await mock.complete("1")
        await mock.complete("2")

        mock.reset()

        assert len(mock.calls) == 0
        response = await mock.complete("3")
        assert response == "A"  # Back to first response

    async def test_complete_with_messages(self):
        mock = MockLLMProvider(responses=["Response"])

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

        response = await mock.complete_with_messages(messages)
        assert response == "Response"

    async def test_stream(self):
        mock = MockLLMProvider(responses=["Hello world!"])

        chunks = []
        async for chunk in mock.stream("test"):
            chunks.append(chunk)

        assert "".join(chunks).strip() == "Hello world!"


class TestMockHTTPClient:
    """Tests for MockHTTPClient."""

    async def test_configured_responses(self):
        mock = MockHTTPClient(responses={
            ("GET", "/api/users"): {"users": []},
            ("POST", "/api/users"): {"id": 1},
        })

        response = await mock.request("GET", "/api/users")
        assert response == {"users": []}

        response = await mock.request("POST", "/api/users", data={"name": "Test"})
        assert response == {"id": 1}

    async def test_default_response(self):
        mock = MockHTTPClient(default_response={"status": "ok"})

        response = await mock.request("GET", "/unknown")
        assert response == {"status": "ok"}

    async def test_raise_on_unknown(self):
        mock = MockHTTPClient(raise_on_unknown=True)

        with pytest.raises(KeyError):
            await mock.request("GET", "/unknown")

    async def test_call_tracking(self):
        mock = MockHTTPClient()

        await mock.get("/api/test", params={"q": "search"})

        assert len(mock.calls) == 1
        assert mock.calls[0]["method"] == "GET"
        assert mock.calls[0]["path"] == "/api/test"

    async def test_shorthand_methods(self):
        mock = MockHTTPClient()

        await mock.get("/get")
        await mock.post("/post")
        await mock.put("/put")
        await mock.delete("/delete")

        assert len(mock.calls) == 4
        assert mock.calls[0]["method"] == "GET"
        assert mock.calls[1]["method"] == "POST"
        assert mock.calls[2]["method"] == "PUT"
        assert mock.calls[3]["method"] == "DELETE"


class TestMockStore:
    """Tests for MockStore."""

    async def test_set_and_get(self):
        store = MockStore()

        await store.set("key1", "value1")
        value = await store.get("key1")

        assert value == "value1"

    async def test_initial_data(self):
        store = MockStore(initial_data={"existing": "data"})

        value = await store.get("existing")
        assert value == "data"

    async def test_delete(self):
        store = MockStore(initial_data={"key": "value"})

        await store.delete("key")
        value = await store.get("key")

        assert value is None

    async def test_exists(self):
        store = MockStore(initial_data={"key": "value"})

        assert await store.exists("key") is True
        assert await store.exists("nonexistent") is False

    async def test_reset(self):
        store = MockStore(initial_data={"initial": "data"})

        await store.set("new", "value")
        store.reset()

        assert await store.get("new") is None
        assert await store.get("initial") == "data"


class TestMockSecretsProvider:
    """Tests for MockSecretsProvider."""

    async def test_get_secret(self):
        secrets = MockSecretsProvider(secrets={
            "api_key": "secret-123",
            "password": "hunter2",
        })

        value = await secrets.get_secret("api_key")
        assert value == "secret-123"

    async def test_missing_secret_raises(self):
        secrets = MockSecretsProvider(raise_on_missing=True)

        with pytest.raises(KeyError):
            await secrets.get_secret("nonexistent")

    async def test_missing_secret_returns_empty(self):
        secrets = MockSecretsProvider(raise_on_missing=False)

        value = await secrets.get_secret("nonexistent")
        assert value == ""

    async def test_call_tracking(self):
        secrets = MockSecretsProvider(secrets={"key": "value"})

        await secrets.get_secret("key")
        await secrets.get_secret("key")

        assert secrets.calls == ["key", "key"]


class TestMockEventEmitter:
    """Tests for MockEventEmitter."""

    async def test_emit_captures_events(self):
        emitter = MockEventEmitter()

        class FakeEvent:
            event_type = "test_event"
            data = {"key": "value"}

        await emitter.emit(FakeEvent())

        assert len(emitter.events) == 1
        assert emitter.events[0].event_type == "test_event"

    async def test_get_events_by_type(self):
        emitter = MockEventEmitter()

        class Event1:
            event_type = "type1"

        class Event2:
            event_type = "type2"

        await emitter.emit(Event1())
        await emitter.emit(Event2())
        await emitter.emit(Event1())

        type1_events = emitter.get_events_by_type("type1")
        assert len(type1_events) == 2

    async def test_assert_event_emitted(self):
        emitter = MockEventEmitter()

        class FakeEvent:
            event_type = "test"

        with pytest.raises(AssertionError):
            emitter.assert_event_emitted("test")

        await emitter.emit(FakeEvent())
        emitter.assert_event_emitted("test")  # Should not raise


class TestTestRunConfig:
    """Tests for TestRunConfig."""

    def test_defaults(self):
        config = TestRunConfig()

        assert config.timeout_seconds == 30.0
        assert config.capture_events is True
        assert config.fail_on_error is True
        assert config.mock_llm_responses == []

    def test_custom_values(self):
        config = TestRunConfig(
            timeout_seconds=60.0,
            mock_llm_responses=["response1", "response2"],
            mock_secrets={"key": "value"},
        )

        assert config.timeout_seconds == 60.0
        assert config.mock_llm_responses == ["response1", "response2"]
        assert config.mock_secrets == {"key": "value"}


class TestHandlerTestCase:
    """Tests for HandlerTestCase."""

    def test_create_basic(self):
        case = HandlerTestCase(
            name="test_case",
            input_data={"x": 1},
            expected_output={"y": 2},
        )

        assert case.name == "test_case"
        assert case.input_data == {"x": 1}
        assert case.expected_output == {"y": 2}
        assert case.should_fail is False

    def test_create_failure_case(self):
        case = HandlerTestCase(
            name="error_case",
            input_data={},
            should_fail=True,
            expected_error="Missing field",
        )

        assert case.should_fail is True
        assert case.expected_error == "Missing field"

    def test_default_expected_output(self):
        case = HandlerTestCase(
            name="test",
            input_data={},
        )

        assert case.expected_output == {}
