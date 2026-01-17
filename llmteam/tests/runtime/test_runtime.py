"""
Tests for Runtime Context module.
"""

import pytest
from typing import Any

from llmteam.runtime import (
    RuntimeContext,
    StepContext,
    RuntimeContextManager,
    StoreRegistry,
    ClientRegistry,
    LLMRegistry,
    ResourceNotFoundError,
    RuntimeContextError,
    current_runtime,
    get_current_runtime,
)


# === Mock implementations ===


class MockStore:
    """Mock store implementation."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    async def get(self, key: str) -> Any:
        return self._data.get(key)

    async def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)


class MockClient:
    """Mock client implementation."""

    async def request(self, method: str, path: str, **kwargs: Any) -> Any:
        return {"method": method, "path": path, **kwargs}


class MockLLMProvider:
    """Mock LLM provider implementation."""

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        return f"Response to: {prompt}"


class MockSecretsProvider:
    """Mock secrets provider implementation."""

    def __init__(self) -> None:
        self._secrets = {"api_key": "secret_value", "db_password": "db_secret"}

    async def get_secret(self, secret_id: str) -> str:
        if secret_id not in self._secrets:
            raise KeyError(f"Secret '{secret_id}' not found")
        return self._secrets[secret_id]


# === Tests ===


class TestStoreRegistry:
    """Tests for StoreRegistry."""

    def test_register_and_get(self) -> None:
        registry = StoreRegistry()
        store = MockStore()

        registry.register("main", store)
        assert registry.get("main") is store

    def test_get_not_found(self) -> None:
        registry = StoreRegistry()

        with pytest.raises(ResourceNotFoundError) as exc_info:
            registry.get("nonexistent")
        assert "Store 'nonexistent' not found" in str(exc_info.value)

    def test_has(self) -> None:
        registry = StoreRegistry()
        store = MockStore()

        assert not registry.has("main")
        registry.register("main", store)
        assert registry.has("main")

    def test_list(self) -> None:
        registry = StoreRegistry()
        registry.register("store1", MockStore())
        registry.register("store2", MockStore())

        stores = registry.list()
        assert "store1" in stores
        assert "store2" in stores

    def test_unregister(self) -> None:
        registry = StoreRegistry()
        registry.register("main", MockStore())
        assert registry.has("main")

        registry.unregister("main")
        assert not registry.has("main")

    def test_clear(self) -> None:
        registry = StoreRegistry()
        registry.register("store1", MockStore())
        registry.register("store2", MockStore())

        registry.clear()
        assert registry.list() == []


class TestClientRegistry:
    """Tests for ClientRegistry."""

    def test_register_and_get(self) -> None:
        registry = ClientRegistry()
        client = MockClient()

        registry.register("http", client)
        assert registry.get("http") is client

    def test_get_not_found(self) -> None:
        registry = ClientRegistry()

        with pytest.raises(ResourceNotFoundError):
            registry.get("nonexistent")


class TestLLMRegistry:
    """Tests for LLMRegistry."""

    def test_register_and_get(self) -> None:
        registry = LLMRegistry()
        provider = MockLLMProvider()

        registry.register("gpt4", provider)
        assert registry.get("gpt4") is provider

    def test_get_not_found(self) -> None:
        registry = LLMRegistry()

        with pytest.raises(ResourceNotFoundError):
            registry.get("nonexistent")


class TestRuntimeContext:
    """Tests for RuntimeContext."""

    def test_create_context(self) -> None:
        ctx = RuntimeContext(
            tenant_id="acme",
            instance_id="inst_123",
            run_id="run_456",
            segment_id="pipeline_1",
        )

        assert ctx.tenant_id == "acme"
        assert ctx.instance_id == "inst_123"
        assert ctx.run_id == "run_456"
        assert ctx.segment_id == "pipeline_1"

    def test_resolve_store(self) -> None:
        ctx = RuntimeContext(
            tenant_id="acme",
            instance_id="inst_123",
            run_id="run_456",
            segment_id="pipeline_1",
        )

        store = MockStore()
        ctx.stores.register("main", store)

        assert ctx.resolve_store("main") is store

    def test_resolve_client(self) -> None:
        ctx = RuntimeContext(
            tenant_id="acme",
            instance_id="inst_123",
            run_id="run_456",
            segment_id="pipeline_1",
        )

        client = MockClient()
        ctx.clients.register("http", client)

        assert ctx.resolve_client("http") is client

    def test_resolve_llm(self) -> None:
        ctx = RuntimeContext(
            tenant_id="acme",
            instance_id="inst_123",
            run_id="run_456",
            segment_id="pipeline_1",
        )

        provider = MockLLMProvider()
        ctx.llms.register("gpt4", provider)

        assert ctx.resolve_llm("gpt4") is provider

    async def test_resolve_secret(self) -> None:
        secrets = MockSecretsProvider()
        ctx = RuntimeContext(
            tenant_id="acme",
            instance_id="inst_123",
            run_id="run_456",
            segment_id="pipeline_1",
            secrets=secrets,
        )

        secret = await ctx.resolve_secret("api_key")
        assert secret == "secret_value"

    async def test_resolve_secret_no_provider(self) -> None:
        ctx = RuntimeContext(
            tenant_id="acme",
            instance_id="inst_123",
            run_id="run_456",
            segment_id="pipeline_1",
        )

        with pytest.raises(ResourceNotFoundError):
            await ctx.resolve_secret("api_key")

    def test_child_context(self) -> None:
        ctx = RuntimeContext(
            tenant_id="acme",
            instance_id="inst_123",
            run_id="run_456",
            segment_id="pipeline_1",
        )

        step_ctx = ctx.child_context("step_1")

        assert isinstance(step_ctx, StepContext)
        assert step_ctx.step_id == "step_1"
        assert step_ctx.runtime is ctx

    def test_copy(self) -> None:
        ctx = RuntimeContext(
            tenant_id="acme",
            instance_id="inst_123",
            run_id="run_456",
            segment_id="pipeline_1",
        )

        copied = ctx.copy(run_id="run_789")

        assert copied.tenant_id == "acme"
        assert copied.run_id == "run_789"
        assert copied is not ctx


class TestStepContext:
    """Tests for StepContext."""

    def test_properties(self) -> None:
        runtime = RuntimeContext(
            tenant_id="acme",
            instance_id="inst_123",
            run_id="run_456",
            segment_id="pipeline_1",
        )

        ctx = StepContext(runtime=runtime, step_id="step_1")

        assert ctx.tenant_id == "acme"
        assert ctx.instance_id == "inst_123"
        assert ctx.run_id == "run_456"
        assert ctx.segment_id == "pipeline_1"
        assert ctx.step_id == "step_1"

    def test_get_store(self) -> None:
        runtime = RuntimeContext(
            tenant_id="acme",
            instance_id="inst_123",
            run_id="run_456",
            segment_id="pipeline_1",
        )
        store = MockStore()
        runtime.stores.register("main", store)

        ctx = StepContext(runtime=runtime, step_id="step_1")

        assert ctx.get_store("main") is store

    def test_state(self) -> None:
        runtime = RuntimeContext(
            tenant_id="acme",
            instance_id="inst_123",
            run_id="run_456",
            segment_id="pipeline_1",
        )
        ctx = StepContext(runtime=runtime, step_id="step_1")

        ctx.set_state("key", "value")
        assert ctx.get_state("key") == "value"
        assert ctx.get_state("missing", "default") == "default"

        ctx.clear_state()
        assert ctx.get_state("key") is None


class TestRuntimeContextManager:
    """Tests for RuntimeContextManager."""

    def test_sync_context_manager(self) -> None:
        ctx = RuntimeContext(
            tenant_id="acme",
            instance_id="inst_123",
            run_id="run_456",
            segment_id="pipeline_1",
        )

        with RuntimeContextManager(ctx) as active_ctx:
            assert active_ctx is ctx
            assert current_runtime.get() is ctx

        assert current_runtime.get() is None

    async def test_async_context_manager(self) -> None:
        ctx = RuntimeContext(
            tenant_id="acme",
            instance_id="inst_123",
            run_id="run_456",
            segment_id="pipeline_1",
        )

        async with RuntimeContextManager(ctx) as active_ctx:
            assert active_ctx is ctx
            assert current_runtime.get() is ctx

        assert current_runtime.get() is None

    def test_get_current_runtime(self) -> None:
        ctx = RuntimeContext(
            tenant_id="acme",
            instance_id="inst_123",
            run_id="run_456",
            segment_id="pipeline_1",
        )

        with RuntimeContextManager(ctx):
            assert get_current_runtime() is ctx

    def test_get_current_runtime_no_context(self) -> None:
        with pytest.raises(RuntimeContextError):
            get_current_runtime()

    def test_nested_contexts(self) -> None:
        ctx1 = RuntimeContext(
            tenant_id="acme",
            instance_id="inst_1",
            run_id="run_1",
            segment_id="pipeline_1",
        )
        ctx2 = RuntimeContext(
            tenant_id="acme",
            instance_id="inst_2",
            run_id="run_2",
            segment_id="pipeline_2",
        )

        with RuntimeContextManager(ctx1):
            assert current_runtime.get() is ctx1

            with RuntimeContextManager(ctx2):
                assert current_runtime.get() is ctx2

            assert current_runtime.get() is ctx1

        assert current_runtime.get() is None
