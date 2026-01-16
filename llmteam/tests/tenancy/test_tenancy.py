"""
Tests for tenancy module.
"""

import pytest
import asyncio

from llmteam.tenancy import (
    TenantConfig,
    TenantTier,
    TenantLimits,
    TenantContext,
    TenantManager,
    TenantIsolatedStore,
    current_tenant,
    TenantNotFoundError,
    TenantLimitExceededError,
    TenantFeatureDisabledError,
    TenantContextError,
)
from llmteam.tenancy.stores import MemoryTenantStore, MemoryKeyValueStore


class TestTenantConfig:
    """Tests for TenantConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = TenantConfig(tenant_id="test", name="Test")
        
        assert config.tenant_id == "test"
        assert config.name == "Test"
        assert config.tier == TenantTier.FREE
        assert config.is_active
    
    def test_effective_limits_default(self):
        """Test getting effective limits with defaults."""
        config = TenantConfig(
            tenant_id="test",
            name="Test",
            tier=TenantTier.FREE,
        )
        
        limits = config.get_effective_limits()
        
        assert limits.max_concurrent_pipelines == 1
        assert limits.max_agents_per_pipeline == 5
        assert "basic_agents" in limits.features
    
    def test_effective_limits_override(self):
        """Test getting effective limits with overrides."""
        config = TenantConfig(
            tenant_id="test",
            name="Test",
            tier=TenantTier.FREE,
            max_concurrent_pipelines=5,  # Override
        )
        
        limits = config.get_effective_limits()
        
        assert limits.max_concurrent_pipelines == 5  # Overridden
        assert limits.max_agents_per_pipeline == 5   # Default
    
    def test_feature_override(self):
        """Test feature enable/disable."""
        config = TenantConfig(
            tenant_id="test",
            name="Test",
            tier=TenantTier.FREE,
            features_enabled={"custom_feature"},
            features_disabled={"basic_agents"},
        )
        
        limits = config.get_effective_limits()
        
        assert "custom_feature" in limits.features
        assert "basic_agents" not in limits.features
    
    def test_action_allowed_no_restrictions(self):
        """Test action allowed with no restrictions."""
        config = TenantConfig(tenant_id="test", name="Test")
        
        assert config.is_action_allowed("any_action")
    
    def test_action_blocked(self):
        """Test action blocked."""
        config = TenantConfig(
            tenant_id="test",
            name="Test",
            blocked_actions={"dangerous_action"},
        )
        
        assert not config.is_action_allowed("dangerous_action")
        assert config.is_action_allowed("safe_action")
    
    def test_action_whitelist(self):
        """Test action whitelist."""
        config = TenantConfig(
            tenant_id="test",
            name="Test",
            allowed_actions={"safe_action"},
        )
        
        assert config.is_action_allowed("safe_action")
        assert not config.is_action_allowed("other_action")
    
    def test_serialization(self):
        """Test to_dict and from_dict."""
        config = TenantConfig(
            tenant_id="test",
            name="Test Tenant",
            tier=TenantTier.PROFESSIONAL,
            features_enabled={"extra"},
        )
        
        data = config.to_dict()
        restored = TenantConfig.from_dict(data)
        
        assert restored.tenant_id == config.tenant_id
        assert restored.tier == config.tier
        assert "extra" in restored.features_enabled


class TestTenantContext:
    """Tests for TenantContext."""
    
    def test_sync_context(self):
        """Test synchronous context manager."""
        assert current_tenant.get() == ""
        
        with TenantContext("test_tenant"):
            assert current_tenant.get() == "test_tenant"
        
        assert current_tenant.get() == ""
    
    @pytest.mark.asyncio
    async def test_async_context(self):
        """Test asynchronous context manager."""
        assert current_tenant.get() == ""
        
        async with TenantContext("async_tenant"):
            assert current_tenant.get() == "async_tenant"
        
        assert current_tenant.get() == ""
    
    def test_nested_contexts(self):
        """Test nested tenant contexts."""
        with TenantContext("outer"):
            assert current_tenant.get() == "outer"
            
            with TenantContext("inner"):
                assert current_tenant.get() == "inner"
            
            assert current_tenant.get() == "outer"
    
    def test_empty_tenant_id_raises(self):
        """Test that empty tenant_id raises error."""
        with pytest.raises(ValueError):
            TenantContext("")


class TestTenantManager:
    """Tests for TenantManager."""
    
    @pytest.fixture
    def manager(self):
        """Create a manager with memory store."""
        store = MemoryTenantStore()
        return TenantManager(store)
    
    @pytest.mark.asyncio
    async def test_create_tenant(self, manager):
        """Test creating a tenant."""
        config = TenantConfig(
            tenant_id="new_tenant",
            name="New Tenant",
            tier=TenantTier.STARTER,
        )
        
        result = await manager.create_tenant(config)
        
        assert result.tenant_id == "new_tenant"
        assert result.created_at is not None
    
    @pytest.mark.asyncio
    async def test_get_tenant(self, manager):
        """Test getting a tenant."""
        await manager.create_tenant(TenantConfig(
            tenant_id="test",
            name="Test",
        ))
        
        config = await manager.get_tenant("test")
        
        assert config.tenant_id == "test"
    
    @pytest.mark.asyncio
    async def test_get_tenant_not_found(self, manager):
        """Test getting non-existent tenant."""
        with pytest.raises(TenantNotFoundError):
            await manager.get_tenant("nonexistent")
    
    @pytest.mark.asyncio
    async def test_update_tenant(self, manager):
        """Test updating a tenant."""
        await manager.create_tenant(TenantConfig(
            tenant_id="test",
            name="Original",
        ))
        
        config = await manager.get_tenant("test")
        config.name = "Updated"
        await manager.update_tenant(config)
        
        updated = await manager.get_tenant("test")
        assert updated.name == "Updated"
    
    @pytest.mark.asyncio
    async def test_check_limit_within(self, manager):
        """Test check_limit within limits."""
        await manager.create_tenant(TenantConfig(
            tenant_id="test",
            name="Test",
            tier=TenantTier.FREE,  # max 1 concurrent pipeline
        ))
        
        result = await manager.check_limit("test", "concurrent_pipelines", 0)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_check_limit_exceeded(self, manager):
        """Test check_limit when exceeded."""
        await manager.create_tenant(TenantConfig(
            tenant_id="test",
            name="Test",
            tier=TenantTier.FREE,  # max 1 concurrent pipeline
        ))
        
        result = await manager.check_limit("test", "concurrent_pipelines", 1)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_enforce_limit(self, manager):
        """Test enforce_limit raises exception."""
        await manager.create_tenant(TenantConfig(
            tenant_id="test",
            name="Test",
            tier=TenantTier.FREE,
        ))
        
        with pytest.raises(TenantLimitExceededError) as exc_info:
            await manager.enforce_limit("test", "concurrent_pipelines", 1)
        
        assert exc_info.value.limit_type == "concurrent_pipelines"
    
    @pytest.mark.asyncio
    async def test_check_feature(self, manager):
        """Test feature checking."""
        await manager.create_tenant(TenantConfig(
            tenant_id="test",
            name="Test",
            tier=TenantTier.FREE,
        ))
        
        assert await manager.check_feature("test", "basic_agents") is True
        assert await manager.check_feature("test", "persistence") is False
    
    @pytest.mark.asyncio
    async def test_require_feature(self, manager):
        """Test require_feature raises exception."""
        await manager.create_tenant(TenantConfig(
            tenant_id="test",
            name="Test",
            tier=TenantTier.FREE,
        ))
        
        with pytest.raises(TenantFeatureDisabledError):
            await manager.require_feature("test", "persistence")
    
    @pytest.mark.asyncio
    async def test_context_method(self, manager):
        """Test creating context via manager."""
        await manager.create_tenant(TenantConfig(
            tenant_id="test",
            name="Test",
        ))
        
        async with manager.context("test"):
            assert current_tenant.get() == "test"


class TestTenantIsolatedStore:
    """Tests for TenantIsolatedStore."""
    
    @pytest.fixture
    def isolated_store(self):
        """Create an isolated store."""
        inner = MemoryKeyValueStore()
        return TenantIsolatedStore(inner)
    
    @pytest.mark.asyncio
    async def test_isolation_between_tenants(self, isolated_store):
        """Test that data is isolated between tenants."""
        # Tenant A sets data
        async with TenantContext("tenant_a"):
            await isolated_store.set("key", "value_a")
        
        # Tenant B sets same key
        async with TenantContext("tenant_b"):
            await isolated_store.set("key", "value_b")
        
        # Each tenant sees their own data
        async with TenantContext("tenant_a"):
            assert await isolated_store.get("key") == "value_a"
        
        async with TenantContext("tenant_b"):
            assert await isolated_store.get("key") == "value_b"
    
    @pytest.mark.asyncio
    async def test_no_context_raises(self, isolated_store):
        """Test that accessing without context raises error."""
        with pytest.raises(TenantContextError):
            await isolated_store.get("key")
    
    @pytest.mark.asyncio
    async def test_list_only_tenant_keys(self, isolated_store):
        """Test that list only returns tenant's keys."""
        async with TenantContext("tenant_a"):
            await isolated_store.set("key1", "v1")
            await isolated_store.set("key2", "v2")
        
        async with TenantContext("tenant_b"):
            await isolated_store.set("key3", "v3")
        
        async with TenantContext("tenant_a"):
            keys = await isolated_store.list()
            assert sorted(keys) == ["key1", "key2"]
