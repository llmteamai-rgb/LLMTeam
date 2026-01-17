"""Tests for secrets management module."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


class TestSecretMetadata:
    """Tests for SecretMetadata."""

    def test_create_metadata(self):
        """Create metadata with basic fields."""
        from llmteam.secrets import SecretMetadata

        metadata = SecretMetadata(
            name="test-secret",
            version="v1",
            provider="Test",
        )

        assert metadata.name == "test-secret"
        assert metadata.version == "v1"
        assert metadata.provider == "Test"
        assert metadata.tags == {}

    def test_metadata_with_timestamps(self):
        """Create metadata with timestamps."""
        from llmteam.secrets import SecretMetadata

        now = datetime.utcnow()
        metadata = SecretMetadata(
            name="test-secret",
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(days=30),
        )

        assert metadata.created_at == now
        assert metadata.expires_at is not None


class TestSecretValue:
    """Tests for SecretValue."""

    def test_create_secret_value(self):
        """Create secret value."""
        from llmteam.secrets import SecretValue, SecretMetadata

        metadata = SecretMetadata(name="test")
        secret = SecretValue(value="secret-data", metadata=metadata)

        assert secret.value == "secret-data"
        assert secret.metadata.name == "test"
        assert secret.binary is False

    def test_is_expired_false(self):
        """Secret is not expired when no expiry."""
        from llmteam.secrets import SecretValue, SecretMetadata

        metadata = SecretMetadata(name="test")
        secret = SecretValue(value="data", metadata=metadata)

        assert secret.is_expired is False

    def test_is_expired_true(self):
        """Secret is expired when past expiry date."""
        from llmteam.secrets import SecretValue, SecretMetadata

        metadata = SecretMetadata(
            name="test",
            expires_at=datetime.utcnow() - timedelta(days=1),
        )
        secret = SecretValue(value="data", metadata=metadata)

        assert secret.is_expired is True


class TestEnvSecretsProvider:
    """Tests for EnvSecretsProvider."""

    def test_initialization_default(self):
        """Initialize with default values."""
        from llmteam.secrets import EnvSecretsProvider

        provider = EnvSecretsProvider()

        assert provider.prefix == ""
        assert provider.case_transform == "upper"
        assert provider.provider_name == "Environment"

    def test_initialization_with_prefix(self):
        """Initialize with prefix."""
        from llmteam.secrets import EnvSecretsProvider

        provider = EnvSecretsProvider(prefix="APP_")

        assert provider.prefix == "APP_"

    async def test_get_secret_exists(self, monkeypatch):
        """Get secret that exists."""
        from llmteam.secrets import EnvSecretsProvider

        monkeypatch.setenv("TEST_API_KEY", "secret-value")
        provider = EnvSecretsProvider()

        secret = await provider.get_secret("test_api_key")

        assert secret.value == "secret-value"

    async def test_get_secret_not_found(self, monkeypatch):
        """Get secret that doesn't exist."""
        from llmteam.secrets import EnvSecretsProvider, SecretNotFoundError

        monkeypatch.delenv("NONEXISTENT_KEY", raising=False)
        provider = EnvSecretsProvider()

        with pytest.raises(SecretNotFoundError):
            await provider.get_secret("nonexistent_key")

    async def test_get_secret_with_prefix(self, monkeypatch):
        """Get secret with prefix."""
        from llmteam.secrets import EnvSecretsProvider

        monkeypatch.setenv("APP_DATABASE_URL", "postgres://...")
        provider = EnvSecretsProvider(prefix="APP_")

        secret = await provider.get_secret("database_url")

        assert secret.value == "postgres://..."

    async def test_set_secret(self, monkeypatch):
        """Set secret in environment."""
        from llmteam.secrets import EnvSecretsProvider
        import os

        monkeypatch.delenv("TEST_NEW_SECRET", raising=False)
        provider = EnvSecretsProvider()

        metadata = await provider.set_secret("test_new_secret", "new-value")

        assert os.environ.get("TEST_NEW_SECRET") == "new-value"
        assert metadata.name == "test_new_secret"

    async def test_delete_secret(self, monkeypatch):
        """Delete secret from environment."""
        from llmteam.secrets import EnvSecretsProvider
        import os

        monkeypatch.setenv("TEST_TO_DELETE", "value")
        provider = EnvSecretsProvider()

        result = await provider.delete_secret("test_to_delete")

        assert result is True
        assert "TEST_TO_DELETE" not in os.environ

    async def test_delete_nonexistent(self, monkeypatch):
        """Delete nonexistent secret returns False."""
        from llmteam.secrets import EnvSecretsProvider

        monkeypatch.delenv("NONEXISTENT", raising=False)
        provider = EnvSecretsProvider()

        result = await provider.delete_secret("nonexistent")

        assert result is False

    async def test_secret_exists(self, monkeypatch):
        """Check if secret exists."""
        from llmteam.secrets import EnvSecretsProvider

        monkeypatch.setenv("EXISTS_KEY", "value")
        monkeypatch.delenv("NOT_EXISTS_KEY", raising=False)
        provider = EnvSecretsProvider()

        assert await provider.secret_exists("exists_key") is True
        assert await provider.secret_exists("not_exists_key") is False


class TestSecretsManager:
    """Tests for SecretsManager."""

    async def test_get_returns_value(self):
        """Get returns just the value."""
        from llmteam.secrets import SecretsManager, EnvSecretsProvider

        provider = EnvSecretsProvider()
        manager = SecretsManager(provider=provider)

        with patch.object(provider, "get_secret") as mock_get:
            from llmteam.secrets import SecretValue, SecretMetadata

            mock_get.return_value = SecretValue(
                value="secret",
                metadata=SecretMetadata(name="test"),
            )

            result = await manager.get("test")

            assert result == "secret"

    async def test_get_with_default(self):
        """Get with default for missing secret."""
        from llmteam.secrets import SecretsManager, EnvSecretsProvider, SecretNotFoundError

        provider = EnvSecretsProvider()
        manager = SecretsManager(provider=provider)

        with patch.object(provider, "get_secret") as mock_get:
            mock_get.side_effect = SecretNotFoundError("test")

            result = await manager.get("test", default="default-value")

            assert result == "default-value"

    async def test_get_secret_returns_full_value(self):
        """Get_secret returns full SecretValue."""
        from llmteam.secrets import SecretsManager, EnvSecretsProvider

        provider = EnvSecretsProvider()
        manager = SecretsManager(provider=provider)

        with patch.object(provider, "get_secret") as mock_get:
            from llmteam.secrets import SecretValue, SecretMetadata

            expected = SecretValue(
                value="secret",
                metadata=SecretMetadata(name="test", version="v1"),
            )
            mock_get.return_value = expected

            result = await manager.get_secret("test")

            assert result.value == "secret"
            assert result.metadata.version == "v1"

    async def test_fallback_provider(self):
        """Uses fallback provider when primary fails."""
        from llmteam.secrets import SecretsManager, EnvSecretsProvider, SecretNotFoundError

        primary = EnvSecretsProvider(prefix="PRIMARY_")
        fallback = EnvSecretsProvider(prefix="FALLBACK_")
        manager = SecretsManager(provider=primary, fallback_providers=[fallback])

        with patch.object(primary, "get_secret") as mock_primary:
            with patch.object(fallback, "get_secret") as mock_fallback:
                from llmteam.secrets import SecretValue, SecretMetadata

                mock_primary.side_effect = SecretNotFoundError("test")
                mock_fallback.return_value = SecretValue(
                    value="fallback-secret",
                    metadata=SecretMetadata(name="test"),
                )

                result = await manager.get("test")

                assert result == "fallback-secret"


class TestCachingSecretsManager:
    """Tests for CachingSecretsManager."""

    async def test_caches_secrets(self):
        """Caches secrets after first fetch."""
        from llmteam.secrets import CachingSecretsManager, EnvSecretsProvider, SecretsCacheConfig

        provider = EnvSecretsProvider()
        config = SecretsCacheConfig(ttl_seconds=60)
        manager = CachingSecretsManager(provider=provider, cache_config=config)

        with patch.object(provider, "get_secret") as mock_get:
            from llmteam.secrets import SecretValue, SecretMetadata

            mock_get.return_value = SecretValue(
                value="secret",
                metadata=SecretMetadata(name="test"),
            )

            # First call - hits provider
            result1 = await manager.get("test")
            # Second call - uses cache
            result2 = await manager.get("test")

            assert result1 == "secret"
            assert result2 == "secret"
            assert mock_get.call_count == 1  # Only called once

    async def test_cache_disabled(self):
        """Cache can be disabled."""
        from llmteam.secrets import CachingSecretsManager, EnvSecretsProvider, SecretsCacheConfig

        provider = EnvSecretsProvider()
        config = SecretsCacheConfig(enabled=False)
        manager = CachingSecretsManager(provider=provider, cache_config=config)

        with patch.object(provider, "get_secret") as mock_get:
            from llmteam.secrets import SecretValue, SecretMetadata

            mock_get.return_value = SecretValue(
                value="secret",
                metadata=SecretMetadata(name="test"),
            )

            await manager.get("test")
            await manager.get("test")

            assert mock_get.call_count == 2  # Called twice (no caching)

    def test_invalidate_specific(self):
        """Invalidate specific secret."""
        from llmteam.secrets import CachingSecretsManager, EnvSecretsProvider

        provider = EnvSecretsProvider()
        manager = CachingSecretsManager(provider=provider)

        # Manually populate cache
        manager._cache["test:latest"] = MagicMock()
        manager._cache["other:latest"] = MagicMock()

        count = manager.invalidate("test")

        assert count == 1
        assert "test:latest" not in manager._cache
        assert "other:latest" in manager._cache

    def test_invalidate_all(self):
        """Invalidate all secrets."""
        from llmteam.secrets import CachingSecretsManager, EnvSecretsProvider

        provider = EnvSecretsProvider()
        manager = CachingSecretsManager(provider=provider)

        manager._cache["test1:latest"] = MagicMock()
        manager._cache["test2:latest"] = MagicMock()

        count = manager.invalidate()

        assert count == 2
        assert len(manager._cache) == 0

    def test_cache_stats(self):
        """Get cache statistics."""
        from llmteam.secrets import CachingSecretsManager, EnvSecretsProvider, SecretsCacheConfig

        config = SecretsCacheConfig(ttl_seconds=300, max_entries=100)
        provider = EnvSecretsProvider()
        manager = CachingSecretsManager(provider=provider, cache_config=config)

        stats = manager.cache_stats()

        assert stats["total_entries"] == 0
        assert stats["max_entries"] == 100
        assert stats["ttl_seconds"] == 300


class TestSecretsExports:
    """Test that secrets module exports are correct."""

    def test_all_classes_exported(self):
        """All classes are exported from the module."""
        from llmteam.secrets import (
            SecretsProvider,
            SecretValue,
            SecretMetadata,
            SecretsError,
            SecretNotFoundError,
            SecretAccessDeniedError,
            SecretsManager,
            CachingSecretsManager,
            SecretsCacheConfig,
            EnvSecretsProvider,
            VaultProvider,
            VaultConfig,
            AWSSecretsProvider,
            AWSSecretsConfig,
            AzureKeyVaultProvider,
            AzureKeyVaultConfig,
        )

        assert SecretsProvider is not None
        assert SecretValue is not None
        assert SecretsManager is not None
        assert EnvSecretsProvider is not None
        assert VaultProvider is not None
        assert AWSSecretsProvider is not None
        assert AzureKeyVaultProvider is not None
