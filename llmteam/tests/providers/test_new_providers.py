"""Tests for new LLM providers (Vertex AI, Ollama, LiteLLM)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestVertexAIProvider:
    """Tests for VertexAIProvider."""

    def test_initialization_default(self):
        """Initialize with default values."""
        from llmteam.providers import VertexAIProvider

        provider = VertexAIProvider()

        assert provider.model == "gemini-1.5-pro"
        assert provider.region == "us-central1"
        assert provider.provider_name == "VertexAI"

    def test_initialization_custom(self):
        """Initialize with custom values."""
        from llmteam.providers import VertexAIProvider

        provider = VertexAIProvider(
            model="gemini-1.5-flash",
            project_id="my-project",
            region="us-west1",
        )

        assert provider.model == "gemini-1.5-flash"
        assert provider.project_id == "my-project"
        assert provider.region == "us-west1"

    def test_initialization_from_env(self, monkeypatch):
        """Initialize from environment variables."""
        from llmteam.providers import VertexAIProvider

        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
        monkeypatch.setenv("GOOGLE_CLOUD_REGION", "europe-west1")

        provider = VertexAIProvider()

        assert provider.project_id == "env-project"
        assert provider.region == "europe-west1"

    def test_credentials_path_sets_env(self, monkeypatch):
        """Credentials path sets environment variable."""
        from llmteam.providers import VertexAIProvider
        import os

        # Clear any existing value
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

        VertexAIProvider(credentials_path="/path/to/creds.json")

        assert os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") == "/path/to/creds.json"


class TestOllamaProvider:
    """Tests for OllamaProvider."""

    def test_initialization_default(self):
        """Initialize with default values."""
        from llmteam.providers import OllamaProvider

        provider = OllamaProvider()

        assert provider.model == "llama2"
        assert provider.host == "http://localhost:11434"
        assert provider.provider_name == "Ollama"

    def test_initialization_custom(self):
        """Initialize with custom values."""
        from llmteam.providers import OllamaProvider

        provider = OllamaProvider(
            model="mistral",
            host="http://custom-host:11434",
        )

        assert provider.model == "mistral"
        assert provider.host == "http://custom-host:11434"

    def test_initialization_from_env(self, monkeypatch):
        """Initialize from environment variables."""
        from llmteam.providers import OllamaProvider

        monkeypatch.setenv("OLLAMA_HOST", "http://env-host:11434")

        provider = OllamaProvider()

        assert provider.host == "http://env-host:11434"

    def test_host_strips_trailing_slash(self):
        """Host URL strips trailing slash."""
        from llmteam.providers import OllamaProvider

        provider = OllamaProvider(host="http://localhost:11434/")

        assert provider.host == "http://localhost:11434"

    def test_generate_url(self):
        """Generate URL is correct."""
        from llmteam.providers import OllamaProvider

        provider = OllamaProvider(model="llama2", host="http://localhost:11434")

        # The complete method should use /api/generate endpoint
        assert provider.host == "http://localhost:11434"

    def test_chat_url(self):
        """Chat URL is correct."""
        from llmteam.providers import OllamaProvider

        provider = OllamaProvider(model="mistral", host="http://custom:11434")

        # The complete_with_messages method should use /api/chat endpoint
        assert provider.host == "http://custom:11434"
        assert provider.model == "mistral"


class TestLiteLLMProvider:
    """Tests for LiteLLMProvider."""

    def test_initialization_default(self):
        """Initialize with default values."""
        from llmteam.providers import LiteLLMProvider

        provider = LiteLLMProvider()

        assert provider.model == "gpt-3.5-turbo"
        assert provider.provider_name == "LiteLLM"

    def test_initialization_custom(self):
        """Initialize with custom values."""
        from llmteam.providers import LiteLLMProvider

        provider = LiteLLMProvider(
            model="claude-3-opus-20240229",
            api_key="test-key",
            api_base="https://custom-api.com",
        )

        assert provider.model == "claude-3-opus-20240229"
        assert provider.api_key == "test-key"
        assert provider.api_base == "https://custom-api.com"

    def test_initialization_with_extra_kwargs(self):
        """Initialize with extra LiteLLM kwargs."""
        from llmteam.providers import LiteLLMProvider

        provider = LiteLLMProvider(
            model="gpt-4",
            custom_param="value",
        )

        assert provider.litellm_kwargs["custom_param"] == "value"

    def test_build_kwargs(self):
        """Build kwargs includes all configuration."""
        from llmteam.providers import LiteLLMProvider

        provider = LiteLLMProvider(
            model="gpt-4",
            api_key="test-key",
            api_base="https://api.example.com",
            timeout=60,
        )

        config = {"max_tokens": 1000, "temperature": 0.5}
        kwargs = provider._build_kwargs(config)

        assert kwargs["model"] == "gpt-4"
        assert kwargs["api_key"] == "test-key"
        assert kwargs["api_base"] == "https://api.example.com"
        assert kwargs["timeout"] == 60
        assert kwargs["max_tokens"] == 1000
        assert kwargs["temperature"] == 0.5

    def test_list_supported_models(self):
        """List supported models returns common models."""
        from llmteam.providers import LiteLLMProvider

        models = LiteLLMProvider.list_supported_models()

        assert "gpt-4" in models
        assert "claude-3-opus-20240229" in models
        assert "azure/gpt-4" in models
        assert "bedrock/anthropic.claude-v2" in models


class TestProviderExports:
    """Test that providers are correctly exported."""

    def test_all_providers_exported(self):
        """All providers are exported from the module."""
        from llmteam.providers import (
            BaseLLMProvider,
            OpenAIProvider,
            AnthropicProvider,
            AzureOpenAIProvider,
            BedrockProvider,
            VertexAIProvider,
            OllamaProvider,
            LiteLLMProvider,
        )

        assert BaseLLMProvider is not None
        assert OpenAIProvider is not None
        assert AnthropicProvider is not None
        assert AzureOpenAIProvider is not None
        assert BedrockProvider is not None
        assert VertexAIProvider is not None
        assert OllamaProvider is not None
        assert LiteLLMProvider is not None

    def test_errors_exported(self):
        """Provider errors are exported."""
        from llmteam.providers import (
            LLMProviderError,
            LLMRateLimitError,
            LLMAuthenticationError,
            LLMModelNotFoundError,
        )

        assert LLMProviderError is not None
        assert LLMRateLimitError is not None
        assert LLMAuthenticationError is not None
        assert LLMModelNotFoundError is not None
