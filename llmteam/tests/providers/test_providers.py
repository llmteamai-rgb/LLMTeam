"""
Tests for LLM Providers module.

Tests the base provider functionality and mock providers.
Real API tests require actual API keys and are skipped by default.
"""

import pytest
from llmteam.providers.base import (
    BaseLLMProvider,
    CompletionConfig,
    LLMProviderError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMModelNotFoundError,
)
from llmteam.providers import (
    OpenAIProvider,
    AnthropicProvider,
    AzureOpenAIProvider,
    BedrockProvider,
)


class TestCompletionConfig:
    """Tests for CompletionConfig."""

    def test_defaults(self):
        config = CompletionConfig()
        assert config.max_tokens == 4096
        assert config.temperature == 0.7
        assert config.top_p == 1.0
        assert config.stop is None
        assert config.presence_penalty == 0.0
        assert config.frequency_penalty == 0.0
        assert config.extra == {}

    def test_custom_values(self):
        config = CompletionConfig(
            max_tokens=1000,
            temperature=0.5,
            stop=["END"],
        )
        assert config.max_tokens == 1000
        assert config.temperature == 0.5
        assert config.stop == ["END"]


class TestLLMProviderErrors:
    """Tests for provider error classes."""

    def test_base_error(self):
        error = LLMProviderError("Test error", provider="TestProvider")
        assert str(error) == "Test error"
        assert error.provider == "TestProvider"
        assert error.details == {}

    def test_rate_limit_error(self):
        error = LLMRateLimitError(retry_after=60.0, provider="OpenAI")
        assert "Rate limit" in str(error)
        assert error.retry_after == 60.0
        assert error.provider == "OpenAI"

    def test_authentication_error(self):
        error = LLMAuthenticationError(provider="Anthropic")
        assert "Authentication" in str(error)
        assert error.provider == "Anthropic"

    def test_model_not_found_error(self):
        error = LLMModelNotFoundError(model="gpt-5", provider="OpenAI")
        assert "gpt-5" in str(error)
        assert error.model == "gpt-5"


class TestOpenAIProvider:
    """Tests for OpenAI provider initialization."""

    def test_create_default(self):
        provider = OpenAIProvider()
        assert provider.model == "gpt-4o"
        assert provider._client is None

    def test_create_custom_model(self):
        provider = OpenAIProvider(model="gpt-4o-mini")
        assert provider.model == "gpt-4o-mini"

    def test_create_with_api_key(self):
        provider = OpenAIProvider(api_key="test-key")
        assert provider._api_key == "test-key"

    def test_supported_models(self):
        assert "gpt-4o" in OpenAIProvider.SUPPORTED_MODELS
        assert "gpt-4o-mini" in OpenAIProvider.SUPPORTED_MODELS
        assert "gpt-4" in OpenAIProvider.SUPPORTED_MODELS

    def test_provider_name(self):
        provider = OpenAIProvider()
        assert provider.provider_name == "OpenAIProvider"


class TestAnthropicProvider:
    """Tests for Anthropic provider initialization."""

    def test_create_default(self):
        provider = AnthropicProvider()
        assert provider.model == "claude-3-5-sonnet-20241022"
        assert provider._client is None

    def test_create_custom_model(self):
        provider = AnthropicProvider(model="claude-3-opus-20240229")
        assert provider.model == "claude-3-opus-20240229"

    def test_supported_models(self):
        assert "claude-3-5-sonnet-20241022" in AnthropicProvider.SUPPORTED_MODELS
        assert "claude-3-5-haiku-20241022" in AnthropicProvider.SUPPORTED_MODELS

    def test_provider_name(self):
        provider = AnthropicProvider()
        assert provider.provider_name == "AnthropicProvider"


class TestAzureOpenAIProvider:
    """Tests for Azure OpenAI provider initialization."""

    def test_create_with_deployment(self):
        provider = AzureOpenAIProvider(
            deployment_name="my-deployment",
            azure_endpoint="https://test.openai.azure.com/",
        )
        assert provider._deployment_name == "my-deployment"
        assert provider._azure_endpoint == "https://test.openai.azure.com/"

    def test_default_api_version(self):
        provider = AzureOpenAIProvider(deployment_name="test")
        assert provider._api_version == "2024-02-01"

    def test_custom_api_version(self):
        provider = AzureOpenAIProvider(
            deployment_name="test",
            api_version="2023-12-01",
        )
        assert provider._api_version == "2023-12-01"

    def test_provider_name(self):
        provider = AzureOpenAIProvider(deployment_name="test")
        assert provider.provider_name == "AzureOpenAIProvider"


class TestBedrockProvider:
    """Tests for AWS Bedrock provider initialization."""

    def test_create_default(self):
        provider = BedrockProvider()
        assert provider._model_id == "anthropic.claude-3-5-sonnet-20241022-v2:0"

    def test_create_custom_model(self):
        provider = BedrockProvider(model_id="meta.llama3-2-90b-instruct-v1:0")
        assert provider._model_id == "meta.llama3-2-90b-instruct-v1:0"

    def test_default_region(self):
        provider = BedrockProvider()
        assert provider._region_name == "us-east-1"

    def test_custom_region(self):
        provider = BedrockProvider(region_name="eu-west-1")
        assert provider._region_name == "eu-west-1"

    def test_is_claude_model(self):
        provider = BedrockProvider(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0")
        assert provider._is_claude_model() is True
        assert provider._is_llama_model() is False
        assert provider._is_titan_model() is False

    def test_is_llama_model(self):
        provider = BedrockProvider(model_id="meta.llama3-2-90b-instruct-v1:0")
        assert provider._is_claude_model() is False
        assert provider._is_llama_model() is True
        assert provider._is_titan_model() is False

    def test_is_titan_model(self):
        provider = BedrockProvider(model_id="amazon.titan-text-premier-v1:0")
        assert provider._is_claude_model() is False
        assert provider._is_llama_model() is False
        assert provider._is_titan_model() is True

    def test_provider_name(self):
        provider = BedrockProvider()
        assert provider.provider_name == "BedrockProvider"

    def test_supported_models(self):
        assert "anthropic.claude-3-5-sonnet-20241022-v2:0" in BedrockProvider.SUPPORTED_MODELS
        assert "meta.llama3-2-90b-instruct-v1:0" in BedrockProvider.SUPPORTED_MODELS
        assert "amazon.titan-text-premier-v1:0" in BedrockProvider.SUPPORTED_MODELS
