"""
LLM Providers module.

Ready-to-use LLM provider implementations for common platforms.

Install with optional dependencies:
    pip install llmteam-ai[providers]  # OpenAI + Anthropic
    pip install llmteam-ai[aws]        # Bedrock
    pip install google-cloud-aiplatform  # Vertex AI
    pip install litellm                 # LiteLLM (100+ providers)

Usage:
    from llmteam.providers import OpenAIProvider

    provider = OpenAIProvider(model="gpt-4o")
    response = await provider.complete("Hello, world!")

    # Or use LiteLLM for unified access
    from llmteam.providers import LiteLLMProvider
    provider = LiteLLMProvider(model="gpt-4")  # Works with any provider

Environment Variables:
    OPENAI_API_KEY - OpenAI API key
    ANTHROPIC_API_KEY - Anthropic API key
    AZURE_OPENAI_API_KEY - Azure OpenAI API key
    AZURE_OPENAI_ENDPOINT - Azure OpenAI endpoint URL
    GOOGLE_APPLICATION_CREDENTIALS - Google Cloud credentials
    GOOGLE_CLOUD_PROJECT - GCP project ID
    OLLAMA_HOST - Ollama server URL (default: http://localhost:11434)
"""

from llmteam.providers.base import (
    BaseLLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMModelNotFoundError,
)

from llmteam.providers.openai import OpenAIProvider
from llmteam.providers.anthropic import AnthropicProvider
from llmteam.providers.azure import AzureOpenAIProvider
from llmteam.providers.bedrock import BedrockProvider
from llmteam.providers.vertex import VertexAIProvider
from llmteam.providers.ollama import OllamaProvider
from llmteam.providers.litellm import LiteLLMProvider

__all__ = [
    # Base
    "BaseLLMProvider",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMAuthenticationError",
    "LLMModelNotFoundError",
    # Providers
    "OpenAIProvider",
    "AnthropicProvider",
    "AzureOpenAIProvider",
    "BedrockProvider",
    "VertexAIProvider",
    "OllamaProvider",
    "LiteLLMProvider",
]
