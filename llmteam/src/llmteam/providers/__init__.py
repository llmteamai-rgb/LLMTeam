"""
LLM Providers module.

Ready-to-use LLM provider implementations for common platforms.

Install with optional dependencies:
    pip install llmteam-ai[providers]  # OpenAI + Anthropic
    pip install llmteam-ai[aws]        # Bedrock

Usage:
    from llmteam.providers import OpenAIProvider

    provider = OpenAIProvider(model="gpt-4o")
    response = await provider.complete("Hello, world!")

Environment Variables:
    OPENAI_API_KEY - OpenAI API key
    ANTHROPIC_API_KEY - Anthropic API key
    AZURE_OPENAI_API_KEY - Azure OpenAI API key
    AZURE_OPENAI_ENDPOINT - Azure OpenAI endpoint URL
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
]
