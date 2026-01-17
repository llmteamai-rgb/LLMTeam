"""
Azure OpenAI LLM Provider.

Supports Azure-hosted OpenAI models.

Usage:
    from llmteam.providers import AzureOpenAIProvider

    provider = AzureOpenAIProvider(
        deployment_name="my-gpt4-deployment",
        azure_endpoint="https://my-resource.openai.azure.com/",
    )
    response = await provider.complete("Hello!")

Environment Variables:
    AZURE_OPENAI_API_KEY - Your Azure OpenAI API key
    AZURE_OPENAI_ENDPOINT - Your Azure OpenAI endpoint URL
    AZURE_OPENAI_API_VERSION - API version (default: 2024-02-01)
"""

import os
from typing import Any, AsyncIterator, Optional

from llmteam.providers.base import (
    BaseLLMProvider,
    CompletionConfig,
    LLMProviderError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMModelNotFoundError,
)


class AzureOpenAIProvider(BaseLLMProvider):
    """
    Azure OpenAI LLM Provider.

    Implements the LLMProvider protocol for Azure OpenAI Service.
    """

    DEFAULT_API_VERSION = "2024-02-01"

    def __init__(
        self,
        deployment_name: str,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        default_config: Optional[CompletionConfig] = None,
    ):
        """
        Initialize Azure OpenAI provider.

        Args:
            deployment_name: The deployment name in Azure.
            azure_endpoint: Azure OpenAI endpoint. If None, uses AZURE_OPENAI_ENDPOINT.
            api_key: API key. If None, uses AZURE_OPENAI_API_KEY env var.
            api_version: API version. If None, uses AZURE_OPENAI_API_VERSION or default.
            default_config: Default completion configuration.
        """
        super().__init__(deployment_name, api_key, default_config)
        self._deployment_name = deployment_name
        self._azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self._api_version = (
            api_version
            or os.environ.get("AZURE_OPENAI_API_VERSION")
            or self.DEFAULT_API_VERSION
        )

    @property
    def provider_name(self) -> str:
        return "AzureOpenAIProvider"

    def _get_client(self) -> Any:
        """Get or create the Azure OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncAzureOpenAI
            except ImportError:
                raise LLMProviderError(
                    "openai package not installed. Install with: pip install llmteam-ai[providers]",
                    provider=self.provider_name,
                )

            api_key = self._api_key or os.environ.get("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise LLMAuthenticationError(
                    "AZURE_OPENAI_API_KEY not set. Provide api_key or set environment variable.",
                    provider=self.provider_name,
                )

            if not self._azure_endpoint:
                raise LLMProviderError(
                    "AZURE_OPENAI_ENDPOINT not set. Provide azure_endpoint or set environment variable.",
                    provider=self.provider_name,
                )

            self._client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=self._azure_endpoint,
                api_version=self._api_version,
            )
        return self._client

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate completion for prompt.

        Args:
            prompt: The prompt to complete.
            **kwargs: Additional arguments (max_tokens, temperature, etc.)

        Returns:
            The completion text.
        """
        client = self._get_client()
        config = self._merge_config(kwargs)

        messages = kwargs.get("messages")
        if messages is None:
            system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

        try:
            response = await client.chat.completions.create(
                model=self._deployment_name,
                messages=messages,
                max_tokens=config["max_tokens"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                stop=config["stop"],
            )
            return response.choices[0].message.content or ""

        except Exception as e:
            self._handle_error(e)

    async def complete_with_messages(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """
        Generate completion for a list of messages.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            **kwargs: Additional arguments.

        Returns:
            The completion text.
        """
        return await self.complete("", messages=messages, **kwargs)

    async def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream completion tokens.

        Args:
            prompt: The prompt to complete.
            **kwargs: Additional arguments.

        Yields:
            Completion tokens/chunks.
        """
        client = self._get_client()
        config = self._merge_config(kwargs)

        messages = kwargs.get("messages")
        if messages is None:
            system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

        try:
            stream = await client.chat.completions.create(
                model=self._deployment_name,
                messages=messages,
                max_tokens=config["max_tokens"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                stop=config["stop"],
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            self._handle_error(e)

    def _handle_error(self, error: Exception) -> None:
        """Convert Azure OpenAI exceptions to LLMProviderError."""
        error_msg = str(error)
        error_type = type(error).__name__

        if "rate_limit" in error_msg.lower() or "RateLimitError" in error_type:
            raise LLMRateLimitError(
                message=error_msg,
                provider=self.provider_name,
            ) from error

        if "authentication" in error_msg.lower() or "AuthenticationError" in error_type:
            raise LLMAuthenticationError(
                message=error_msg,
                provider=self.provider_name,
            ) from error

        if "deployment" in error_msg.lower() and "not found" in error_msg.lower():
            raise LLMModelNotFoundError(
                model=self._deployment_name,
                provider=self.provider_name,
            ) from error

        raise LLMProviderError(
            message=error_msg,
            provider=self.provider_name,
        ) from error

    async def close(self) -> None:
        """Close the client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
