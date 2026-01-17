"""
OpenAI LLM Provider.

Supports GPT-4, GPT-4o, GPT-4o-mini, and other OpenAI models.

Usage:
    from llmteam.providers import OpenAIProvider

    provider = OpenAIProvider(model="gpt-4o")
    response = await provider.complete("Hello!")

Environment Variables:
    OPENAI_API_KEY - Your OpenAI API key
    OPENAI_ORG_ID - (Optional) Organization ID
    OPENAI_BASE_URL - (Optional) Custom base URL
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


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM Provider.

    Implements the LLMProvider protocol for OpenAI's API.
    """

    SUPPORTED_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "o1-preview",
        "o1-mini",
    ]

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        default_config: Optional[CompletionConfig] = None,
    ):
        """
        Initialize OpenAI provider.

        Args:
            model: Model name (default: gpt-4o)
            api_key: API key. If None, uses OPENAI_API_KEY env var.
            organization: Organization ID. If None, uses OPENAI_ORG_ID env var.
            base_url: Custom base URL. If None, uses OPENAI_BASE_URL env var.
            default_config: Default completion configuration.
        """
        super().__init__(model, api_key, default_config)
        self._organization = organization or os.environ.get("OPENAI_ORG_ID")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")

    def _get_client(self) -> Any:
        """Get or create the OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise LLMProviderError(
                    "openai package not installed. Install with: pip install llmteam-ai[providers]",
                    provider=self.provider_name,
                )

            api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise LLMAuthenticationError(
                    "OPENAI_API_KEY not set. Provide api_key or set environment variable.",
                    provider=self.provider_name,
                )

            self._client = AsyncOpenAI(
                api_key=api_key,
                organization=self._organization,
                base_url=self._base_url,
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
            # Convert single prompt to messages format
            system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

        try:
            response = await client.chat.completions.create(
                model=self.model,
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
                model=self.model,
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
        """Convert OpenAI exceptions to LLMProviderError."""
        error_msg = str(error)
        error_type = type(error).__name__

        # Check for specific error types
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

        if "model" in error_msg.lower() and "not found" in error_msg.lower():
            raise LLMModelNotFoundError(
                model=self.model,
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
