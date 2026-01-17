"""
Anthropic LLM Provider.

Supports Claude 3.5, Claude 3 Opus, Sonnet, and Haiku models.

Usage:
    from llmteam.providers import AnthropicProvider

    provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")
    response = await provider.complete("Hello!")

Environment Variables:
    ANTHROPIC_API_KEY - Your Anthropic API key
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


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic LLM Provider.

    Implements the LLMProvider protocol for Anthropic's Claude API.
    """

    SUPPORTED_MODELS = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_config: Optional[CompletionConfig] = None,
    ):
        """
        Initialize Anthropic provider.

        Args:
            model: Model name (default: claude-3-5-sonnet-20241022)
            api_key: API key. If None, uses ANTHROPIC_API_KEY env var.
            base_url: Custom base URL for API calls.
            default_config: Default completion configuration.
        """
        super().__init__(model, api_key, default_config)
        self._base_url = base_url

    def _get_client(self) -> Any:
        """Get or create the Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise LLMProviderError(
                    "anthropic package not installed. Install with: pip install llmteam-ai[providers]",
                    provider=self.provider_name,
                )

            api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise LLMAuthenticationError(
                    "ANTHROPIC_API_KEY not set. Provide api_key or set environment variable.",
                    provider=self.provider_name,
                )

            kwargs: dict[str, Any] = {"api_key": api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url

            self._client = AsyncAnthropic(**kwargs)
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
            messages = [{"role": "user", "content": prompt}]

        system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")

        try:
            response = await client.messages.create(
                model=self.model,
                max_tokens=config["max_tokens"],
                system=system_prompt,
                messages=messages,
                temperature=config["temperature"],
                top_p=config["top_p"],
                stop_sequences=config["stop"] or [],
            )

            # Extract text from response
            content = response.content
            if content and len(content) > 0:
                return content[0].text
            return ""

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
        # Extract system message if present
        system_prompt = kwargs.get("system_prompt")
        filtered_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                if system_prompt is None:
                    system_prompt = msg.get("content", "")
            else:
                filtered_messages.append(msg)

        return await self.complete(
            "",
            messages=filtered_messages,
            system_prompt=system_prompt or "You are a helpful assistant.",
            **kwargs,
        )

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
            messages = [{"role": "user", "content": prompt}]

        system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")

        try:
            async with client.messages.stream(
                model=self.model,
                max_tokens=config["max_tokens"],
                system=system_prompt,
                messages=messages,
                temperature=config["temperature"],
                top_p=config["top_p"],
                stop_sequences=config["stop"] or [],
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            self._handle_error(e)

    def _handle_error(self, error: Exception) -> None:
        """Convert Anthropic exceptions to LLMProviderError."""
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
