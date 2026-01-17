"""
Base LLM Provider implementation.

Provides common functionality for all LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Optional
from dataclasses import dataclass, field


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""

    def __init__(self, message: str, provider: str = "", details: Optional[dict] = None):
        super().__init__(message)
        self.provider = provider
        self.details = details or {}


class LLMRateLimitError(LLMProviderError):
    """Rate limit exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        provider: str = "",
        retry_after: Optional[float] = None,
    ):
        super().__init__(message, provider, {"retry_after": retry_after})
        self.retry_after = retry_after


class LLMAuthenticationError(LLMProviderError):
    """Authentication failed."""

    def __init__(self, message: str = "Authentication failed", provider: str = ""):
        super().__init__(message, provider)


class LLMModelNotFoundError(LLMProviderError):
    """Model not found or not accessible."""

    def __init__(self, model: str, provider: str = ""):
        super().__init__(f"Model not found: {model}", provider, {"model": model})
        self.model = model


@dataclass
class CompletionConfig:
    """Configuration for completion requests."""

    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    stop: Optional[list[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    # Additional provider-specific options
    extra: dict[str, Any] = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Implements the LLMProvider protocol from llmteam.runtime.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        default_config: Optional[CompletionConfig] = None,
    ):
        """
        Initialize the provider.

        Args:
            model: Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
            api_key: API key. If None, will try to load from environment.
            default_config: Default completion configuration.
        """
        self.model = model
        self._api_key = api_key
        self._default_config = default_config or CompletionConfig()
        self._client: Any = None

    @property
    def provider_name(self) -> str:
        """Return the provider name for error messages."""
        return self.__class__.__name__

    @abstractmethod
    async def complete(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate completion for prompt.

        This method implements the LLMProvider protocol.

        Args:
            prompt: The prompt to complete.
            **kwargs: Additional arguments (max_tokens, temperature, etc.)

        Returns:
            The completion text.

        Raises:
            LLMProviderError: On provider errors.
            LLMRateLimitError: When rate limited.
            LLMAuthenticationError: When authentication fails.
        """
        ...

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
        # Default implementation: convert to single prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(f"User: {content}")
        prompt = "\n\n".join(prompt_parts)
        return await self.complete(prompt, **kwargs)

    async def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream completion tokens.

        Default implementation yields the full response at once.
        Override for true streaming support.

        Args:
            prompt: The prompt to complete.
            **kwargs: Additional arguments.

        Yields:
            Completion tokens/chunks.
        """
        response = await self.complete(prompt, **kwargs)
        yield response

    def _merge_config(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Merge kwargs with default config."""
        config = {
            "max_tokens": kwargs.get("max_tokens", self._default_config.max_tokens),
            "temperature": kwargs.get("temperature", self._default_config.temperature),
            "top_p": kwargs.get("top_p", self._default_config.top_p),
            "stop": kwargs.get("stop", self._default_config.stop),
        }
        # Add any extra config
        for key, value in self._default_config.extra.items():
            if key not in config:
                config[key] = kwargs.get(key, value)
        return config

    async def __aenter__(self) -> "BaseLLMProvider":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close any resources (override if needed)."""
        pass
