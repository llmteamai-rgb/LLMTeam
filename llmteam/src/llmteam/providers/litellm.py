"""
LiteLLM Provider.

Provides a unified API to access 100+ LLM providers through LiteLLM.

LiteLLM supports:
- OpenAI, Anthropic, Azure, AWS Bedrock, Google Vertex AI
- Ollama, Hugging Face, Replicate, Together AI
- And many more

Requires:
    pip install litellm

Environment Variables:
    Depends on the underlying provider (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY)
"""

from typing import Any, AsyncIterator, Optional

from llmteam.providers.base import (
    BaseLLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMModelNotFoundError,
    CompletionConfig,
)


class LiteLLMProvider(BaseLLMProvider):
    """
    LiteLLM unified provider for 100+ LLM backends.

    LiteLLM provides a consistent interface to multiple LLM providers,
    making it easy to switch between models without changing code.

    Usage:
        # OpenAI via LiteLLM
        provider = LiteLLMProvider(model="gpt-4")

        # Anthropic via LiteLLM
        provider = LiteLLMProvider(model="claude-3-opus-20240229")

        # Azure via LiteLLM
        provider = LiteLLMProvider(
            model="azure/gpt-4",
            api_base="https://my-resource.openai.azure.com",
        )

        # AWS Bedrock via LiteLLM
        provider = LiteLLMProvider(model="bedrock/anthropic.claude-v2")

        # Local Ollama via LiteLLM
        provider = LiteLLMProvider(
            model="ollama/llama2",
            api_base="http://localhost:11434",
        )

    Environment Variables:
        Depends on the underlying provider. Common ones:
        - OPENAI_API_KEY
        - ANTHROPIC_API_KEY
        - AZURE_API_KEY
        - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        timeout: Optional[float] = None,
        default_config: Optional[CompletionConfig] = None,
        **litellm_kwargs: Any,
    ):
        """
        Initialize LiteLLM provider.

        Args:
            model: Model name with optional provider prefix
                   (e.g., "gpt-4", "claude-3-opus-20240229", "azure/gpt-4")
            api_key: API key (or use environment variable)
            api_base: Custom API endpoint
            api_version: API version (for Azure)
            timeout: Request timeout in seconds
            default_config: Default completion configuration
            **litellm_kwargs: Additional LiteLLM parameters
        """
        super().__init__(model=model, default_config=default_config)

        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version
        self.timeout = timeout
        self.litellm_kwargs = litellm_kwargs

    @property
    def provider_name(self) -> str:
        return "LiteLLM"

    def _build_kwargs(self, config: dict[str, Any]) -> dict[str, Any]:
        """Build kwargs for LiteLLM completion call."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": config.get("max_tokens", 4096),
            "temperature": config.get("temperature", 0.7),
            "top_p": config.get("top_p", 1.0),
        }

        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_version:
            kwargs["api_version"] = self.api_version
        if self.timeout:
            kwargs["timeout"] = self.timeout

        if config.get("stop"):
            kwargs["stop"] = config["stop"]

        # Add any extra LiteLLM kwargs
        kwargs.update(self.litellm_kwargs)

        return kwargs

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        """Generate completion using LiteLLM."""
        try:
            import litellm

            config = self._merge_config(kwargs)
            litellm_kwargs = self._build_kwargs(config)

            # Use messages format
            messages = [{"role": "user", "content": prompt}]

            response = await litellm.acompletion(
                messages=messages,
                **litellm_kwargs,
            )

            return response.choices[0].message.content or ""

        except ImportError:
            raise LLMProviderError(
                "litellm is required. Install with: pip install litellm",
                provider=self.provider_name,
            )
        except Exception as e:
            self._handle_error(e)
            raise  # Re-raise if not handled

    async def complete_with_messages(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Generate completion from chat messages."""
        try:
            import litellm

            config = self._merge_config(kwargs)
            litellm_kwargs = self._build_kwargs(config)

            # Convert messages to LiteLLM format
            litellm_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                litellm_messages.append({"role": role, "content": content})

            response = await litellm.acompletion(
                messages=litellm_messages,
                **litellm_kwargs,
            )

            return response.choices[0].message.content or ""

        except ImportError:
            return await super().complete_with_messages(messages, **kwargs)
        except Exception as e:
            self._handle_error(e)
            raise

    async def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream completion tokens."""
        try:
            import litellm

            config = self._merge_config(kwargs)
            litellm_kwargs = self._build_kwargs(config)

            messages = [{"role": "user", "content": prompt}]

            response = await litellm.acompletion(
                messages=messages,
                stream=True,
                **litellm_kwargs,
            )

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except ImportError:
            response = await self.complete(prompt, **kwargs)
            yield response
        except Exception as e:
            self._handle_error(e)
            raise

    async def stream_with_messages(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream completion from chat messages."""
        try:
            import litellm

            config = self._merge_config(kwargs)
            litellm_kwargs = self._build_kwargs(config)

            litellm_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                litellm_messages.append({"role": role, "content": content})

            response = await litellm.acompletion(
                messages=litellm_messages,
                stream=True,
                **litellm_kwargs,
            )

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except ImportError:
            response = await self.complete_with_messages(messages, **kwargs)
            yield response
        except Exception as e:
            self._handle_error(e)
            raise

    def _handle_error(self, error: Exception) -> None:
        """Handle LiteLLM errors and convert to standard exceptions."""
        error_str = str(error).lower()

        if "rate limit" in error_str or "rate_limit" in error_str:
            raise LLMRateLimitError(str(error), provider=self.provider_name)
        elif "authentication" in error_str or "api key" in error_str or "invalid_api_key" in error_str:
            raise LLMAuthenticationError(str(error), provider=self.provider_name)
        elif "not found" in error_str or "model_not_found" in error_str:
            raise LLMModelNotFoundError(self.model, provider=self.provider_name)
        else:
            raise LLMProviderError(str(error), provider=self.provider_name)

    @staticmethod
    def list_supported_models() -> list[str]:
        """
        List commonly used model identifiers.

        Note: LiteLLM supports many more models. Check litellm.model_list
        for a complete list or visit: https://docs.litellm.ai/docs/providers
        """
        return [
            # OpenAI
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-3.5-turbo",
            # Anthropic
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            # Azure OpenAI (prefix with azure/)
            "azure/gpt-4",
            "azure/gpt-35-turbo",
            # AWS Bedrock (prefix with bedrock/)
            "bedrock/anthropic.claude-v2",
            "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            "bedrock/amazon.titan-text-express-v1",
            # Google Vertex AI (prefix with vertex_ai/)
            "vertex_ai/gemini-1.5-pro",
            "vertex_ai/gemini-1.5-flash",
            # Ollama (prefix with ollama/)
            "ollama/llama2",
            "ollama/mistral",
            "ollama/codellama",
            # Together AI (prefix with together_ai/)
            "together_ai/meta-llama/Llama-3-70b-chat-hf",
            # Replicate
            "replicate/meta/llama-2-70b-chat",
            # Hugging Face
            "huggingface/meta-llama/Llama-2-7b-chat-hf",
        ]

    @staticmethod
    async def check_model_support(model: str) -> bool:
        """
        Check if a model is supported by LiteLLM.

        Args:
            model: Model identifier

        Returns:
            True if model is supported
        """
        try:
            import litellm

            # LiteLLM has a model_list attribute
            return model in litellm.model_list or "/" in model
        except ImportError:
            return False
