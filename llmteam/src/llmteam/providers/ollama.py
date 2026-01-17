"""
Ollama Provider.

Provides access to locally-hosted LLMs via Ollama.

Ollama is a local LLM runner that supports many open models:
- Llama 2/3
- Mistral
- Phi
- CodeLlama
- And many more

No additional dependencies required (uses HTTP API).

Environment Variables:
    OLLAMA_HOST - Ollama server URL (default: http://localhost:11434)
"""

from typing import Any, AsyncIterator, Optional
import os
import json

from llmteam.providers.base import (
    BaseLLMProvider,
    LLMProviderError,
    LLMModelNotFoundError,
    CompletionConfig,
)


class OllamaProvider(BaseLLMProvider):
    """
    Ollama LLM provider for local models.

    Supports any model available in Ollama (llama2, mistral, codellama, etc.)

    Usage:
        # Using default localhost
        provider = OllamaProvider(model="llama2")
        response = await provider.complete("Hello!")

        # Custom Ollama server
        provider = OllamaProvider(
            model="mistral",
            host="http://my-ollama-server:11434",
        )

    Environment Variables:
        OLLAMA_HOST - Ollama server URL (default: http://localhost:11434)
    """

    def __init__(
        self,
        model: str = "llama2",
        host: Optional[str] = None,
        default_config: Optional[CompletionConfig] = None,
    ):
        """
        Initialize Ollama provider.

        Args:
            model: Model name (e.g., "llama2", "mistral", "codellama")
            host: Ollama server URL
            default_config: Default completion configuration
        """
        super().__init__(model=model, default_config=default_config)

        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.host = self.host.rstrip("/")

    @property
    def provider_name(self) -> str:
        return "Ollama"

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        """Generate completion using Ollama."""
        try:
            import aiohttp

            config = self._merge_config(kwargs)

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": config.get("max_tokens", 4096),
                    "temperature": config.get("temperature", 0.7),
                    "top_p": config.get("top_p", 1.0),
                },
            }

            if config.get("stop"):
                payload["options"]["stop"] = config["stop"]

            url = f"{self.host}/api/generate"

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 404:
                        raise LLMModelNotFoundError(self.model, provider=self.provider_name)
                    elif response.status != 200:
                        error_text = await response.text()
                        raise LLMProviderError(
                            f"Ollama API error: {response.status} - {error_text}",
                            provider=self.provider_name,
                        )

                    data = await response.json()
                    return data.get("response", "")

        except ImportError:
            raise LLMProviderError(
                "aiohttp is required for Ollama. Install with: pip install aiohttp",
                provider=self.provider_name,
            )
        except aiohttp.ClientConnectorError:
            raise LLMProviderError(
                f"Cannot connect to Ollama at {self.host}. "
                "Is Ollama running? Start with: ollama serve",
                provider=self.provider_name,
            )
        except LLMProviderError:
            raise
        except Exception as e:
            raise LLMProviderError(str(e), provider=self.provider_name)

    async def complete_with_messages(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Generate completion from chat messages."""
        try:
            import aiohttp

            config = self._merge_config(kwargs)

            # Convert messages to Ollama format
            ollama_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                # Ollama uses 'assistant' for model responses
                if role == "assistant":
                    ollama_messages.append({"role": "assistant", "content": content})
                elif role == "system":
                    ollama_messages.append({"role": "system", "content": content})
                else:
                    ollama_messages.append({"role": "user", "content": content})

            payload = {
                "model": self.model,
                "messages": ollama_messages,
                "stream": False,
                "options": {
                    "num_predict": config.get("max_tokens", 4096),
                    "temperature": config.get("temperature", 0.7),
                    "top_p": config.get("top_p", 1.0),
                },
            }

            url = f"{self.host}/api/chat"

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 404:
                        raise LLMModelNotFoundError(self.model, provider=self.provider_name)
                    elif response.status != 200:
                        error_text = await response.text()
                        raise LLMProviderError(
                            f"Ollama API error: {response.status} - {error_text}",
                            provider=self.provider_name,
                        )

                    data = await response.json()
                    return data.get("message", {}).get("content", "")

        except ImportError:
            return await super().complete_with_messages(messages, **kwargs)
        except LLMProviderError:
            raise
        except Exception as e:
            raise LLMProviderError(str(e), provider=self.provider_name)

    async def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream completion tokens."""
        try:
            import aiohttp

            config = self._merge_config(kwargs)

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_predict": config.get("max_tokens", 4096),
                    "temperature": config.get("temperature", 0.7),
                    "top_p": config.get("top_p", 1.0),
                },
            }

            url = f"{self.host}/api/generate"

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 404:
                        raise LLMModelNotFoundError(self.model, provider=self.provider_name)
                    elif response.status != 200:
                        error_text = await response.text()
                        raise LLMProviderError(
                            f"Ollama API error: {response.status} - {error_text}",
                            provider=self.provider_name,
                        )

                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode())
                                if "response" in data:
                                    yield data["response"]
                            except json.JSONDecodeError:
                                continue

        except ImportError:
            response = await self.complete(prompt, **kwargs)
            yield response
        except LLMProviderError:
            raise
        except Exception as e:
            raise LLMProviderError(str(e), provider=self.provider_name)

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models on the Ollama server."""
        try:
            import aiohttp

            url = f"{self.host}/api/tags"

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise LLMProviderError(
                            f"Failed to list models: {response.status}",
                            provider=self.provider_name,
                        )

                    data = await response.json()
                    return data.get("models", [])

        except ImportError:
            raise LLMProviderError(
                "aiohttp is required for Ollama",
                provider=self.provider_name,
            )
        except Exception as e:
            raise LLMProviderError(str(e), provider=self.provider_name)

    async def pull_model(self, model: Optional[str] = None) -> bool:
        """
        Pull (download) a model from the Ollama registry.

        Args:
            model: Model name to pull (default: self.model)

        Returns:
            True if successful
        """
        try:
            import aiohttp

            model_name = model or self.model
            url = f"{self.host}/api/pull"
            payload = {"name": model_name, "stream": False}

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMProviderError(
                            f"Failed to pull model: {error_text}",
                            provider=self.provider_name,
                        )
                    return True

        except ImportError:
            raise LLMProviderError(
                "aiohttp is required for Ollama",
                provider=self.provider_name,
            )
        except LLMProviderError:
            raise
        except Exception as e:
            raise LLMProviderError(str(e), provider=self.provider_name)
