"""
Google Vertex AI Provider.

Provides access to Google's Vertex AI models including Gemini.

Requires:
    pip install google-cloud-aiplatform

Environment Variables:
    GOOGLE_APPLICATION_CREDENTIALS - Path to service account JSON
    GOOGLE_CLOUD_PROJECT - GCP project ID
    GOOGLE_CLOUD_REGION - GCP region (default: us-central1)
"""

from typing import Any, AsyncIterator, Optional
import os

from llmteam.providers.base import (
    BaseLLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMModelNotFoundError,
    CompletionConfig,
)


class VertexAIProvider(BaseLLMProvider):
    """
    Google Vertex AI LLM provider.

    Supports Gemini models (gemini-1.5-pro, gemini-1.5-flash, etc.)

    Usage:
        provider = VertexAIProvider(
            model="gemini-1.5-pro",
            project_id="my-project",
            region="us-central1",
        )
        response = await provider.complete("Hello!")

    Environment Variables:
        GOOGLE_APPLICATION_CREDENTIALS - Service account key file
        GOOGLE_CLOUD_PROJECT - Default project ID
        GOOGLE_CLOUD_REGION - Default region
    """

    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        credentials_path: Optional[str] = None,
        default_config: Optional[CompletionConfig] = None,
    ):
        """
        Initialize Vertex AI provider.

        Args:
            model: Model name (e.g., "gemini-1.5-pro", "gemini-1.5-flash")
            project_id: GCP project ID
            region: GCP region (default: us-central1)
            credentials_path: Path to service account credentials JSON
            default_config: Default completion configuration
        """
        super().__init__(model=model, default_config=default_config)

        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self.region = region or os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        self._model_client: Any = None

    @property
    def provider_name(self) -> str:
        return "VertexAI"

    def _get_client(self) -> Any:
        """Get or create the Vertex AI client."""
        if self._model_client is None:
            try:
                import vertexai
                from vertexai.generative_models import GenerativeModel

                vertexai.init(project=self.project_id, location=self.region)
                self._model_client = GenerativeModel(self.model)

            except ImportError:
                raise LLMProviderError(
                    "google-cloud-aiplatform is required for Vertex AI. "
                    "Install with: pip install google-cloud-aiplatform",
                    provider=self.provider_name,
                )
            except Exception as e:
                if "credentials" in str(e).lower() or "authentication" in str(e).lower():
                    raise LLMAuthenticationError(
                        f"Vertex AI authentication failed: {e}",
                        provider=self.provider_name,
                    )
                raise LLMProviderError(str(e), provider=self.provider_name)

        return self._model_client

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        """Generate completion using Vertex AI."""
        try:
            from vertexai.generative_models import GenerationConfig

            client = self._get_client()
            config = self._merge_config(kwargs)

            generation_config = GenerationConfig(
                max_output_tokens=config.get("max_tokens", 4096),
                temperature=config.get("temperature", 0.7),
                top_p=config.get("top_p", 1.0),
                stop_sequences=config.get("stop") or [],
            )

            response = await client.generate_content_async(
                prompt,
                generation_config=generation_config,
            )

            return response.text

        except ImportError:
            raise LLMProviderError(
                "google-cloud-aiplatform is required",
                provider=self.provider_name,
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "rate" in error_msg:
                raise LLMRateLimitError(str(e), provider=self.provider_name)
            elif "not found" in error_msg or "invalid model" in error_msg:
                raise LLMModelNotFoundError(self.model, provider=self.provider_name)
            elif "permission" in error_msg or "credentials" in error_msg:
                raise LLMAuthenticationError(str(e), provider=self.provider_name)
            raise LLMProviderError(str(e), provider=self.provider_name)

    async def complete_with_messages(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Generate completion from messages."""
        try:
            from vertexai.generative_models import GenerationConfig, Content, Part

            client = self._get_client()
            config = self._merge_config(kwargs)

            # Convert messages to Vertex AI format
            contents = []
            system_instruction = None

            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if role == "system":
                    system_instruction = content
                elif role == "assistant":
                    contents.append(Content(
                        role="model",
                        parts=[Part.from_text(content)],
                    ))
                else:
                    contents.append(Content(
                        role="user",
                        parts=[Part.from_text(content)],
                    ))

            generation_config = GenerationConfig(
                max_output_tokens=config.get("max_tokens", 4096),
                temperature=config.get("temperature", 0.7),
                top_p=config.get("top_p", 1.0),
            )

            # Create model with system instruction if provided
            if system_instruction:
                from vertexai.generative_models import GenerativeModel
                model = GenerativeModel(
                    self.model,
                    system_instruction=system_instruction,
                )
                response = await model.generate_content_async(
                    contents,
                    generation_config=generation_config,
                )
            else:
                response = await client.generate_content_async(
                    contents,
                    generation_config=generation_config,
                )

            return response.text

        except ImportError:
            # Fall back to default implementation
            return await super().complete_with_messages(messages, **kwargs)
        except Exception as e:
            raise LLMProviderError(str(e), provider=self.provider_name)

    async def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream completion tokens."""
        try:
            from vertexai.generative_models import GenerationConfig

            client = self._get_client()
            config = self._merge_config(kwargs)

            generation_config = GenerationConfig(
                max_output_tokens=config.get("max_tokens", 4096),
                temperature=config.get("temperature", 0.7),
                top_p=config.get("top_p", 1.0),
            )

            response = await client.generate_content_async(
                prompt,
                generation_config=generation_config,
                stream=True,
            )

            async for chunk in response:
                if chunk.text:
                    yield chunk.text

        except ImportError:
            # Fall back to non-streaming
            response = await self.complete(prompt, **kwargs)
            yield response
        except Exception as e:
            raise LLMProviderError(str(e), provider=self.provider_name)
