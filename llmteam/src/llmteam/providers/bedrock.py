"""
AWS Bedrock LLM Provider.

Supports Amazon Bedrock-hosted models including Claude, Llama, and Titan.

Usage:
    from llmteam.providers import BedrockProvider

    provider = BedrockProvider(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0")
    response = await provider.complete("Hello!")

Environment Variables:
    AWS_ACCESS_KEY_ID - AWS access key
    AWS_SECRET_ACCESS_KEY - AWS secret key
    AWS_REGION - AWS region (default: us-east-1)
"""

import os
import json
from typing import Any, AsyncIterator, Optional

from llmteam.providers.base import (
    BaseLLMProvider,
    CompletionConfig,
    LLMProviderError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMModelNotFoundError,
)


class BedrockProvider(BaseLLMProvider):
    """
    AWS Bedrock LLM Provider.

    Implements the LLMProvider protocol for AWS Bedrock.
    """

    SUPPORTED_MODELS = [
        # Anthropic Claude
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "anthropic.claude-3-5-haiku-20241022-v1:0",
        "anthropic.claude-3-opus-20240229-v1:0",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
        # Meta Llama
        "meta.llama3-2-90b-instruct-v1:0",
        "meta.llama3-2-11b-instruct-v1:0",
        "meta.llama3-1-70b-instruct-v1:0",
        # Amazon Titan
        "amazon.titan-text-premier-v1:0",
        "amazon.titan-text-express-v1",
    ]

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        default_config: Optional[CompletionConfig] = None,
    ):
        """
        Initialize Bedrock provider.

        Args:
            model_id: Bedrock model ID.
            region_name: AWS region. If None, uses AWS_REGION or default.
            aws_access_key_id: AWS access key. If None, uses environment/credentials.
            aws_secret_access_key: AWS secret key. If None, uses environment/credentials.
            default_config: Default completion configuration.
        """
        super().__init__(model_id, None, default_config)
        self._model_id = model_id
        self._region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key

    @property
    def provider_name(self) -> str:
        return "BedrockProvider"

    def _get_client(self) -> Any:
        """Get or create the Bedrock runtime client."""
        if self._client is None:
            try:
                import boto3
            except ImportError:
                raise LLMProviderError(
                    "boto3 package not installed. Install with: pip install llmteam-ai[aws]",
                    provider=self.provider_name,
                )

            session_kwargs: dict[str, Any] = {"region_name": self._region_name}
            if self._aws_access_key_id and self._aws_secret_access_key:
                session_kwargs["aws_access_key_id"] = self._aws_access_key_id
                session_kwargs["aws_secret_access_key"] = self._aws_secret_access_key

            session = boto3.Session(**session_kwargs)
            self._client = session.client("bedrock-runtime")
        return self._client

    def _is_claude_model(self) -> bool:
        """Check if current model is a Claude model."""
        return self._model_id.startswith("anthropic.claude")

    def _is_llama_model(self) -> bool:
        """Check if current model is a Llama model."""
        return self._model_id.startswith("meta.llama")

    def _is_titan_model(self) -> bool:
        """Check if current model is a Titan model."""
        return self._model_id.startswith("amazon.titan")

    def _build_request_body(
        self,
        prompt: str,
        config: dict[str, Any],
        messages: Optional[list[dict[str, str]]] = None,
        system_prompt: str = "You are a helpful assistant.",
    ) -> dict[str, Any]:
        """Build the request body based on model type."""
        if self._is_claude_model():
            # Claude models use messages format
            if messages is None:
                messages = [{"role": "user", "content": prompt}]
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": config["max_tokens"],
                "temperature": config["temperature"],
                "top_p": config["top_p"],
                "system": system_prompt,
                "messages": messages,
            }
        elif self._is_llama_model():
            # Llama models
            if messages:
                prompt = self._format_llama_prompt(messages, system_prompt)
            else:
                prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            return {
                "prompt": prompt,
                "max_gen_len": config["max_tokens"],
                "temperature": config["temperature"],
                "top_p": config["top_p"],
            }
        elif self._is_titan_model():
            # Titan models
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            return {
                "inputText": full_prompt,
                "textGenerationConfig": {
                    "maxTokenCount": config["max_tokens"],
                    "temperature": config["temperature"],
                    "topP": config["top_p"],
                },
            }
        else:
            raise LLMProviderError(
                f"Unsupported model: {self._model_id}",
                provider=self.provider_name,
            )

    def _format_llama_prompt(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
    ) -> str:
        """Format messages for Llama models."""
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt

    def _parse_response(self, response_body: dict[str, Any]) -> str:
        """Parse the response based on model type."""
        if self._is_claude_model():
            content = response_body.get("content", [])
            if content and len(content) > 0:
                return content[0].get("text", "")
            return ""
        elif self._is_llama_model():
            return response_body.get("generation", "")
        elif self._is_titan_model():
            results = response_body.get("results", [])
            if results and len(results) > 0:
                return results[0].get("outputText", "")
            return ""
        return ""

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate completion for prompt.

        Note: boto3 is synchronous, so this runs in a thread pool.

        Args:
            prompt: The prompt to complete.
            **kwargs: Additional arguments (max_tokens, temperature, etc.)

        Returns:
            The completion text.
        """
        import asyncio

        client = self._get_client()
        config = self._merge_config(kwargs)

        messages = kwargs.get("messages")
        system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")
        body = self._build_request_body(prompt, config, messages, system_prompt)

        try:
            # Run synchronous boto3 call in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.invoke_model(
                    modelId=self._model_id,
                    body=json.dumps(body),
                    contentType="application/json",
                    accept="application/json",
                ),
            )

            response_body = json.loads(response["body"].read())
            return self._parse_response(response_body)

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

        Note: Uses invoke_model_with_response_stream for Claude models.

        Args:
            prompt: The prompt to complete.
            **kwargs: Additional arguments.

        Yields:
            Completion tokens/chunks.
        """
        import asyncio

        client = self._get_client()
        config = self._merge_config(kwargs)

        messages = kwargs.get("messages")
        system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")
        body = self._build_request_body(prompt, config, messages, system_prompt)

        try:
            # Run synchronous boto3 streaming call
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.invoke_model_with_response_stream(
                    modelId=self._model_id,
                    body=json.dumps(body),
                    contentType="application/json",
                    accept="application/json",
                ),
            )

            # Process stream
            event_stream = response.get("body")
            if event_stream:
                for event in event_stream:
                    chunk = event.get("chunk")
                    if chunk:
                        chunk_data = json.loads(chunk.get("bytes", b"{}").decode())
                        if self._is_claude_model():
                            delta = chunk_data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                yield delta.get("text", "")
                        elif self._is_llama_model():
                            yield chunk_data.get("generation", "")
                        elif self._is_titan_model():
                            yield chunk_data.get("outputText", "")

        except Exception as e:
            self._handle_error(e)

    def _handle_error(self, error: Exception) -> None:
        """Convert Bedrock exceptions to LLMProviderError."""
        error_msg = str(error)
        error_type = type(error).__name__

        if "ThrottlingException" in error_type or "throttl" in error_msg.lower():
            raise LLMRateLimitError(
                message=error_msg,
                provider=self.provider_name,
            ) from error

        if "AccessDeniedException" in error_type or "credential" in error_msg.lower():
            raise LLMAuthenticationError(
                message=error_msg,
                provider=self.provider_name,
            ) from error

        if "ResourceNotFoundException" in error_type or "model" in error_msg.lower():
            raise LLMModelNotFoundError(
                model=self._model_id,
                provider=self.provider_name,
            ) from error

        raise LLMProviderError(
            message=error_msg,
            provider=self.provider_name,
        ) from error
