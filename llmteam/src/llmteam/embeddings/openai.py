"""
OpenAI embedding provider.

Provides embeddings using OpenAI's embedding models.
"""

import os
from typing import List, Optional, Union


from llmteam.embeddings.base import BaseEmbedding


# Model dimensions mapping
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedding(BaseEmbedding):
    """
    OpenAI embedding provider.

    Uses OpenAI's embedding API to generate text embeddings.

    Supported models:
        - text-embedding-3-small (1536 dimensions, recommended)
        - text-embedding-3-large (3072 dimensions)
        - text-embedding-ada-002 (1536 dimensions, legacy)
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        batch_size: int = 100,
    ):
        """
        Initialize the OpenAI embedding provider.

        Args:
            model: OpenAI embedding model to use.
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
            batch_size: Maximum number of texts to embed in one API call.
        """
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._batch_size = batch_size
        self._client = None

        if model not in MODEL_DIMENSIONS:
            raise ValueError(
                f"Unknown model: {model}. Supported: {list(MODEL_DIMENSIONS.keys())}"
            )

    def _ensure_client(self) -> None:
        """Ensure OpenAI client is initialized."""
        if self._client is not None:
            return

        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "OpenAI embeddings require 'openai'. Install it with:\n"
                "  pip install openai\n"
                "or:\n"
                "  pip install llmteam-ai[providers]"
            )

        if not self._api_key:
            raise ValueError(
                "OpenAI API key is required. Either pass api_key parameter or "
                "set OPENAI_API_KEY environment variable."
            )

        self._client = AsyncOpenAI(api_key=self._api_key)

    @property
    def dimensions(self) -> int:
        """Return the dimensionality of the embeddings."""
        return MODEL_DIMENSIONS[self._model]

    @property
    def model_name(self) -> str:
        """Return the name of the embedding model."""
        return self._model

    async def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Embed text(s) into vector(s).

        Args:
            texts: Single text or list of texts to embed.

        Returns:
            List of embedding vectors.
        """
        self._ensure_client()

        # Normalize input to list
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        # Process in batches
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]

            response = await self._client.embeddings.create(
                model=self._model,
                input=batch,
            )

            # Sort by index to ensure correct order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            batch_embeddings = [item.embedding for item in sorted_data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings


__all__ = ["OpenAIEmbedding", "MODEL_DIMENSIONS"]
