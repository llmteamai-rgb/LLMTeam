"""
Base embedding protocol.

Defines the interface for all embedding providers.
"""

from abc import ABC, abstractmethod
from typing import List, Union


class BaseEmbedding(ABC):
    """
    Abstract base class for embedding providers.

    All embedding providers must implement the embed method
    to convert text into vector representations.
    """

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of the embeddings."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the embedding model."""
        pass

    @abstractmethod
    async def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Embed text(s) into vector(s).

        Args:
            texts: Single text or list of texts to embed.

        Returns:
            List of embedding vectors (each is a list of floats).
        """
        pass

    async def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text: Query text to embed.

        Returns:
            Single embedding vector.
        """
        embeddings = await self.embed(text)
        return embeddings[0]

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents.

        Args:
            texts: List of document texts to embed.

        Returns:
            List of embedding vectors.
        """
        return await self.embed(texts)


__all__ = ["BaseEmbedding"]
