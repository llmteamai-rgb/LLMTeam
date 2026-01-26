"""
HuggingFace embedding provider.

Provides embeddings using sentence-transformers models.
"""

from typing import List, Optional, Union
import asyncio

from llmteam.embeddings.base import BaseEmbedding


# Model dimensions mapping for common models
MODEL_DIMENSIONS = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "all-MiniLM-L12-v2": 384,
    "paraphrase-MiniLM-L6-v2": 384,
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
    "intfloat/multilingual-e5-large": 1024,
    "intfloat/multilingual-e5-base": 768,
    "intfloat/multilingual-e5-small": 384,
    "intfloat/e5-large-v2": 1024,
    "intfloat/e5-base-v2": 768,
    "intfloat/e5-small-v2": 384,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
}


class HuggingFaceEmbedding(BaseEmbedding):
    """
    HuggingFace embedding provider using sentence-transformers.

    Runs locally without API calls - great for offline use.

    Requires:
        pip install sentence-transformers

    Supported models:
        - all-MiniLM-L6-v2 (384 dimensions, fast, good quality)
        - all-mpnet-base-v2 (768 dimensions, best quality)
        - intfloat/multilingual-e5-large (1024 dimensions, multilingual)
        - BAAI/bge-large-en-v1.5 (1024 dimensions, high quality)
        - Any sentence-transformers compatible model
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress: bool = False,
    ):
        """
        Initialize the HuggingFace embedding provider.

        Args:
            model: HuggingFace model name or path.
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto).
            batch_size: Batch size for encoding.
            normalize_embeddings: Whether to normalize embeddings to unit length.
            show_progress: Show progress bar during encoding.
        """
        self._model_name = model
        self._device = device
        self._batch_size = batch_size
        self._normalize_embeddings = normalize_embeddings
        self._show_progress = show_progress
        self._model = None
        self._dimensions: Optional[int] = None

    def _ensure_model(self) -> None:
        """Ensure model is loaded."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "HuggingFace embeddings require 'sentence-transformers'. "
                "Install with: pip install sentence-transformers"
            )

        self._model = SentenceTransformer(
            self._model_name,
            device=self._device,
        )

        # Get actual dimensions from model
        self._dimensions = self._model.get_sentence_embedding_dimension()

    @property
    def dimensions(self) -> int:
        """Return the dimensionality of the embeddings."""
        # Try to use known dimensions first to avoid loading model
        if self._dimensions is not None:
            return self._dimensions

        if self._model_name in MODEL_DIMENSIONS:
            return MODEL_DIMENSIONS[self._model_name]

        # Load model to get dimensions
        self._ensure_model()
        return self._dimensions

    @property
    def model_name(self) -> str:
        """Return the name of the embedding model."""
        return self._model_name

    async def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Embed text(s) into vector(s).

        Args:
            texts: Single text or list of texts to embed.

        Returns:
            List of embedding vectors.
        """
        self._ensure_model()

        # Normalize input to list
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        # Run encoding in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            self._encode_sync,
            texts,
        )

        return embeddings

    def _encode_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronously encode texts."""
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize_embeddings,
            show_progress_bar=self._show_progress,
            convert_to_numpy=True,
        )

        # Convert numpy arrays to lists
        return [emb.tolist() for emb in embeddings]

    async def embed_many(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts. Alias for embed().

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        return await self.embed(texts)


__all__ = ["HuggingFaceEmbedding", "MODEL_DIMENSIONS"]
