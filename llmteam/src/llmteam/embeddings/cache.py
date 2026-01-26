"""
Embedding cache using SQLite.

Provides caching for embeddings to avoid redundant API calls.
"""

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Union

from llmteam.embeddings.base import BaseEmbedding


class EmbeddingCache(BaseEmbedding):
    """
    Caching wrapper for embedding providers.

    Caches embeddings in a SQLite database to avoid redundant API calls.
    """

    def __init__(
        self,
        embedding: BaseEmbedding,
        cache_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the embedding cache.

        Args:
            embedding: The underlying embedding provider to cache.
            cache_path: Path to the SQLite cache file. Defaults to ~/.llmteam/embeddings.db
        """
        self._embedding = embedding

        if cache_path is None:
            cache_dir = Path.home() / ".llmteam"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / "embeddings.db"

        self._cache_path = Path(cache_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        with sqlite3.connect(self._cache_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    hash TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_hash ON embeddings(model, hash)
            """)
            conn.commit()

    def _hash_text(self, text: str) -> str:
        """Create a hash of the text for cache lookup."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_cached(self, texts: List[str]) -> Dict[str, List[float]]:
        """Get cached embeddings for texts."""
        cached: Dict[str, List[float]] = {}

        with sqlite3.connect(self._cache_path) as conn:
            for text in texts:
                text_hash = self._hash_text(text)
                cursor = conn.execute(
                    "SELECT embedding FROM embeddings WHERE hash = ? AND model = ?",
                    (text_hash, self._embedding.model_name),
                )
                row = cursor.fetchone()
                if row:
                    cached[text] = json.loads(row[0])

        return cached

    def _save_to_cache(self, text: str, embedding: List[float]) -> None:
        """Save an embedding to the cache."""
        text_hash = self._hash_text(text)

        with sqlite3.connect(self._cache_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings (hash, model, text, embedding)
                VALUES (?, ?, ?, ?)
                """,
                (text_hash, self._embedding.model_name, text, json.dumps(embedding)),
            )
            conn.commit()

    @property
    def dimensions(self) -> int:
        """Return the dimensionality of the embeddings."""
        return self._embedding.dimensions

    @property
    def model_name(self) -> str:
        """Return the name of the embedding model."""
        return self._embedding.model_name

    async def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Embed text(s) into vector(s), using cache where available.

        Args:
            texts: Single text or list of texts to embed.

        Returns:
            List of embedding vectors.
        """
        # Normalize input to list
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        # Check cache
        cached = self._get_cached(texts)

        # Find texts that need embedding
        texts_to_embed = [t for t in texts if t not in cached]

        # Embed uncached texts
        if texts_to_embed:
            new_embeddings = await self._embedding.embed(texts_to_embed)

            # Save to cache
            for text, embedding in zip(texts_to_embed, new_embeddings):
                cached[text] = embedding
                self._save_to_cache(text, embedding)

        # Return in original order
        return [cached[text] for text in texts]

    def clear_cache(self) -> int:
        """
        Clear all cached embeddings.

        Returns:
            Number of embeddings cleared.
        """
        with sqlite3.connect(self._cache_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
            count = cursor.fetchone()[0]
            conn.execute("DELETE FROM embeddings")
            conn.commit()
            return count

    def cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dict with 'total', 'by_model' counts.
        """
        with sqlite3.connect(self._cache_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
            total = cursor.fetchone()[0]

            cursor = conn.execute(
                "SELECT model, COUNT(*) FROM embeddings GROUP BY model"
            )
            by_model = dict(cursor.fetchall())

        return {"total": total, "by_model": by_model}


__all__ = ["EmbeddingCache"]
