"""
Semantic text chunker.

Splits text into semantically coherent chunks using embeddings.
"""

from typing import List, Optional, Union

import numpy as np

from llmteam.documents.models import Document, Chunk
from llmteam.documents.chunkers.base import BaseChunker


class SemanticChunker(BaseChunker):
    """
    Semantic text chunker.

    Splits text into semantically coherent chunks by:
    1. Splitting into sentences
    2. Computing embeddings for sentence groups
    3. Finding semantic boundaries where similarity drops

    This creates more meaningful chunks than simple character/token splitting.

    Requires:
        An embedding function or provider.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        similarity_threshold: float = 0.7,
        min_chunk_size: int = 100,
        embedding_fn=None,
        buffer_size: int = 3,
        strip_whitespace: bool = True,
    ):
        """
        Initialize the semantic chunker.

        Args:
            chunk_size: Target maximum size for each chunk in characters.
            chunk_overlap: Number of characters to overlap between chunks.
            similarity_threshold: Minimum cosine similarity to keep sentences together.
            min_chunk_size: Minimum chunk size in characters.
            embedding_fn: Async function that takes List[str] and returns List[List[float]].
                If None, will use OpenAI embeddings.
            buffer_size: Number of sentences to combine for embedding calculation.
            strip_whitespace: Whether to strip whitespace from chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.buffer_size = buffer_size
        self.strip_whitespace = strip_whitespace
        self._embedding_fn = embedding_fn

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts."""
        if self._embedding_fn is not None:
            return await self._embedding_fn(texts)

        # Default to OpenAI embeddings
        try:
            from llmteam.embeddings.openai import OpenAIEmbedding
        except ImportError:
            raise ImportError(
                "Semantic chunking requires embeddings. "
                "Either provide embedding_fn or install OpenAI: pip install openai"
            )

        embedding = OpenAIEmbedding()
        return await embedding.embed_many(texts)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        import re

        # Split on sentence-ending punctuation followed by whitespace or end
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def chunk(self, source: Union[str, Document, List[Document]]) -> List[Chunk]:
        """
        Synchronous chunk method - raises error for semantic chunker.

        Use chunk_async() instead.
        """
        raise NotImplementedError(
            "SemanticChunker requires async operation. "
            "Use 'await chunker.chunk_async(source)' instead."
        )

    async def chunk_async(
        self, source: Union[str, Document, List[Document]]
    ) -> List[Chunk]:
        """
        Split source into semantically coherent chunks.

        Args:
            source: Text string, Document, or list of Documents to chunk.

        Returns:
            List of Chunk objects.
        """
        if isinstance(source, str):
            return await self._chunk_text(source, {})
        elif isinstance(source, Document):
            return await self._chunk_text(source.content, source.metadata)
        elif isinstance(source, list):
            all_chunks: List[Chunk] = []
            for doc in source:
                chunks = await self._chunk_text(doc.content, doc.metadata)
                all_chunks.extend(chunks)
            return all_chunks
        else:
            raise TypeError(f"Expected str, Document, or List[Document], got {type(source)}")

    async def _chunk_text(self, text: str, base_metadata: dict) -> List[Chunk]:
        """Chunk a single text with metadata using semantic boundaries."""
        sentences = self._split_into_sentences(text)

        if len(sentences) == 0:
            return []

        # If text is short enough, return as single chunk
        if len(text) <= self.chunk_size:
            chunk_text = text
            if self.strip_whitespace:
                chunk_text = chunk_text.strip()

            if chunk_text:
                return [
                    Chunk(
                        text=chunk_text,
                        metadata={
                            **base_metadata,
                            "chunk_index": 0,
                            "sentence_count": len(sentences),
                        },
                        index=0,
                    )
                ]
            return []

        # Create sentence groups for embedding
        sentence_groups: List[str] = []
        for i in range(len(sentences)):
            start = max(0, i - self.buffer_size // 2)
            end = min(len(sentences), i + self.buffer_size // 2 + 1)
            group = " ".join(sentences[start:end])
            sentence_groups.append(group)

        # Get embeddings for sentence groups
        embeddings = await self._get_embeddings(sentence_groups)

        # Find semantic boundaries
        boundaries = self._find_semantic_boundaries(embeddings, sentences)

        # Create chunks based on boundaries
        chunks = self._create_chunks_from_boundaries(
            sentences, boundaries, text, base_metadata
        )

        return chunks

    def _find_semantic_boundaries(
        self, embeddings: List[List[float]], sentences: List[str]
    ) -> List[int]:
        """Find indices where semantic breaks occur."""
        if len(embeddings) < 2:
            return []

        boundaries: List[int] = []
        similarities: List[float] = []

        # Calculate similarities between consecutive sentences
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Find boundaries where similarity drops below threshold
        # Also consider cumulative length to respect chunk_size
        cumulative_length = len(sentences[0])

        for i, sim in enumerate(similarities):
            cumulative_length += len(sentences[i + 1])

            # Create boundary if similarity is low or chunk is getting too big
            if sim < self.similarity_threshold or cumulative_length > self.chunk_size:
                boundaries.append(i + 1)  # Boundary after sentence i
                cumulative_length = 0

        return boundaries

    def _create_chunks_from_boundaries(
        self,
        sentences: List[str],
        boundaries: List[int],
        original_text: str,
        base_metadata: dict,
    ) -> List[Chunk]:
        """Create chunks based on semantic boundaries."""
        chunks: List[Chunk] = []
        boundaries = [0] + boundaries + [len(sentences)]

        char_pos = 0
        chunk_index = 0

        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]

            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = " ".join(chunk_sentences)

            if self.strip_whitespace:
                chunk_text = chunk_text.strip()

            if not chunk_text or len(chunk_text) < self.min_chunk_size:
                continue

            # Find position in original text
            start_char = original_text.find(chunk_text[:50], char_pos)
            if start_char == -1:
                start_char = char_pos
            end_char = start_char + len(chunk_text)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        **base_metadata,
                        "chunk_index": chunk_index,
                        "start_char": start_char,
                        "end_char": end_char,
                        "sentence_count": len(chunk_sentences),
                    },
                    index=chunk_index,
                )
            )

            char_pos = end_char
            chunk_index += 1

        return chunks


__all__ = ["SemanticChunker"]
