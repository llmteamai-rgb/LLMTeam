"""
Token-based text chunker.

Splits text based on token count using tiktoken.
"""

from typing import List, Optional, Union

from llmteam.documents.models import Document, Chunk
from llmteam.documents.chunkers.base import BaseChunker


class TokenChunker(BaseChunker):
    """
    Token-based text chunker.

    Splits text into chunks based on token count using tiktoken.
    This ensures chunks respect the token limits of LLM models.

    Requires:
        pip install tiktoken
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        model: str = "gpt-4",
        encoding_name: Optional[str] = None,
        strip_whitespace: bool = True,
    ):
        """
        Initialize the token chunker.

        Args:
            chunk_size: Target size for each chunk in tokens.
            chunk_overlap: Number of tokens to overlap between chunks.
            model: Model name for encoding selection (default: gpt-4).
            encoding_name: Override encoding name (e.g., "cl100k_base").
            strip_whitespace: Whether to strip whitespace from chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = model
        self.encoding_name = encoding_name
        self.strip_whitespace = strip_whitespace

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self._encoding = None

    def _get_encoding(self):
        """Get tiktoken encoding lazily."""
        if self._encoding is None:
            try:
                import tiktoken
            except ImportError:
                raise ImportError(
                    "Token chunking requires 'tiktoken'. "
                    "Install with: pip install tiktoken"
                )

            if self.encoding_name:
                self._encoding = tiktoken.get_encoding(self.encoding_name)
            else:
                self._encoding = tiktoken.encoding_for_model(self.model)

        return self._encoding

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        encoding = self._get_encoding()
        return len(encoding.encode(text))

    def _encode(self, text: str) -> List[int]:
        """Encode text to tokens."""
        encoding = self._get_encoding()
        return encoding.encode(text)

    def _decode(self, tokens: List[int]) -> str:
        """Decode tokens to text."""
        encoding = self._get_encoding()
        return encoding.decode(tokens)

    def chunk(self, source: Union[str, Document, List[Document]]) -> List[Chunk]:
        """
        Split source into chunks based on token count.

        Args:
            source: Text string, Document, or list of Documents to chunk.

        Returns:
            List of Chunk objects.
        """
        if isinstance(source, str):
            return self._chunk_text(source, {})
        elif isinstance(source, Document):
            return self._chunk_text(source.content, source.metadata)
        elif isinstance(source, list):
            all_chunks: List[Chunk] = []
            for doc in source:
                all_chunks.extend(self._chunk_text(doc.content, doc.metadata))
            return all_chunks
        else:
            raise TypeError(f"Expected str, Document, or List[Document], got {type(source)}")

    def _chunk_text(self, text: str, base_metadata: dict) -> List[Chunk]:
        """Chunk a single text with metadata."""
        tokens = self._encode(text)
        chunks: List[Chunk] = []

        if len(tokens) == 0:
            return []

        # If text fits in one chunk
        if len(tokens) <= self.chunk_size:
            chunk_text = text
            if self.strip_whitespace:
                chunk_text = chunk_text.strip()

            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata={
                            **base_metadata,
                            "chunk_index": 0,
                            "token_count": len(tokens),
                        },
                        index=0,
                    )
                )
            return chunks

        # Split into chunks with overlap
        step = self.chunk_size - self.chunk_overlap
        chunk_index = 0

        for start in range(0, len(tokens), step):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self._decode(chunk_tokens)

            if self.strip_whitespace:
                chunk_text = chunk_text.strip()

            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata={
                            **base_metadata,
                            "chunk_index": chunk_index,
                            "token_count": len(chunk_tokens),
                            "token_start": start,
                            "token_end": end,
                        },
                        index=chunk_index,
                    )
                )
                chunk_index += 1

            # Stop if we've processed all tokens
            if end >= len(tokens):
                break

        return chunks


__all__ = ["TokenChunker"]
