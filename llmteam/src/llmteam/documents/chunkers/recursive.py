"""
Recursive character text splitter.

LangChain-style recursive splitting that tries to keep semantically
related text together by splitting on different separators in order.
"""

from typing import List, Optional, Union

from llmteam.documents.models import Document, Chunk
from llmteam.documents.chunkers.base import BaseChunker


class RecursiveChunker(BaseChunker):
    """
    Recursive character text splitter.

    Splits text recursively using a hierarchy of separators,
    trying to keep semantically related text together.

    Default separators (in order):
        1. "\\n\\n" - paragraph breaks
        2. "\\n" - line breaks
        3. " " - spaces
        4. "" - individual characters (last resort)
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        strip_whitespace: bool = True,
    ):
        """
        Initialize the recursive chunker.

        Args:
            chunk_size: Target size for each chunk in characters.
            chunk_overlap: Number of characters to overlap between chunks.
            separators: List of separators to try, in order of preference.
            keep_separator: Whether to keep the separator in the chunk.
            strip_whitespace: Whether to strip whitespace from chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.keep_separator = keep_separator
        self.strip_whitespace = strip_whitespace

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

    def chunk(self, source: Union[str, Document, List[Document]]) -> List[Chunk]:
        """
        Split source into chunks.

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
        chunks: List[Chunk] = []
        splits = self._split_text(text, self.separators)

        # Now merge splits into chunk_size pieces with overlap
        merged = self._merge_splits(splits)

        char_pos = 0
        for i, chunk_text in enumerate(merged):
            if self.strip_whitespace:
                chunk_text = chunk_text.strip()

            if not chunk_text:
                continue

            # Find actual position in original text
            start_char = text.find(chunk_text, char_pos)
            if start_char == -1:
                start_char = char_pos
            end_char = start_char + len(chunk_text)

            metadata = {
                **base_metadata,
                "chunk_index": i,
                "start_char": start_char,
                "end_char": end_char,
            }

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata=metadata,
                    index=i,
                )
            )

            # Update position for next search (account for overlap)
            char_pos = max(start_char + 1, end_char - self.chunk_overlap)

        # Re-index chunks
        for i, chunk in enumerate(chunks):
            chunk.index = i
            chunk.metadata["chunk_index"] = i

        return chunks

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators."""
        if not text:
            return []

        # Get the appropriate separator
        separator = separators[-1]  # Default to last (usually "")
        new_separators: List[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        # Split using the chosen separator
        if separator:
            splits = text.split(separator)
        else:
            # Character-by-character split
            splits = list(text)

        # Add separator back if needed
        if self.keep_separator and separator:
            final_splits = []
            for i, s in enumerate(splits):
                if i < len(splits) - 1:
                    final_splits.append(s + separator)
                else:
                    final_splits.append(s)
            splits = final_splits

        # Recursively split any chunks that are still too large
        final_chunks: List[str] = []
        for split in splits:
            if not split:
                continue

            if len(split) <= self.chunk_size:
                final_chunks.append(split)
            elif new_separators:
                # Recursively split with remaining separators
                final_chunks.extend(self._split_text(split, new_separators))
            else:
                # No more separators, split by character count
                final_chunks.extend(self._split_by_size(split))

        return final_chunks

    def _split_by_size(self, text: str) -> List[str]:
        """Split text into fixed-size chunks."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i : i + self.chunk_size]
            if chunk:
                chunks.append(chunk)
        return chunks

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Merge small splits into chunks of target size."""
        merged: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for split in splits:
            split_len = len(split)

            # If adding this split would exceed chunk_size, save current and start new
            if current_length + split_len > self.chunk_size and current_chunk:
                merged.append("".join(current_chunk))

                # Keep overlap from end of current chunk
                overlap_text = "".join(current_chunk)
                if len(overlap_text) > self.chunk_overlap:
                    overlap_text = overlap_text[-self.chunk_overlap :]
                    current_chunk = [overlap_text]
                    current_length = len(overlap_text)
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(split)
            current_length += split_len

        # Don't forget the last chunk
        if current_chunk:
            merged.append("".join(current_chunk))

        return merged


__all__ = ["RecursiveChunker"]
