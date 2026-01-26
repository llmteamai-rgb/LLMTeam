"""
Sentence-based text chunker.

Splits text into chunks at sentence boundaries.
"""

from typing import List, Optional, Union

from llmteam.documents.models import Document, Chunk
from llmteam.documents.chunkers.base import BaseChunker


class SentenceChunker(BaseChunker):
    """
    Sentence-based text chunker.

    Splits text at sentence boundaries and combines sentences
    into chunks of the target size.
    """

    # Common abbreviations that shouldn't end sentences
    ABBREVIATIONS = {"Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "vs", "etc"}

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_sentence_length: int = 10,
        strip_whitespace: bool = True,
    ):
        """
        Initialize the sentence chunker.

        Args:
            chunk_size: Target size for each chunk in characters.
            chunk_overlap: Number of characters to overlap (in terms of sentences).
            min_sentence_length: Minimum length for a valid sentence.
            strip_whitespace: Whether to strip whitespace from chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_sentence_length = min_sentence_length
        self.strip_whitespace = strip_whitespace

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

    def chunk(self, source: Union[str, Document, List[Document]]) -> List[Chunk]:
        """
        Split source into chunks at sentence boundaries.

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

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = []
        current_sentence = []
        words = text.split()

        for i, word in enumerate(words):
            current_sentence.append(word)

            # Check if word ends with sentence-ending punctuation
            if word.rstrip('").\'') and word.rstrip('").\'')[-1] in '.!?':
                # Check if it's an abbreviation
                word_without_punct = word.rstrip('.!?"\').,:;')
                if word_without_punct not in self.ABBREVIATIONS:
                    sentence = ' '.join(current_sentence)
                    if len(sentence.strip()) >= self.min_sentence_length:
                        sentences.append(sentence)
                    current_sentence = []

        # Don't forget remaining words
        if current_sentence:
            remaining = ' '.join(current_sentence)
            if len(remaining.strip()) >= self.min_sentence_length:
                sentences.append(remaining)
            elif sentences:
                # Append to last sentence if too short
                sentences[-1] = sentences[-1] + ' ' + remaining

        # If no sentences found, treat whole text as one
        if not sentences and text.strip():
            sentences = [text]

        return sentences

    def _chunk_text(self, text: str, base_metadata: dict) -> List[Chunk]:
        """Chunk a single text with metadata."""
        sentences = self._split_into_sentences(text)

        if not sentences:
            return []

        chunks: List[Chunk] = []
        current_sentences: List[str] = []
        current_length = 0
        char_pos = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If adding this sentence would exceed chunk_size
            if current_length + sentence_len > self.chunk_size and current_sentences:
                # Create chunk from current sentences
                chunk_text = "".join(current_sentences)
                if self.strip_whitespace:
                    chunk_text = chunk_text.strip()

                if chunk_text:
                    start_char = text.find(chunk_text, char_pos)
                    if start_char == -1:
                        start_char = char_pos
                    end_char = start_char + len(chunk_text)

                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            metadata={
                                **base_metadata,
                                "chunk_index": len(chunks),
                                "start_char": start_char,
                                "end_char": end_char,
                                "sentence_count": len(current_sentences),
                            },
                            index=len(chunks),
                        )
                    )
                    char_pos = end_char

                # Calculate overlap (keep last N characters worth of sentences)
                overlap_sentences: List[str] = []
                overlap_length = 0
                for s in reversed(current_sentences):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break

                current_sentences = overlap_sentences
                current_length = overlap_length

            current_sentences.append(sentence)
            current_length += sentence_len

        # Don't forget the last chunk
        if current_sentences:
            chunk_text = "".join(current_sentences)
            if self.strip_whitespace:
                chunk_text = chunk_text.strip()

            if chunk_text:
                start_char = text.find(chunk_text, char_pos)
                if start_char == -1:
                    start_char = char_pos
                end_char = start_char + len(chunk_text)

                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata={
                            **base_metadata,
                            "chunk_index": len(chunks),
                            "start_char": start_char,
                            "end_char": end_char,
                            "sentence_count": len(current_sentences),
                        },
                        index=len(chunks),
                    )
                )

        return chunks


__all__ = ["SentenceChunker"]
