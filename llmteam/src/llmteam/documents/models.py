"""
Document models for RAG pipeline.

Provides Document and Chunk dataclasses for document processing.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Document:
    """
    A document loaded from a file or text source.

    Attributes:
        content: The text content of the document.
        metadata: Metadata about the document (source, page, etc.).
    """

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> Optional[str]:
        """Get the source path/name of the document."""
        return self.metadata.get("source")

    @property
    def page(self) -> Optional[int]:
        """Get the page number if applicable."""
        return self.metadata.get("page")

    def __len__(self) -> int:
        """Return the length of the content."""
        return len(self.content)


@dataclass
class Chunk:
    """
    A chunk of text from a document.

    Attributes:
        text: The text content of the chunk.
        metadata: Metadata about the chunk (source, page, chunk index, etc.).
        index: The index of this chunk within its source document.
    """

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    index: int = 0

    @property
    def source(self) -> Optional[str]:
        """Get the source path/name of the chunk."""
        return self.metadata.get("source")

    @property
    def page(self) -> Optional[int]:
        """Get the page number if applicable."""
        return self.metadata.get("page")

    @property
    def start_char(self) -> Optional[int]:
        """Get the starting character position in the original document."""
        return self.metadata.get("start_char")

    @property
    def end_char(self) -> Optional[int]:
        """Get the ending character position in the original document."""
        return self.metadata.get("end_char")

    def __len__(self) -> int:
        """Return the length of the text."""
        return len(self.text)


__all__ = ["Document", "Chunk"]
