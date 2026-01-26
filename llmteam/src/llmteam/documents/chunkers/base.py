"""
Base chunker protocol for text chunking.

Defines the interface for all text chunkers.
"""

from abc import ABC, abstractmethod
from typing import List, Union

from llmteam.documents.models import Document, Chunk


class BaseChunker(ABC):
    """
    Abstract base class for text chunkers.

    All chunkers must implement the chunk method to split
    documents or text into smaller chunks.
    """

    @abstractmethod
    def chunk(self, source: Union[str, Document, List[Document]]) -> List[Chunk]:
        """
        Split source into chunks.

        Args:
            source: Text string, Document, or list of Documents to chunk.

        Returns:
            List of Chunk objects.
        """
        pass

    def chunk_text(self, text: str, metadata: dict = None) -> List[Chunk]:
        """
        Chunk a raw text string.

        Args:
            text: Text to chunk.
            metadata: Optional metadata to attach to chunks.

        Returns:
            List of Chunk objects.
        """
        doc = Document(content=text, metadata=metadata or {})
        return self.chunk(doc)


__all__ = ["BaseChunker"]
