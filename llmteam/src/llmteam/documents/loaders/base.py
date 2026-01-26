"""
Base loader protocol for document loading.

Defines the interface for all document loaders.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

from llmteam.documents.models import Document


class BaseLoader(ABC):
    """
    Abstract base class for document loaders.

    All loaders must implement the load method to convert
    files or text into Document objects.
    """

    @abstractmethod
    def load(self, source: Union[str, Path]) -> List[Document]:
        """
        Load documents from a source.

        Args:
            source: File path or text content to load.

        Returns:
            List of Document objects.
        """
        pass

    @abstractmethod
    def supports(self, path: Union[str, Path]) -> bool:
        """
        Check if this loader supports the given file.

        Args:
            path: File path to check.

        Returns:
            True if this loader can handle the file.
        """
        pass

    def _get_extension(self, path: Union[str, Path]) -> str:
        """Get lowercase file extension."""
        return Path(path).suffix.lower()


__all__ = ["BaseLoader"]
