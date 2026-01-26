"""
Text loader for plain text and markdown files.

Supports: .txt, .md, .markdown, .rst, .text
"""

from pathlib import Path
from typing import List, Union

from llmteam.documents.models import Document
from llmteam.documents.loaders.base import BaseLoader


class TextLoader(BaseLoader):
    """
    Loader for plain text and markdown files.

    Supports extensions: .txt, .md, .markdown, .rst, .text
    """

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".text"}

    def __init__(self, encoding: str = "utf-8"):
        """
        Initialize the text loader.

        Args:
            encoding: File encoding to use (default: utf-8).
        """
        self.encoding = encoding

    def load(self, source: Union[str, Path]) -> List[Document]:
        """
        Load a text file as a Document.

        Args:
            source: Path to the text file.

        Returns:
            List containing a single Document with the file content.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not supported.
        """
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not self.supports(path):
            raise ValueError(
                f"Unsupported file extension: {path.suffix}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        content = path.read_text(encoding=self.encoding)

        return [
            Document(
                content=content,
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "extension": path.suffix.lower(),
                    "size_bytes": path.stat().st_size,
                },
            )
        ]

    def supports(self, path: Union[str, Path]) -> bool:
        """Check if this loader supports the given file."""
        return self._get_extension(path) in self.SUPPORTED_EXTENSIONS


__all__ = ["TextLoader"]
