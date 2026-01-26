"""
Auto loader that detects file type and uses appropriate loader.

Automatically selects the correct loader based on file extension.
"""

from pathlib import Path
from typing import Dict, List, Optional, Type, Union

from llmteam.documents.models import Document
from llmteam.documents.loaders.base import BaseLoader
from llmteam.documents.loaders.text import TextLoader


class AutoLoader(BaseLoader):
    """
    Automatic loader that detects file type and uses the appropriate loader.

    Supports:
        - .txt, .md, .markdown, .rst, .text (TextLoader)
        - .pdf (PDFLoader, requires pypdf)
        - .docx (DocxLoader, requires python-docx)
    """

    # Mapping of extensions to loader classes (lazy loaded)
    _EXTENSION_MAP: Dict[str, str] = {
        ".txt": "text",
        ".md": "text",
        ".markdown": "text",
        ".rst": "text",
        ".text": "text",
        ".pdf": "pdf",
        ".docx": "docx",
    }

    def __init__(self):
        """Initialize the auto loader."""
        self._loaders: Dict[str, BaseLoader] = {}
        # Text loader is always available
        self._loaders["text"] = TextLoader()

    def _get_loader(self, loader_type: str) -> BaseLoader:
        """Get or create a loader instance."""
        if loader_type in self._loaders:
            return self._loaders[loader_type]

        if loader_type == "pdf":
            from llmteam.documents.loaders.pdf import PDFLoader

            self._loaders["pdf"] = PDFLoader()
        elif loader_type == "docx":
            from llmteam.documents.loaders.docx import DocxLoader

            self._loaders["docx"] = DocxLoader()
        else:
            raise ValueError(f"Unknown loader type: {loader_type}")

        return self._loaders[loader_type]

    def _get_loader_type(self, path: Union[str, Path]) -> Optional[str]:
        """Get the loader type for a file."""
        ext = self._get_extension(path)
        return self._EXTENSION_MAP.get(ext)

    def load(self, source: Union[str, Path]) -> List[Document]:
        """
        Load a document using the appropriate loader.

        Args:
            source: Path to the file to load.

        Returns:
            List of Documents from the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file type is not supported.
        """
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        loader_type = self._get_loader_type(path)
        if loader_type is None:
            raise ValueError(
                f"Unsupported file extension: {path.suffix}. "
                f"Supported: {list(self._EXTENSION_MAP.keys())}"
            )

        loader = self._get_loader(loader_type)
        return loader.load(path)

    def load_many(self, sources: List[Union[str, Path]]) -> List[Document]:
        """
        Load multiple documents.

        Args:
            sources: List of file paths to load.

        Returns:
            List of all Documents from all files.
        """
        documents: List[Document] = []
        for source in sources:
            documents.extend(self.load(source))
        return documents

    def supports(self, path: Union[str, Path]) -> bool:
        """Check if this loader supports the given file."""
        return self._get_loader_type(path) is not None

    @classmethod
    def supported_extensions(cls) -> List[str]:
        """Get list of supported file extensions."""
        return list(cls._EXTENSION_MAP.keys())


# Convenience function
def load_document(source: Union[str, Path]) -> List[Document]:
    """
    Load a document using automatic type detection.

    Args:
        source: Path to the file to load.

    Returns:
        List of Documents from the file.
    """
    loader = AutoLoader()
    return loader.load(source)


def load_documents(sources: List[Union[str, Path]]) -> List[Document]:
    """
    Load multiple documents using automatic type detection.

    Args:
        sources: List of file paths to load.

    Returns:
        List of all Documents from all files.
    """
    loader = AutoLoader()
    return loader.load_many(sources)


__all__ = ["AutoLoader", "load_document", "load_documents"]
