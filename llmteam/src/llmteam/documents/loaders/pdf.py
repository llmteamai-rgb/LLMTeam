"""
PDF loader for PDF files.

Requires: pypdf (pip install llmteam-ai[pdf])
"""

from pathlib import Path
from typing import List, Union

from llmteam.documents.models import Document
from llmteam.documents.loaders.base import BaseLoader


class PDFLoader(BaseLoader):
    """
    Loader for PDF files.

    Requires the pypdf package:
        pip install pypdf
        # or
        pip install llmteam-ai[pdf]
    """

    SUPPORTED_EXTENSIONS = {".pdf"}

    def __init__(self, extract_images: bool = False, page_separator: str = "\n\n"):
        """
        Initialize the PDF loader.

        Args:
            extract_images: Whether to extract text from images (not supported yet).
            page_separator: String to separate pages in the combined output.
        """
        self.extract_images = extract_images
        self.page_separator = page_separator
        self._ensure_pypdf()

    def _ensure_pypdf(self) -> None:
        """Ensure pypdf is installed."""
        try:
            import pypdf  # noqa: F401
        except ImportError:
            raise ImportError(
                "PDF support requires 'pypdf'. Install it with:\n"
                "  pip install pypdf\n"
                "or:\n"
                "  pip install llmteam-ai[pdf]"
            )

    def load(self, source: Union[str, Path]) -> List[Document]:
        """
        Load a PDF file as Documents (one per page).

        Args:
            source: Path to the PDF file.

        Returns:
            List of Documents, one per page.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not supported.
        """
        import pypdf

        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not self.supports(path):
            raise ValueError(f"Unsupported file extension: {path.suffix}. Expected: .pdf")

        documents: List[Document] = []

        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)
            total_pages = len(reader.pages)

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""

                if text.strip():  # Only add non-empty pages
                    documents.append(
                        Document(
                            content=text,
                            metadata={
                                "source": str(path),
                                "filename": path.name,
                                "extension": ".pdf",
                                "page": page_num + 1,  # 1-indexed
                                "total_pages": total_pages,
                            },
                        )
                    )

        # If no pages had text, return single empty document
        if not documents:
            documents.append(
                Document(
                    content="",
                    metadata={
                        "source": str(path),
                        "filename": path.name,
                        "extension": ".pdf",
                        "total_pages": total_pages,
                    },
                )
            )

        return documents

    def load_combined(self, source: Union[str, Path]) -> Document:
        """
        Load a PDF file as a single combined Document.

        Args:
            source: Path to the PDF file.

        Returns:
            Single Document with all pages combined.
        """
        documents = self.load(source)
        path = Path(source)

        combined_content = self.page_separator.join(doc.content for doc in documents)

        return Document(
            content=combined_content,
            metadata={
                "source": str(path),
                "filename": path.name,
                "extension": ".pdf",
                "total_pages": len(documents),
            },
        )

    def supports(self, path: Union[str, Path]) -> bool:
        """Check if this loader supports the given file."""
        return self._get_extension(path) in self.SUPPORTED_EXTENSIONS


__all__ = ["PDFLoader"]
