"""
HTML loader for web pages and HTML files.

Uses BeautifulSoup for parsing.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from llmteam.documents.models import Document
from llmteam.documents.loaders.base import BaseLoader


class HTMLLoader(BaseLoader):
    """
    Loader for HTML files and content.

    Extracts text content from HTML, removing scripts and styles.

    Requires:
        pip install beautifulsoup4 lxml
    """

    SUPPORTED_EXTENSIONS = {".html", ".htm", ".xhtml"}

    def __init__(
        self,
        extract_title: bool = True,
        extract_metadata: bool = True,
        remove_scripts: bool = True,
        remove_styles: bool = True,
        parser: str = "lxml",
    ):
        """
        Initialize the HTML loader.

        Args:
            extract_title: Extract page title to metadata.
            extract_metadata: Extract meta tags to metadata.
            remove_scripts: Remove script tags.
            remove_styles: Remove style tags.
            parser: BeautifulSoup parser to use.
        """
        self.extract_title = extract_title
        self.extract_metadata = extract_metadata
        self.remove_scripts = remove_scripts
        self.remove_styles = remove_styles
        self.parser = parser

    def load(self, source: Union[str, Path]) -> List[Document]:
        """
        Load an HTML file.

        Args:
            source: Path to the HTML file.

        Returns:
            List containing a single Document.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "HTML support requires 'beautifulsoup4'. "
                "Install with: pip install beautifulsoup4 lxml"
            )

        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()

        return self.load_html(html_content, source=str(path))

    def load_html(
        self,
        html_content: str,
        source: Optional[str] = None,
    ) -> List[Document]:
        """
        Load HTML content directly.

        Args:
            html_content: Raw HTML string.
            source: Optional source identifier.

        Returns:
            List containing a single Document.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "HTML support requires 'beautifulsoup4'. "
                "Install with: pip install beautifulsoup4 lxml"
            )

        soup = BeautifulSoup(html_content, self.parser)

        # Remove unwanted elements
        if self.remove_scripts:
            for script in soup.find_all("script"):
                script.decompose()

        if self.remove_styles:
            for style in soup.find_all("style"):
                style.decompose()

        # Extract metadata
        metadata: Dict[str, Any] = {
            "source_type": "html",
        }

        if source:
            metadata["source"] = source
            if "/" in source or "\\" in source:
                metadata["file_name"] = Path(source).name

        if self.extract_title:
            title_tag = soup.find("title")
            if title_tag:
                metadata["title"] = title_tag.get_text(strip=True)

        if self.extract_metadata:
            # Extract common meta tags
            for meta in soup.find_all("meta"):
                name = meta.get("name", "").lower()
                content = meta.get("content", "")
                if name and content:
                    if name in {"description", "keywords", "author", "date"}:
                        metadata[f"meta_{name}"] = content

        # Extract text content
        text = soup.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace
        lines = [line.strip() for line in text.split("\n")]
        lines = [line for line in lines if line]
        content = "\n".join(lines)

        return [Document(content=content, metadata=metadata)]

    def supports(self, path: Union[str, Path]) -> bool:
        """Check if this loader supports the given file."""
        return self._get_extension(path) in self.SUPPORTED_EXTENSIONS


__all__ = ["HTMLLoader"]
