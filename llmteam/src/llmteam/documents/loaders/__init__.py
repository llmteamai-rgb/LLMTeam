"""
Document loaders package.

Provides loaders for various document formats.
"""

from llmteam.documents.loaders.base import BaseLoader
from llmteam.documents.loaders.text import TextLoader
from llmteam.documents.loaders.auto import AutoLoader, load_document, load_documents

# Format-specific loaders are lazy-loaded to avoid import errors when dependencies are missing
# Use: from llmteam.documents.loaders.pdf import PDFLoader
# Use: from llmteam.documents.loaders.docx import DocxLoader
# Use: from llmteam.documents.loaders.xlsx import ExcelLoader
# Use: from llmteam.documents.loaders.html import HTMLLoader
# Use: from llmteam.documents.loaders.pptx import PPTXLoader

__all__ = [
    "BaseLoader",
    "TextLoader",
    "AutoLoader",
    "load_document",
    "load_documents",
]
