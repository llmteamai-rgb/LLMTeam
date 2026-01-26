"""
Documents package for RAG pipeline.

Provides document loading and chunking capabilities.
"""

from llmteam.documents.models import Document, Chunk
from llmteam.documents.loaders import (
    BaseLoader,
    TextLoader,
    AutoLoader,
    load_document,
    load_documents,
)
from llmteam.documents.chunkers import (
    BaseChunker,
    RecursiveChunker,
    SentenceChunker,
    TokenChunker,
    SemanticChunker,
)

__all__ = [
    # Models
    "Document",
    "Chunk",
    # Loaders
    "BaseLoader",
    "TextLoader",
    "AutoLoader",
    "load_document",
    "load_documents",
    # Chunkers
    "BaseChunker",
    "RecursiveChunker",
    "SentenceChunker",
    "TokenChunker",
    "SemanticChunker",
]
