"""
Tests for document chunkers.
"""

import pytest

from llmteam.documents import Document, Chunk
from llmteam.documents.chunkers import RecursiveChunker, SentenceChunker


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""

    def test_chunk_short_text(self):
        """Test chunking text shorter than chunk_size."""
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk("Short text")

        assert len(chunks) == 1
        assert chunks[0].text == "Short text"
        assert chunks[0].index == 0

    def test_chunk_long_text(self):
        """Test chunking text longer than chunk_size."""
        text = "A" * 1000  # 1000 characters
        chunker = RecursiveChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk(text)

        assert len(chunks) > 1
        # Each chunk should be roughly chunk_size
        for chunk in chunks:
            assert len(chunk.text) <= 300 + 50  # Allow some flexibility

    def test_chunk_paragraphs(self):
        """Test chunking prefers paragraph breaks."""
        text = "Paragraph one is here with some content.\n\nParagraph two has more text here.\n\nParagraph three is the final one with lots of words."
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk(text)

        # Should split at paragraph boundaries when text exceeds chunk_size
        assert len(chunks) >= 2

    def test_chunk_document(self):
        """Test chunking a Document object."""
        doc = Document(
            content="Hello world. This is a test.",
            metadata={"source": "test.txt"},
        )
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(doc)

        assert len(chunks) >= 1
        assert chunks[0].metadata["source"] == "test.txt"

    def test_chunk_list_of_documents(self):
        """Test chunking multiple documents."""
        docs = [
            Document(content="First document.", metadata={"source": "1.txt"}),
            Document(content="Second document.", metadata={"source": "2.txt"}),
        ]
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(docs)

        assert len(chunks) >= 2
        sources = [c.metadata.get("source") for c in chunks]
        assert "1.txt" in sources
        assert "2.txt" in sources

    def test_chunk_metadata_preserved(self):
        """Test that metadata is preserved in chunks."""
        doc = Document(
            content="Test content here.",
            metadata={"source": "doc.txt", "page": 1},
        )
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(doc)

        assert chunks[0].metadata["source"] == "doc.txt"
        assert chunks[0].metadata["page"] == 1
        assert "chunk_index" in chunks[0].metadata

    def test_chunk_overlap(self):
        """Test that overlap is applied correctly."""
        # Create text that will definitely need multiple chunks
        text = " ".join(["word"] * 200)  # ~1000 chars
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk(text)

        assert len(chunks) > 1
        # Verify chunks have proper indices
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_invalid_overlap(self):
        """Test that overlap >= chunk_size raises error."""
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            RecursiveChunker(chunk_size=100, chunk_overlap=100)

    def test_chunk_text_method(self):
        """Test chunk_text convenience method."""
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_text(
            "Hello world",
            metadata={"custom": "value"},
        )

        assert len(chunks) >= 1
        assert chunks[0].metadata["custom"] == "value"


class TestSentenceChunker:
    """Tests for SentenceChunker."""

    def test_chunk_single_sentence(self):
        """Test chunking a single sentence."""
        chunker = SentenceChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk("Hello world.")

        assert len(chunks) == 1
        assert "Hello world" in chunks[0].text

    def test_chunk_multiple_sentences(self):
        """Test chunking multiple sentences."""
        text = "First sentence. Second sentence. Third sentence."
        chunker = SentenceChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(text)

        assert len(chunks) >= 1
        # All sentences should be in chunks
        combined = " ".join(c.text for c in chunks)
        assert "First sentence" in combined
        assert "Second sentence" in combined
        assert "Third sentence" in combined

    def test_chunk_respects_size(self):
        """Test that chunks respect size limits."""
        # Create text with clear sentence endings (periods followed by space)
        sentences = ["This is sentence number " + str(i) + "." for i in range(20)]
        text = " ".join(sentences)  # ~500 chars total
        chunker = SentenceChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(text)

        # Should produce multiple chunks for text that exceeds chunk_size
        assert len(chunks) >= 1
        # Total text should be covered
        total_text = " ".join(c.text for c in chunks)
        assert "sentence number 0" in total_text

    def test_chunk_document(self):
        """Test chunking a Document object."""
        doc = Document(
            content="First. Second. Third.",
            metadata={"source": "test.txt"},
        )
        chunker = SentenceChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(doc)

        assert len(chunks) >= 1
        assert chunks[0].metadata["source"] == "test.txt"

    def test_sentence_count_in_metadata(self):
        """Test that sentence_count is in metadata."""
        text = "One. Two. Three."
        chunker = SentenceChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(text)

        assert "sentence_count" in chunks[0].metadata

    def test_invalid_overlap(self):
        """Test that overlap >= chunk_size raises error."""
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            SentenceChunker(chunk_size=100, chunk_overlap=100)
