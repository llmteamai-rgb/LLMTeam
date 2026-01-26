"""
Tests for document loaders.
"""

import tempfile
from pathlib import Path

import pytest

from llmteam.documents import Document
from llmteam.documents.loaders import TextLoader, AutoLoader, load_document, load_documents


class TestTextLoader:
    """Tests for TextLoader."""

    def test_load_txt_file(self, tmp_path: Path):
        """Test loading a .txt file."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Hello, world!\nThis is a test.")

        loader = TextLoader()
        docs = loader.load(file_path)

        assert len(docs) == 1
        assert docs[0].content == "Hello, world!\nThis is a test."
        assert docs[0].metadata["filename"] == "test.txt"
        assert docs[0].metadata["extension"] == ".txt"

    def test_load_md_file(self, tmp_path: Path):
        """Test loading a .md file."""
        file_path = tmp_path / "readme.md"
        file_path.write_text("# Title\n\nContent here.")

        loader = TextLoader()
        docs = loader.load(file_path)

        assert len(docs) == 1
        assert "# Title" in docs[0].content

    def test_supports_extensions(self):
        """Test that supports() works correctly."""
        loader = TextLoader()

        assert loader.supports("file.txt")
        assert loader.supports("file.md")
        assert loader.supports("file.markdown")
        assert loader.supports("file.rst")
        assert loader.supports("file.text")
        assert not loader.supports("file.pdf")
        assert not loader.supports("file.docx")

    def test_file_not_found(self, tmp_path: Path):
        """Test error on missing file."""
        loader = TextLoader()

        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "nonexistent.txt")

    def test_unsupported_extension(self, tmp_path: Path):
        """Test error on unsupported extension."""
        file_path = tmp_path / "test.xyz"
        file_path.write_text("content")

        loader = TextLoader()

        with pytest.raises(ValueError, match="Unsupported file extension"):
            loader.load(file_path)


class TestAutoLoader:
    """Tests for AutoLoader."""

    def test_auto_detect_txt(self, tmp_path: Path):
        """Test automatic detection of .txt files."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Test content")

        loader = AutoLoader()
        docs = loader.load(file_path)

        assert len(docs) == 1
        assert docs[0].content == "Test content"

    def test_auto_detect_md(self, tmp_path: Path):
        """Test automatic detection of .md files."""
        file_path = tmp_path / "readme.md"
        file_path.write_text("# Markdown")

        loader = AutoLoader()
        docs = loader.load(file_path)

        assert len(docs) == 1
        assert "# Markdown" in docs[0].content

    def test_load_many(self, tmp_path: Path):
        """Test loading multiple files."""
        (tmp_path / "file1.txt").write_text("Content 1")
        (tmp_path / "file2.txt").write_text("Content 2")

        loader = AutoLoader()
        docs = loader.load_many([
            tmp_path / "file1.txt",
            tmp_path / "file2.txt",
        ])

        assert len(docs) == 2
        contents = [d.content for d in docs]
        assert "Content 1" in contents
        assert "Content 2" in contents

    def test_unsupported_extension(self, tmp_path: Path):
        """Test error on unsupported extension."""
        file_path = tmp_path / "test.xyz"
        file_path.write_text("content")

        loader = AutoLoader()

        with pytest.raises(ValueError, match="Unsupported file extension"):
            loader.load(file_path)

    def test_supported_extensions(self):
        """Test supported_extensions class method."""
        extensions = AutoLoader.supported_extensions()

        assert ".txt" in extensions
        assert ".md" in extensions
        assert ".pdf" in extensions
        assert ".docx" in extensions


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_load_document(self, tmp_path: Path):
        """Test load_document function."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Hello")

        docs = load_document(file_path)

        assert len(docs) == 1
        assert docs[0].content == "Hello"

    def test_load_documents(self, tmp_path: Path):
        """Test load_documents function."""
        (tmp_path / "a.txt").write_text("A")
        (tmp_path / "b.txt").write_text("B")

        docs = load_documents([tmp_path / "a.txt", tmp_path / "b.txt"])

        assert len(docs) == 2
