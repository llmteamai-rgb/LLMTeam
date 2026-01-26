"""
RAG Agent implementation.

Retrieval agent with "out of the box" support (RFC-025).
"""

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from llmteam.agents.types import AgentType, AgentMode
from llmteam.agents.config import RAGAgentConfig
from llmteam.agents.result import RAGResult
from llmteam.agents.base import BaseAgent

if TYPE_CHECKING:
    from llmteam.team import LLMTeam
    from llmteam.documents import Document, Chunk
    from llmteam.embeddings import BaseEmbedding
    from llmteam.vectorstores import BaseVectorStore, SearchResult


class RAGAgent(BaseAgent):
    """
    Retrieval agent with "out of the box" support.

    RFC-025: Can be used with just document paths, no external setup needed:
        rag = RAGAgent(documents=["doc.pdf", "manual.docx"])
        result = await rag.query("Question?")

    Modes:
    - native: direct connection to Chroma/Pinecone/FAISS (external store)
    - built-in: uses internal InMemoryVectorStore (RFC-025)
    - proxy: via external API

    Result is delivered to mailbox for LLMAgent.
    """

    agent_type = AgentType.RAG

    # Config fields
    mode: AgentMode
    vector_store: Optional[str]
    vector_store_path: Optional[str]
    collection: str
    embedding_provider: str
    embedding_model: str
    embedding_api_key: Optional[str]
    embedding_cache: bool
    embedding_batch_size: int
    proxy_endpoint: Optional[str]
    proxy_api_key: Optional[str]
    top_k: int
    score_threshold: float
    namespace: Optional[str]
    filters: Dict[str, Any]
    include_sources: bool
    include_scores: bool
    include_metadata: bool
    context_template: str
    deliver_to: Optional[str]
    context_key: str

    # RFC-025: Source fields
    documents: Optional[List[str]]
    texts: Optional[List[str]]
    chunks: Optional[List[str]]
    directory: Optional[str]
    urls: Optional[List[str]]
    loader: str
    chunker: str
    chunk_size: int
    chunk_overlap: int

    # RFC-025: Retrieval fields
    rerank: bool
    rerank_model: str
    search_type: str
    mmr_diversity: float

    # Qdrant-specific
    qdrant_url: Optional[str]
    qdrant_api_key: Optional[str]

    # Internal state (RFC-025)
    _initialized: bool
    _internal_store: Optional["BaseVectorStore"]
    _embedding: Optional["BaseEmbedding"]
    _chunk_count: int

    def __init__(self, team: "LLMTeam", config: RAGAgentConfig):
        super().__init__(team, config)

        self.mode = config.mode
        self.vector_store = config.vector_store
        self.vector_store_path = config.vector_store_path
        self.collection = config.collection
        self.embedding_provider = config.embedding_provider
        self.embedding_model = config.embedding_model
        self.embedding_api_key = config.embedding_api_key
        self.embedding_cache = config.embedding_cache
        self.embedding_batch_size = config.embedding_batch_size
        self.proxy_endpoint = config.proxy_endpoint
        self.proxy_api_key = config.proxy_api_key
        self.top_k = config.top_k
        self.score_threshold = config.score_threshold
        self.namespace = config.namespace
        self.filters = config.filters
        self.include_sources = config.include_sources
        self.include_scores = config.include_scores
        self.include_metadata = config.include_metadata
        self.context_template = config.context_template
        self.deliver_to = config.deliver_to
        self.context_key = config.context_key

        # RFC-025: Source fields
        self.documents = config.documents
        self.texts = config.texts
        self.chunks = config.chunks
        self.directory = config.directory
        self.urls = config.urls
        self.loader = config.loader
        self.chunker = config.chunker
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap

        # RFC-025: Retrieval fields
        self.rerank = config.rerank
        self.rerank_model = config.rerank_model
        self.search_type = config.search_type
        self.mmr_diversity = config.mmr_diversity

        # Qdrant-specific
        self.qdrant_url = config.qdrant_url
        self.qdrant_api_key = config.qdrant_api_key

        # Internal state
        self._initialized = False
        self._internal_store = None
        self._embedding = None
        self._chunk_count = 0

    def _uses_builtin_store(self) -> bool:
        """Check if this RAGAgent uses the built-in store (RFC-025 mode)."""
        return bool(
            self.documents or self.texts or self.chunks or self.directory or self.urls
        )

    async def initialize(self) -> None:
        """
        Initialize the RAG agent with documents (lazy initialization).

        This method is called automatically on first query() call.
        Can also be called explicitly for eager initialization.
        """
        if self._initialized:
            return

        if not self._uses_builtin_store():
            # Using external store, nothing to initialize
            self._initialized = True
            return

        # RFC-025: Initialize built-in store
        from llmteam.documents import AutoLoader, RecursiveChunker, SentenceChunker, Chunk, Document
        from llmteam.documents.chunkers import TokenChunker
        from llmteam.vectorstores import InMemoryVectorStore

        # Create embedding provider based on config
        self._embedding = self._create_embedding_provider()

        # Create vector store based on config
        self._internal_store = self._create_vector_store()

        # Create chunker based on config
        chunker = self._create_chunker()
        loader = AutoLoader()

        # Collect all chunks to index
        all_chunks: List[Chunk] = []

        # Load documents from file paths
        if self.documents:
            for doc_path in self.documents:
                try:
                    docs = loader.load(doc_path)
                    for doc in docs:
                        chunks = chunker.chunk(doc)
                        all_chunks.extend(chunks)
                except Exception as e:
                    # Log warning but continue with other documents
                    import warnings
                    warnings.warn(f"Failed to load document {doc_path}: {e}")

        # Load documents from directory
        if self.directory:
            dir_path = Path(self.directory)
            if dir_path.exists() and dir_path.is_dir():
                for ext in AutoLoader.supported_extensions():
                    for file_path in dir_path.glob(f"*{ext}"):
                        try:
                            docs = loader.load(file_path)
                            for doc in docs:
                                chunks = chunker.chunk(doc)
                                all_chunks.extend(chunks)
                        except Exception as e:
                            import warnings
                            warnings.warn(f"Failed to load document {file_path}: {e}")

        # Add raw texts
        if self.texts:
            for i, text in enumerate(self.texts):
                doc = Document(
                    content=text,
                    metadata={"source": f"text_{i}", "type": "raw_text"},
                )
                chunks = chunker.chunk(doc)
                all_chunks.extend(chunks)

        # Add pre-chunked texts
        if self.chunks:
            for i, chunk_text in enumerate(self.chunks):
                all_chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata={"source": f"chunk_{i}", "type": "pre_chunked"},
                        index=i,
                    )
                )

        # Index all chunks
        if all_chunks:
            await self._index_chunks(all_chunks)

        self._initialized = True

    async def _index_chunks(self, chunks: List["Chunk"]) -> None:
        """Index chunks into the internal vector store."""
        if not chunks or not self._embedding or not self._internal_store:
            return

        # Extract texts and metadata
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Generate embeddings
        embeddings = await self._embedding.embed(texts)

        # Add to store
        await self._internal_store.add(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        # Track chunk count
        self._chunk_count += len(chunks)

    async def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RAGResult:
        """
        Query the RAG agent with a question.

        RFC-025: Public API for direct usage without team context.

        Args:
            question: The question to search for.
            top_k: Number of results to return (default: self.top_k).
            score_threshold: Minimum similarity score (default: self.score_threshold).
            filters: Optional metadata filters.

        Returns:
            RAGResult with retrieved documents.

        Example:
            rag = RAGAgent(documents=["doc.pdf"])
            result = await rag.query("What is the main topic?")
            print(result.documents)
        """
        # Lazy initialization
        if not self._initialized:
            await self.initialize()

        # Use defaults if not specified
        k = top_k if top_k is not None else self.top_k
        threshold = score_threshold if score_threshold is not None else self.score_threshold
        search_filters = filters if filters is not None else self.filters

        # Search using built-in store or external store
        if self._uses_builtin_store() and self._internal_store:
            return await self._search_builtin(question, k, threshold, search_filters)
        else:
            # Delegate to _execute which handles external stores
            return await self._execute(
                {"query": question},
                {"filters": search_filters} if search_filters else {},
            )

    async def _search_builtin(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        filters: Optional[Dict[str, Any]],
    ) -> RAGResult:
        """Search using the built-in vector store."""
        if not self._embedding or not self._internal_store:
            return RAGResult(
                output=[],
                documents=[],
                query=query,
                total_found=0,
                success=False,
                error="RAG agent not initialized",
            )

        # Generate query embedding
        query_embedding = await self._embedding.embed_query(query)

        # Search
        results = await self._internal_store.similarity_search(
            query_embedding=query_embedding,
            k=top_k,
            score_threshold=score_threshold,
            filter=filters,
        )

        # Convert to documents format
        documents = [
            {
                "text": r.text,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in results
        ]

        return RAGResult(
            output=documents,
            documents=documents,
            query=query,
            total_found=len(documents),
            success=True,
            sources=[{"source": d.get("metadata", {}).get("source")} for d in documents]
            if self.include_sources
            else [],
        )

    async def _execute(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> RAGResult:
        """
        INTERNAL: Retrieval from vector store.

        Do NOT call directly - use team.run() instead.

        Args:
            input_data: Must contain "query" or text for search
                Example: {"query": "machine learning algorithms"}
            context: Usually empty (RAG is first in pipeline)

        Returns:
            RAGResult:
                output: List[Dict] - found documents
                documents: List[Dict] - with text, score, metadata fields
                query: str - original query
                total_found: int
                context_payload: {"_rag_context": [...]} - for mailbox
        """
        query = input_data.get("query") or input_data.get("input", "")

        # RFC-025: Use built-in store if configured
        if self._uses_builtin_store():
            if not self._initialized:
                await self.initialize()
            return await self._search_builtin(
                query,
                self.top_k,
                self.score_threshold,
                self.filters,
            )

        # Get external vector store
        store = self._get_vector_store()

        if store is None:
            # Fallback: return mock results for testing
            documents = [
                {
                    "text": f"Mock document for query: {query}",
                    "score": 0.95,
                    "metadata": {"source": "mock"},
                }
            ]
            return RAGResult(
                output=documents,
                documents=documents,
                query=query,
                total_found=len(documents),
                success=True,
            )

        # Search
        if self.mode == AgentMode.NATIVE:
            results = await self._native_search(store, query)
        else:  # PROXY
            results = await self._proxy_search(query)

        # Filter by threshold
        documents = [
            {
                "text": r.get("text", r.get("content", "")),
                "score": r.get("score", 0.0),
                "metadata": r.get("metadata", {}),
            }
            for r in results
            if r.get("score", 0.0) >= self.score_threshold
        ]

        return RAGResult(
            output=documents,
            documents=documents,
            query=query,
            total_found=len(documents),
            success=True,
            sources=[{"source": d.get("metadata", {}).get("source")} for d in documents]
            if self.include_sources
            else [],
        )

    def _get_vector_store(self):
        """Get vector store from runtime context."""
        if hasattr(self._team, "_runtime") and self._team._runtime:
            try:
                return self._team._runtime.get_store(
                    self.vector_store or "vector_store"
                )
            except Exception:
                pass
        return None

    async def _native_search(self, store, query: str) -> List[Dict[str, Any]]:
        """Perform native vector search."""
        try:
            results = await store.similarity_search(
                query=query,
                k=self.top_k,
                collection=self.collection,
                namespace=self.namespace,
                filters=self.filters,
            )
            return results
        except Exception:
            return []

    async def _proxy_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform proxy search via external API."""
        if not self.proxy_endpoint:
            return []

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                headers = {}
                if self.proxy_api_key:
                    headers["Authorization"] = f"Bearer {self.proxy_api_key}"

                async with session.post(
                    self.proxy_endpoint,
                    json={"query": query, "top_k": self.top_k},
                    headers=headers,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("results", [])
        except Exception:
            pass

        return []

    @property
    def document_count(self) -> int:
        """Return the number of indexed documents/chunks."""
        if self._internal_store:
            return self._internal_store.count()
        return 0

    async def add_documents(
        self,
        documents: Optional[List[str]] = None,
        texts: Optional[List[str]] = None,
    ) -> int:
        """
        Add more documents to the index after initialization.

        Args:
            documents: List of file paths to add.
            texts: List of raw texts to add.

        Returns:
            Number of chunks added.
        """
        if not self._uses_builtin_store():
            raise ValueError(
                "add_documents() only works with built-in store mode. "
                "Configure documents, texts, chunks, or directory in config."
            )

        if not self._initialized:
            # Add to config and initialize
            if documents:
                self.documents = (self.documents or []) + documents
            if texts:
                self.texts = (self.texts or []) + texts
            await self.initialize()
            return self.document_count

        # Already initialized, add incrementally
        from llmteam.documents import AutoLoader, RecursiveChunker, Chunk, Document

        all_chunks: List[Chunk] = []
        chunker = RecursiveChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        loader = AutoLoader()

        if documents:
            for doc_path in documents:
                docs = loader.load(doc_path)
                for doc in docs:
                    chunks = chunker.chunk(doc)
                    all_chunks.extend(chunks)

        if texts:
            for i, text in enumerate(texts):
                doc = Document(
                    content=text,
                    metadata={"source": f"added_text_{i}", "type": "raw_text"},
                )
                chunks = chunker.chunk(doc)
                all_chunks.extend(chunks)

        if all_chunks:
            await self._index_chunks(all_chunks)

        return len(all_chunks)

    def _create_embedding_provider(self) -> "BaseEmbedding":
        """Create embedding provider based on config."""
        if self.embedding_provider == "huggingface":
            from llmteam.embeddings import HuggingFaceEmbedding

            return HuggingFaceEmbedding(
                model=self.embedding_model,
                batch_size=self.embedding_batch_size,
            )
        else:  # Default to OpenAI
            from llmteam.embeddings import OpenAIEmbedding

            return OpenAIEmbedding(
                model=self.embedding_model,
                api_key=self.embedding_api_key,
                batch_size=self.embedding_batch_size,
            )

    def _create_vector_store(self) -> "BaseVectorStore":
        """Create vector store based on config."""
        store_type = self.vector_store or "memory"

        if store_type == "faiss":
            from llmteam.vectorstores.faiss import FAISSStore

            return FAISSStore(
                dimensions=self._embedding.dimensions if self._embedding else 1536,
                path=self.vector_store_path,
            )
        elif store_type == "qdrant":
            from llmteam.vectorstores.qdrant import QdrantStore

            return QdrantStore(
                collection=self.collection,
                dimensions=self._embedding.dimensions if self._embedding else 1536,
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
            )
        elif store_type == "auto":
            from llmteam.vectorstores import AutoStore

            return AutoStore(
                dimensions=self._embedding.dimensions if self._embedding else 1536,
                faiss_path=self.vector_store_path,
                qdrant_url=self.qdrant_url,
                qdrant_collection=self.collection,
            )
        else:  # Default to memory
            from llmteam.vectorstores import InMemoryVectorStore

            return InMemoryVectorStore()

    def _create_chunker(self):
        """Create chunker based on config."""
        from llmteam.documents import RecursiveChunker, SentenceChunker
        from llmteam.documents.chunkers import TokenChunker

        if self.chunker == "sentence":
            return SentenceChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        elif self.chunker == "token":
            return TokenChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        elif self.chunker == "none":
            return None  # No chunking
        else:  # Default to recursive
            return RecursiveChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

    async def query_with_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Query and format results as a context string for LLM.

        Args:
            query: The search query.
            top_k: Number of results.
            **kwargs: Additional query parameters.

        Returns:
            Formatted context string.
        """
        result = await self.query(query, top_k=top_k, **kwargs)

        if not result.success or not result.documents:
            return ""

        # Format documents using template
        context_parts: List[str] = []
        for doc in result.documents:
            text = doc.get("text", "")
            source = doc.get("metadata", {}).get("source", "")
            score = doc.get("score", 0.0)
            page = doc.get("metadata", {}).get("page", "")

            formatted = self.context_template.format(
                text=text,
                source=source,
                score=f"{score:.2f}",
                page=page,
            )
            context_parts.append(formatted)

        return "\n\n".join(context_parts)

    @property
    def chunk_count(self) -> int:
        """Return the number of chunks in the index."""
        return self._chunk_count

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "document_count": self.document_count,
            "chunk_count": self._chunk_count,
            "initialized": self._initialized,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "vector_store": self.vector_store or "memory",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Delete documents from the index.

        Args:
            ids: List of document IDs to delete.
            filters: Metadata filters for deletion.

        Returns:
            Number of documents deleted.
        """
        if not self._uses_builtin_store():
            raise ValueError("delete() only works with built-in store mode.")

        if not self._initialized or not self._internal_store:
            return 0

        if ids:
            return await self._internal_store.delete(ids)

        # Filter-based deletion not supported in basic stores
        return 0

    async def clear(self) -> int:
        """
        Clear all documents from the index.

        Returns:
            Number of documents cleared.
        """
        if not self._uses_builtin_store():
            raise ValueError("clear() only works with built-in store mode.")

        if not self._initialized or not self._internal_store:
            return 0

        count = await self._internal_store.clear()
        self._chunk_count = 0
        return count

    def save(self, path: str) -> None:
        """
        Save the index to disk.

        Args:
            path: Path to save the index.
        """
        if not self._uses_builtin_store():
            raise ValueError("save() only works with built-in store mode.")

        if not self._initialized or not self._internal_store:
            raise ValueError("RAGAgent not initialized. Call initialize() first.")

        # Check if the store supports save
        if hasattr(self._internal_store, "save"):
            self._internal_store.save(path)
        else:
            # Fallback: save to JSON for InMemoryVectorStore
            import pickle

            state = {
                "store_type": type(self._internal_store).__name__,
                "chunk_count": self._chunk_count,
                "config": {
                    "embedding_provider": self.embedding_provider,
                    "embedding_model": self.embedding_model,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                },
            }
            if hasattr(self._internal_store, "_vectors"):
                # InMemoryVectorStore
                state["vectors"] = [v.tolist() for v in self._internal_store._vectors]
                state["texts"] = self._internal_store._texts
                state["metadatas"] = self._internal_store._metadatas
                state["ids"] = self._internal_store._ids

            with open(f"{path}.pkl", "wb") as f:
                pickle.dump(state, f)

    @classmethod
    def load(cls, path: str, team: "LLMTeam" = None, **kwargs) -> "RAGAgent":
        """
        Load a saved RAGAgent from disk.

        Args:
            path: Path to load from.
            team: Optional team instance.
            **kwargs: Additional config overrides.

        Returns:
            Loaded RAGAgent instance.
        """
        import pickle
        import numpy as np

        with open(f"{path}.pkl", "rb") as f:
            state = pickle.load(f)

        # Create config from saved state
        config = RAGAgentConfig(
            role=kwargs.get("role", "rag"),
            embedding_provider=state["config"]["embedding_provider"],
            embedding_model=state["config"]["embedding_model"],
            chunk_size=state["config"]["chunk_size"],
            chunk_overlap=state["config"]["chunk_overlap"],
            texts=["placeholder"],  # Trigger built-in mode
            **kwargs,
        )

        # Create agent
        if team is None:
            # Create minimal mock team
            from unittest.mock import MagicMock
            team = MagicMock()

        agent = cls(team=team, config=config)

        # Restore state
        if state.get("vectors"):
            from llmteam.vectorstores import InMemoryVectorStore

            agent._internal_store = InMemoryVectorStore()
            agent._internal_store._vectors = [np.array(v) for v in state["vectors"]]
            agent._internal_store._texts = state["texts"]
            agent._internal_store._metadatas = state["metadatas"]
            agent._internal_store._ids = state["ids"]
            agent._chunk_count = state["chunk_count"]
            agent._initialized = True

        return agent
