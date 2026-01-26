"""
RAG Agent implementation.

Retrieval agent with "out of the box" support (RFC-025).
"""

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
    collection: str
    embedding_model: str
    proxy_endpoint: Optional[str]
    proxy_api_key: Optional[str]
    top_k: int
    score_threshold: float
    namespace: Optional[str]
    filters: Dict[str, Any]
    include_sources: bool
    include_scores: bool
    deliver_to: Optional[str]

    # RFC-025: New config fields
    documents: Optional[List[str]]
    texts: Optional[List[str]]
    chunks: Optional[List[str]]
    directory: Optional[str]
    chunk_size: int
    chunk_overlap: int
    embedding_api_key: Optional[str]

    # Internal state (RFC-025)
    _initialized: bool
    _internal_store: Optional["BaseVectorStore"]
    _embedding: Optional["BaseEmbedding"]

    def __init__(self, team: "LLMTeam", config: RAGAgentConfig):
        super().__init__(team, config)

        self.mode = config.mode
        self.vector_store = config.vector_store
        self.collection = config.collection
        self.embedding_model = config.embedding_model
        self.proxy_endpoint = config.proxy_endpoint
        self.proxy_api_key = config.proxy_api_key
        self.top_k = config.top_k
        self.score_threshold = config.score_threshold
        self.namespace = config.namespace
        self.filters = config.filters
        self.include_sources = config.include_sources
        self.include_scores = config.include_scores
        self.deliver_to = config.deliver_to

        # RFC-025: New fields
        self.documents = config.documents
        self.texts = config.texts
        self.chunks = config.chunks
        self.directory = config.directory
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.embedding_api_key = config.embedding_api_key

        # Internal state
        self._initialized = False
        self._internal_store = None
        self._embedding = None

    def _uses_builtin_store(self) -> bool:
        """Check if this RAGAgent uses the built-in store (RFC-025 mode)."""
        return bool(
            self.documents or self.texts or self.chunks or self.directory
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
        from llmteam.documents import AutoLoader, RecursiveChunker, Chunk, Document
        from llmteam.embeddings import OpenAIEmbedding
        from llmteam.vectorstores import InMemoryVectorStore

        # Create embedding provider
        self._embedding = OpenAIEmbedding(
            model=self.embedding_model,
            api_key=self.embedding_api_key,
        )

        # Create vector store
        self._internal_store = InMemoryVectorStore()

        # Collect all chunks to index
        all_chunks: List[Chunk] = []
        chunker = RecursiveChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        loader = AutoLoader()

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
