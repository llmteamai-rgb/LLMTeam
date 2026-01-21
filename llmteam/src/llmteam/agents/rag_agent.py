"""
RAG Agent implementation.

Retrieval agent from vector store.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from llmteam.agents.types import AgentType, AgentMode
from llmteam.agents.config import RAGAgentConfig
from llmteam.agents.result import RAGResult
from llmteam.agents.base import BaseAgent

if TYPE_CHECKING:
    from llmteam.team import LLMTeam


class RAGAgent(BaseAgent):
    """
    Retrieval agent from vector store.

    Modes:
    - native: direct connection to Chroma/Pinecone/FAISS
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

        # Get vector store
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
