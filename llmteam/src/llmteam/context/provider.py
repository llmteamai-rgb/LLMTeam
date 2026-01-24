"""
Context Provider Abstraction.

Decouples Knowledge Retrieval from Agent Logic.
Supports two modes:
1. Native: Local management of embeddings and retrieval (default).
2. Proxy: Delegated retrieval to external platform (e.g., KorpOS).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


from enum import Enum

@dataclass
class RetrievalQuery:
    """Standardized retrieval query."""
    query: str
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None
    namespace: Optional[str] = None
    include_metadata: bool = True


class RetrievalMode(str, Enum):
    """Mode for context retrieval (native vs proxy)."""
    NATIVE = "native"
    PROXY = "proxy"


# Backward compatibility alias
ContextMode = RetrievalMode


@dataclass
class RetrievalResult:
    """Standardized retrieval result."""
    content: str
    score: float
    metadata: Dict[str, Any]
    source_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "source_id": self.source_id
        }


@dataclass
class ContextResponse:
    """Response from context provider."""
    results: List[RetrievalResult]
    query_time_ms: int = 0

    def to_context_string(self) -> str:
        """Convert results to string for LLM context."""
        return "\\n\\n".join([f"[{r.source_id}] {r.content}" for r in self.results])


class ContextProvider(ABC):
    """Abstract base class for context providers."""

    @abstractmethod
    async def retrieve(
        self,
        query: RetrievalQuery,
    ) -> "ContextResponse":
        """
        Retrieve relevant context for a query.
        """
        pass

    @abstractmethod
    async def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add document to context (if supported).
        
        Returns:
            Document ID
        """
        pass


class NativeContextProvider(ContextProvider):
    """
    Native implementation using local vector store.
    (Placeholder for existing logic)
    """

    def __init__(self, collection_name: str = "default"):
        self.collection_name = collection_name
        # TODO: Initialize local vector store (Chroma/FAISS)

    async def retrieve(
        self,
        query: RetrievalQuery,
    ) -> ContextResponse:
        # Simulating local retrieval
        return ContextResponse(
            results=[
                RetrievalResult(
                    content=f"Native result for {query.query}",
                    score=0.9,
                    metadata={"source": "local", "filters": query.filters},
                    source_id="local_1"
                )
            ],
            query_time_ms=10
        )

    async def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        return "doc_id_123"


class ProxyContextProvider(ContextProvider):
    """
    Proxy implementation delegating to external Context API.
    Used for KorpOS integration.
    """

    def __init__(self, endpoint: str, api_key: Optional[str] = None):
        self.endpoint = endpoint
        self.api_key = api_key

    async def retrieve(
        self,
        query: RetrievalQuery,
    ) -> ContextResponse:
        # Call external API
        # response = await self.client.post(...)
        
        # Mocking external response
        return ContextResponse(
            results=[
                RetrievalResult(
                    content=f"Proxy result from {self.endpoint} for {query.query}",
                    score=0.95,
                    metadata={"source": "remote", "endpoint": self.endpoint},
                    source_id="remote_1"
                )
            ],
            query_time_ms=45
        )

    async def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        # Proxy might not support adding documents if read-only
        raise NotImplementedError("Proxy context is read-only via this agent")


def create_provider(
    mode: RetrievalMode,
    config: Optional[Dict[str, Any]] = None
) -> ContextProvider:
    """Factory to create context provider."""
    config = config or {}
    if mode == RetrievalMode.PROXY:
        return ProxyContextProvider(
            endpoint=config.get("proxy_endpoint", ""),
            api_key=config.get("proxy_api_key"),
        )
    return NativeContextProvider(collection_name=config.get("collection", "default"))
