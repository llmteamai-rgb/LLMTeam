"""
Context Provider Abstraction for RAG/KAG Integration.

Decouples knowledge retrieval from agents, enabling:
- Native mode: Local retrieval (current behavior)
- Proxy mode: External API delegation (VCR/KorpOS integration)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable
import asyncio


class ContextMode(str, Enum):
    """Context retrieval mode."""

    NATIVE = "native"  # Local retrieval
    PROXY = "proxy"  # External API delegation


@dataclass
class RetrievalQuery:
    """
    Query for context retrieval.

    Attributes:
        query: The search/retrieval query
        top_k: Number of results to return
        filters: Additional filters (metadata, tags, etc.)
        namespace: Optional namespace/collection to search
        include_metadata: Whether to include metadata in results
    """

    query: str
    top_k: int = 5
    filters: dict[str, Any] = field(default_factory=dict)
    namespace: Optional[str] = None
    include_metadata: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "top_k": self.top_k,
            "filters": self.filters,
            "namespace": self.namespace,
            "include_metadata": self.include_metadata,
        }


@dataclass
class RetrievalResult:
    """
    Result from context retrieval.

    Attributes:
        content: Retrieved content/text
        score: Relevance score (0-1)
        metadata: Additional metadata
        source: Source identifier
        chunk_id: Chunk/segment identifier
    """

    content: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    chunk_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "source": self.source,
            "chunk_id": self.chunk_id,
        }


@dataclass
class ContextResponse:
    """
    Response from context provider.

    Attributes:
        results: List of retrieval results
        total_results: Total number of matching results
        query_time_ms: Query execution time in milliseconds
        provider_metadata: Provider-specific metadata
    """

    results: list[RetrievalResult]
    total_results: int = 0
    query_time_ms: float = 0.0
    provider_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "results": [r.to_dict() for r in self.results],
            "total_results": self.total_results,
            "query_time_ms": self.query_time_ms,
            "provider_metadata": self.provider_metadata,
        }

    def to_context_string(self, separator: str = "\n\n---\n\n") -> str:
        """Convert results to a single context string for LLM prompts."""
        return separator.join(r.content for r in self.results)


@runtime_checkable
class ContextProvider(Protocol):
    """
    Protocol for context providers.

    Implement this to create custom context retrieval backends.
    """

    @property
    def mode(self) -> ContextMode:
        """Get the provider mode."""
        ...

    async def retrieve(self, query: RetrievalQuery) -> ContextResponse:
        """
        Retrieve context based on query.

        Args:
            query: The retrieval query

        Returns:
            ContextResponse with retrieved results
        """
        ...

    async def health_check(self) -> bool:
        """Check if provider is healthy and accessible."""
        ...


class BaseContextProvider(ABC):
    """Abstract base class for context providers."""

    def __init__(self, mode: ContextMode = ContextMode.NATIVE):
        self._mode = mode

    @property
    def mode(self) -> ContextMode:
        return self._mode

    @abstractmethod
    async def retrieve(self, query: RetrievalQuery) -> ContextResponse:
        """Retrieve context."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check provider health."""
        pass


class NativeContextProvider(BaseContextProvider):
    """
    Native context provider using local retrieval.

    Integrates with local vector stores, databases, or in-memory stores.
    """

    def __init__(
        self,
        store: Optional[Any] = None,  # VectorStore or similar
        embedding_fn: Optional[Any] = None,  # Embedding function
    ):
        super().__init__(mode=ContextMode.NATIVE)
        self._store = store
        self._embedding_fn = embedding_fn
        self._documents: list[dict[str, Any]] = []  # In-memory fallback

    def add_document(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """Add a document to the local store."""
        import uuid

        doc_id = doc_id or str(uuid.uuid4())
        self._documents.append({
            "id": doc_id,
            "content": content,
            "metadata": metadata or {},
        })
        return doc_id

    async def retrieve(self, query: RetrievalQuery) -> ContextResponse:
        """Retrieve from local store."""
        import time

        start = time.time()

        # If we have a vector store, use it
        if self._store is not None:
            # Assuming store has a similarity_search method
            results = await self._similarity_search(query)
        else:
            # Simple keyword matching fallback
            results = self._keyword_search(query)

        elapsed = (time.time() - start) * 1000

        return ContextResponse(
            results=results,
            total_results=len(results),
            query_time_ms=elapsed,
            provider_metadata={"mode": "native"},
        )

    async def _similarity_search(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Perform similarity search using vector store."""
        # This would integrate with actual vector store
        # For now, fall back to keyword search
        return self._keyword_search(query)

    def _keyword_search(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Simple keyword-based search."""
        query_lower = query.query.lower()
        results = []

        for doc in self._documents:
            content = doc["content"]
            # Simple scoring: count query words in content
            words = query_lower.split()
            score = sum(1 for w in words if w in content.lower()) / max(len(words), 1)

            if score > 0:
                results.append(RetrievalResult(
                    content=content,
                    score=score,
                    metadata=doc.get("metadata", {}),
                    chunk_id=doc["id"],
                ))

        # Sort by score and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[: query.top_k]

    async def health_check(self) -> bool:
        """Check if local store is accessible."""
        return True  # In-memory is always available


class ProxyContextProvider(BaseContextProvider):
    """
    Proxy context provider for external API delegation.

    Delegates retrieval to external RAG/KAG services (VCR, KorpOS).
    """

    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
        timeout: float = 30.0,
    ):
        super().__init__(mode=ContextMode.PROXY)
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.headers = headers or {}
        self.timeout = timeout

        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    async def retrieve(self, query: RetrievalQuery) -> ContextResponse:
        """Delegate retrieval to external service."""
        import time

        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp is required for ProxyContextProvider")

        start = time.time()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.endpoint}/retrieve",
                json=query.to_dict(),
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Proxy retrieval failed: {response.status} {await response.text()}"
                    )

                data = await response.json()

        elapsed = (time.time() - start) * 1000

        # Parse response
        results = [
            RetrievalResult(
                content=r.get("content", ""),
                score=r.get("score", 0.0),
                metadata=r.get("metadata", {}),
                source=r.get("source"),
                chunk_id=r.get("chunk_id"),
            )
            for r in data.get("results", [])
        ]

        return ContextResponse(
            results=results,
            total_results=data.get("total_results", len(results)),
            query_time_ms=elapsed,
            provider_metadata={
                "mode": "proxy",
                "endpoint": self.endpoint,
                "server_time_ms": data.get("query_time_ms", 0),
            },
        )

    async def health_check(self) -> bool:
        """Check if external service is accessible."""
        try:
            import aiohttp
        except ImportError:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.endpoint}/health",
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as response:
                    return response.status == 200
        except Exception:
            return False


class CompositeContextProvider(BaseContextProvider):
    """
    Composite provider that combines multiple providers.

    Can merge results from multiple sources (e.g., local + remote).
    """

    def __init__(
        self,
        providers: list[BaseContextProvider],
        strategy: str = "merge",  # "merge", "fallback", "round_robin"
    ):
        super().__init__(mode=ContextMode.NATIVE)
        self.providers = providers
        self.strategy = strategy
        self._current_index = 0

    async def retrieve(self, query: RetrievalQuery) -> ContextResponse:
        """Retrieve from multiple providers based on strategy."""
        if self.strategy == "merge":
            return await self._merge_retrieve(query)
        elif self.strategy == "fallback":
            return await self._fallback_retrieve(query)
        elif self.strategy == "round_robin":
            return await self._round_robin_retrieve(query)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    async def _merge_retrieve(self, query: RetrievalQuery) -> ContextResponse:
        """Merge results from all providers."""
        import time

        start = time.time()

        # Query all providers concurrently
        tasks = [p.retrieve(query) for p in self.providers]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results
        all_results = []
        for resp in responses:
            if isinstance(resp, ContextResponse):
                all_results.extend(resp.results)

        # Sort by score and deduplicate
        all_results.sort(key=lambda r: r.score, reverse=True)
        seen_content = set()
        unique_results = []
        for r in all_results:
            if r.content not in seen_content:
                seen_content.add(r.content)
                unique_results.append(r)
                if len(unique_results) >= query.top_k:
                    break

        elapsed = (time.time() - start) * 1000

        return ContextResponse(
            results=unique_results,
            total_results=len(all_results),
            query_time_ms=elapsed,
            provider_metadata={"mode": "composite", "strategy": "merge"},
        )

    async def _fallback_retrieve(self, query: RetrievalQuery) -> ContextResponse:
        """Try providers in order until one succeeds."""
        for provider in self.providers:
            try:
                response = await provider.retrieve(query)
                if response.results:
                    return response
            except Exception:
                continue

        return ContextResponse(results=[], total_results=0)

    async def _round_robin_retrieve(self, query: RetrievalQuery) -> ContextResponse:
        """Use providers in round-robin fashion."""
        provider = self.providers[self._current_index % len(self.providers)]
        self._current_index += 1
        return await provider.retrieve(query)

    async def health_check(self) -> bool:
        """Check if at least one provider is healthy."""
        for provider in self.providers:
            if await provider.health_check():
                return True
        return False


# Provider registry
_provider_registry: dict[str, BaseContextProvider] = {}


def register_provider(name: str, provider: BaseContextProvider) -> None:
    """Register a context provider."""
    _provider_registry[name] = provider


def get_provider(name: str) -> Optional[BaseContextProvider]:
    """Get a registered provider by name."""
    return _provider_registry.get(name)


def list_providers() -> list[str]:
    """List all registered provider names."""
    return list(_provider_registry.keys())


def create_provider(
    mode: ContextMode,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> BaseContextProvider:
    """
    Factory function to create a context provider.

    Args:
        mode: The context mode (native or proxy)
        endpoint: API endpoint for proxy mode
        api_key: API key for proxy mode
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured context provider
    """
    if mode == ContextMode.NATIVE:
        return NativeContextProvider(**kwargs)
    elif mode == ContextMode.PROXY:
        if not endpoint:
            raise ValueError("endpoint is required for proxy mode")
        return ProxyContextProvider(endpoint=endpoint, api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")
