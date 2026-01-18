"""
RAG Handler for Retrieval-Augmented Generation.

Supports both native (local) and proxy (external API) modes for
context retrieval, enabling enterprise RAG/KAG integration.
"""

from typing import Any, Optional

from llmteam.runtime import StepContext
from llmteam.observability import get_logger
from llmteam.context.provider import (
    ContextProvider,
    ContextMode,
    RetrievalQuery,
    NativeContextProvider,
    ProxyContextProvider,
    create_provider,
)

logger = get_logger(__name__)


class RAGHandler:
    """
    RAG (Retrieval-Augmented Generation) Handler.

    Configuration:
        mode (str): "native" or "proxy" - retrieval mode
        context_ref (str): Reference to registered context provider (optional)

        # For proxy mode:
        proxy_endpoint (str): External RAG API endpoint
        proxy_api_key (str): API key for proxy endpoint

        # Retrieval settings:
        top_k (int): Number of context chunks to retrieve (default: 5)
        namespace (str): Optional namespace/collection to search
        filters (dict): Additional filters for retrieval

        # LLM settings:
        llm_ref (str): Reference to LLM provider
        prompt_template (str): Template with {context} and {query} placeholders
        system_prompt (str): System prompt for LLM

        # Output settings:
        include_sources (bool): Include source references in output
        include_context (bool): Include retrieved context in output

    Input:
        query (str): The user query/question
        filters (dict): Optional runtime filters

    Output:
        response (str): LLM-generated response
        context (list): Retrieved context chunks (if include_context=True)
        sources (list): Source references (if include_sources=True)

    Usage:
        {
            "step_id": "rag_query",
            "type": "rag",
            "config": {
                "mode": "proxy",
                "proxy_endpoint": "https://rag.korpos.internal/api/v1",
                "proxy_api_key": "{secrets.rag_api_key}",
                "top_k": 3,
                "llm_ref": "gpt4",
                "prompt_template": "Context:\\n{context}\\n\\nQuestion: {query}\\n\\nAnswer:",
                "include_sources": true
            }
        }
    """

    STEP_TYPE = "rag"
    DISPLAY_NAME = "RAG Query"
    DESCRIPTION = "Retrieval-Augmented Generation with native or proxy mode"
    CATEGORY = "ai"

    def __init__(
        self,
        default_provider: Optional[ContextProvider] = None,
        llm_registry: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize handler.

        Args:
            default_provider: Default context provider
            llm_registry: Registry of LLM providers
        """
        self._default_provider = default_provider
        self._llm_registry = llm_registry or {}
        self._provider_cache: dict[str, ContextProvider] = {}

    def register_llm(self, name: str, provider: Any) -> None:
        """Register an LLM provider."""
        self._llm_registry[name] = provider

    def register_context_provider(self, name: str, provider: ContextProvider) -> None:
        """Register a context provider."""
        self._provider_cache[name] = provider

    async def __call__(
        self,
        ctx: StepContext,
        config: dict[str, Any],
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute RAG query."""
        # Extract config
        mode = ContextMode(config.get("mode", "native"))
        context_ref = config.get("context_ref")
        top_k = config.get("top_k", 5)
        namespace = config.get("namespace")
        filters = config.get("filters", {})
        llm_ref = config.get("llm_ref")
        prompt_template = config.get(
            "prompt_template",
            "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
        )
        system_prompt = config.get("system_prompt", "You are a helpful assistant.")
        include_sources = config.get("include_sources", False)
        include_context = config.get("include_context", False)

        # Get query from input
        query = input_data.get("query", "")
        runtime_filters = input_data.get("filters", {})

        if not query:
            return {
                "error": {"type": "ValidationError", "message": "Query is required"}
            }

        # Merge filters
        merged_filters = {**filters, **runtime_filters}

        # Get context provider
        provider = self._resolve_provider(ctx, config, mode, context_ref)

        logger.info(
            f"RAG query: mode={mode.value}, top_k={top_k}",
            extra={"step_id": ctx.step_id, "query_length": len(query)},
        )

        # Retrieve context
        try:
            retrieval_query = RetrievalQuery(
                query=query,
                top_k=top_k,
                filters=merged_filters,
                namespace=namespace,
            )
            context_response = await provider.retrieve(retrieval_query)
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return {
                "error": {
                    "type": "RetrievalError",
                    "message": f"Failed to retrieve context: {e}",
                }
            }

        # Build context string
        context_str = context_response.to_context_string()

        # Build prompt
        prompt = prompt_template.format(context=context_str, query=query)

        # Get LLM response
        response_text = ""
        if llm_ref:
            llm_provider = self._resolve_llm(ctx, llm_ref)
            if llm_provider:
                try:
                    response_text = await self._call_llm(
                        llm_provider, prompt, system_prompt
                    )
                except Exception as e:
                    logger.error(f"LLM call failed: {e}")
                    return {
                        "error": {
                            "type": "LLMError",
                            "message": f"LLM call failed: {e}",
                        }
                    }
            else:
                logger.warning(f"LLM provider not found: {llm_ref}")
                # Return context without LLM processing
                response_text = f"[No LLM configured] Context retrieved:\n{context_str}"

        # Build output
        output: dict[str, Any] = {
            "output": {
                "response": response_text,
                "query": query,
                "retrieval_time_ms": context_response.query_time_ms,
            }
        }

        if include_context:
            output["output"]["context"] = [r.to_dict() for r in context_response.results]

        if include_sources:
            sources = []
            for r in context_response.results:
                if r.source:
                    sources.append({
                        "source": r.source,
                        "chunk_id": r.chunk_id,
                        "score": r.score,
                    })
            output["output"]["sources"] = sources

        return output

    def _resolve_provider(
        self,
        ctx: StepContext,
        config: dict[str, Any],
        mode: ContextMode,
        context_ref: Optional[str],
    ) -> ContextProvider:
        """Resolve context provider from config or registry."""
        # Check cached providers
        if context_ref and context_ref in self._provider_cache:
            return self._provider_cache[context_ref]

        # Check runtime context for registered providers
        if context_ref and hasattr(ctx.runtime, "context_providers"):
            registry = getattr(ctx.runtime, "context_providers", {})
            if context_ref in registry:
                return registry[context_ref]

        # Check default provider
        if self._default_provider:
            return self._default_provider

        # Create provider based on mode
        if mode == ContextMode.PROXY:
            endpoint = config.get("proxy_endpoint")
            api_key = config.get("proxy_api_key")
            if not endpoint:
                raise ValueError("proxy_endpoint is required for proxy mode")
            return ProxyContextProvider(endpoint=endpoint, api_key=api_key)
        else:
            return NativeContextProvider()

    def _resolve_llm(self, ctx: StepContext, llm_ref: str) -> Optional[Any]:
        """Resolve LLM provider from registry or runtime."""
        # Check local registry
        if llm_ref in self._llm_registry:
            return self._llm_registry[llm_ref]

        # Check runtime context
        if hasattr(ctx.runtime, "llm_providers"):
            registry = getattr(ctx.runtime, "llm_providers", {})
            if llm_ref in registry:
                return registry[llm_ref]

        # Try to get from runtime stores
        if hasattr(ctx.runtime, "get_store"):
            return ctx.runtime.get_store(f"llm:{llm_ref}")

        return None

    async def _call_llm(
        self, provider: Any, prompt: str, system_prompt: str
    ) -> str:
        """Call LLM provider."""
        # Support different provider interfaces
        if hasattr(provider, "complete"):
            # LLMTeam provider interface
            response = await provider.complete(
                prompt, system_prompt=system_prompt
            )
            if isinstance(response, dict):
                return response.get("content", response.get("text", str(response)))
            return str(response)
        elif hasattr(provider, "generate"):
            # Alternative interface
            response = await provider.generate(prompt)
            return str(response)
        elif hasattr(provider, "__call__"):
            # Callable interface
            response = await provider(prompt)
            return str(response)
        else:
            raise ValueError(f"Unknown LLM provider interface: {type(provider)}")


class RAGQueryBuilder:
    """
    Builder for RAG queries with fluent interface.

    Usage:
        query = (
            RAGQueryBuilder()
            .with_query("What is the refund policy?")
            .with_top_k(3)
            .with_filter("category", "policies")
            .with_namespace("customer_docs")
            .build()
        )
    """

    def __init__(self):
        self._query = ""
        self._top_k = 5
        self._filters: dict[str, Any] = {}
        self._namespace: Optional[str] = None
        self._include_metadata = True

    def with_query(self, query: str) -> "RAGQueryBuilder":
        """Set the query."""
        self._query = query
        return self

    def with_top_k(self, top_k: int) -> "RAGQueryBuilder":
        """Set number of results."""
        self._top_k = top_k
        return self

    def with_filter(self, key: str, value: Any) -> "RAGQueryBuilder":
        """Add a filter."""
        self._filters[key] = value
        return self

    def with_filters(self, filters: dict[str, Any]) -> "RAGQueryBuilder":
        """Set multiple filters."""
        self._filters.update(filters)
        return self

    def with_namespace(self, namespace: str) -> "RAGQueryBuilder":
        """Set namespace."""
        self._namespace = namespace
        return self

    def without_metadata(self) -> "RAGQueryBuilder":
        """Exclude metadata from results."""
        self._include_metadata = False
        return self

    def build(self) -> RetrievalQuery:
        """Build the retrieval query."""
        return RetrievalQuery(
            query=self._query,
            top_k=self._top_k,
            filters=self._filters,
            namespace=self._namespace,
            include_metadata=self._include_metadata,
        )
