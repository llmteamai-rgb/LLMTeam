"""
Agent configuration dataclasses.

Defines configuration structures for all agent types.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from llmteam.agents.types import AgentType, AgentMode

if TYPE_CHECKING:
    from llmteam.agents.retry import RetryPolicy, CircuitBreakerPolicy
    from llmteam.tools import ToolDefinition


@dataclass
class AgentConfig:
    """
    Base agent configuration.

    Used for creating agents via LLMTeam.add_agent(config).
    Does not contain runtime state.
    """

    # Required (but with defaults for dataclass inheritance compatibility)
    type: AgentType = field(default=AgentType.LLM)
    role: str = ""  # Unique identifier within team (required, validated)

    # Optional (common)
    id: Optional[str] = None  # Explicit ID (default: role)
    name: Optional[str] = None  # Human-readable name
    description: str = ""

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # RFC-012: Per-agent retry & circuit breaker policies
    retry_policy: Optional["RetryPolicy"] = None
    circuit_breaker: Optional["CircuitBreakerPolicy"] = None

    # RFC-013: Per-agent tools
    tools: Optional[List["ToolDefinition"]] = None

    def __post_init__(self):
        if not self.role:
            raise ValueError("AgentConfig.role is required")
        if self.id is None:
            self.id = self.role
        if self.name is None:
            self.name = self.role.replace("_", " ").title()


@dataclass
class LLMAgentConfig(AgentConfig):
    """LLM agent configuration."""

    type: AgentType = AgentType.LLM

    # LLM Settings
    prompt: str = ""  # Prompt template with {variables}
    system_prompt: Optional[str] = None  # Auto-generated if None
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1000

    # RFC-016: Tool execution loop
    max_tool_rounds: int = 5  # Max tool call loops before stopping

    # Output
    output_key: Optional[str] = None  # Key for saving result
    output_format: str = "text"  # "text" | "json" | "structured"

    # Context
    use_context: bool = True  # Use context from RAG/KAG


@dataclass
class RAGAgentConfig(AgentConfig):
    """RAG agent configuration (RFC-025)."""

    type: AgentType = AgentType.RAG
    role: str = "rag"

    # Mode
    mode: AgentMode = AgentMode.NATIVE

    # ═══════════════════════════════════════════════════════
    # SOURCES (RFC-025)
    # ═══════════════════════════════════════════════════════
    documents: Optional[List[str]] = None  # List of file paths or glob patterns
    texts: Optional[List[str]] = None  # List of raw texts
    chunks: Optional[List[str]] = None  # Pre-chunked texts
    directory: Optional[str] = None  # Directory to scan recursively
    urls: Optional[List[str]] = None  # URLs to fetch and index

    # ═══════════════════════════════════════════════════════
    # DOCUMENT LOADING
    # ═══════════════════════════════════════════════════════
    loader: str = "auto"  # "auto" | "pdf" | "docx" | "xlsx" | "html" | "text"

    # ═══════════════════════════════════════════════════════
    # CHUNKING
    # ═══════════════════════════════════════════════════════
    chunker: str = "recursive"  # "recursive" | "sentence" | "token" | "semantic" | "none"
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ═══════════════════════════════════════════════════════
    # EMBEDDINGS
    # ═══════════════════════════════════════════════════════
    embedding_provider: str = "openai"  # "openai" | "huggingface"
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: Optional[str] = None
    embedding_cache: bool = True
    embedding_batch_size: int = 100

    # ═══════════════════════════════════════════════════════
    # VECTOR STORE
    # ═══════════════════════════════════════════════════════
    vector_store: Optional[str] = None  # "auto" | "memory" | "faiss" | "qdrant"
    vector_store_path: Optional[str] = None  # Path for persistence
    collection: str = "default"

    # Qdrant-specific
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None

    # ═══════════════════════════════════════════════════════
    # RETRIEVAL
    # ═══════════════════════════════════════════════════════
    top_k: int = 5
    score_threshold: float = 0.0
    rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    search_type: str = "similarity"  # "similarity" | "mmr"
    mmr_diversity: float = 0.3
    namespace: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)

    # ═══════════════════════════════════════════════════════
    # OUTPUT
    # ═══════════════════════════════════════════════════════
    include_sources: bool = True
    include_scores: bool = True
    include_metadata: bool = True
    context_template: str = "{text}"

    # ═══════════════════════════════════════════════════════
    # PROXY MODE
    # ═══════════════════════════════════════════════════════
    proxy_endpoint: Optional[str] = None
    proxy_api_key: Optional[str] = None

    # ═══════════════════════════════════════════════════════
    # DELIVERY
    # ═══════════════════════════════════════════════════════
    deliver_to: Optional[str] = None
    context_key: str = "_rag_context"


@dataclass
class KAGAgentConfig(AgentConfig):
    """KAG agent configuration (RFC-025)."""

    type: AgentType = AgentType.KAG
    role: str = "kag"

    # Mode
    mode: AgentMode = AgentMode.NATIVE

    # ═══════════════════════════════════════════════════════
    # SOURCES
    # ═══════════════════════════════════════════════════════
    documents: Optional[List[str]] = None  # Documents to build graph from
    text: Optional[str] = None  # Raw text
    texts: Optional[List[str]] = None  # Multiple texts
    entities: Optional[List[Dict[str, Any]]] = None  # Pre-defined entities
    relations: Optional[List[Dict[str, Any]]] = None  # Pre-defined relations

    # ═══════════════════════════════════════════════════════
    # NER (Named Entity Recognition)
    # ═══════════════════════════════════════════════════════
    ner_provider: str = "llm"  # "llm" | "spacy" | "simple"
    ner_model: Optional[str] = None  # SpaCy model name
    entity_types: List[str] = field(default_factory=lambda: [
        "Person", "Organization", "Location", "Date",
        "Product", "Event", "Technology",
    ])
    extract_properties: bool = True

    # ═══════════════════════════════════════════════════════
    # RELATION EXTRACTION
    # ═══════════════════════════════════════════════════════
    relation_extractor: str = "llm"  # "llm" | "pattern"
    relation_types: List[str] = field(default_factory=lambda: [
        "WORKS_FOR", "FOUNDED", "CEO_OF", "LOCATED_IN",
        "OWNS", "PART_OF", "CREATED", "RELATED_TO",
    ])

    # ═══════════════════════════════════════════════════════
    # GRAPH STORE
    # ═══════════════════════════════════════════════════════
    graph_store: str = "memory"  # "memory" | "neo4j"
    graph_store_path: Optional[str] = None

    # Neo4j settings
    neo4j_uri: Optional[str] = None
    neo4j_user: Optional[str] = None
    neo4j_password: Optional[str] = None

    # ═══════════════════════════════════════════════════════
    # GRAPH TRAVERSAL
    # ═══════════════════════════════════════════════════════
    max_hops: int = 2
    max_entities: int = 30
    max_relations: int = 50
    include_relations: bool = True

    # ═══════════════════════════════════════════════════════
    # QUERY
    # ═══════════════════════════════════════════════════════
    query_mode: str = "hybrid"  # "entity" | "relation" | "path" | "hybrid"
    extract_query_entities: bool = True

    # ═══════════════════════════════════════════════════════
    # OUTPUT
    # ═══════════════════════════════════════════════════════
    return_subgraph: bool = True
    return_paths: bool = False
    context_template: str = "Entities: {entities}\nRelations: {relations}"

    # ═══════════════════════════════════════════════════════
    # DELIVERY
    # ═══════════════════════════════════════════════════════
    deliver_to: Optional[str] = None
    context_key: str = "_kag_context"

    # Legacy fields
    graph_uri: Optional[str] = None  # Deprecated, use neo4j_uri
    graph_user: Optional[str] = None  # Deprecated, use neo4j_user
    graph_password: Optional[str] = None  # Deprecated, use neo4j_password
    proxy_endpoint: Optional[str] = None
    proxy_api_key: Optional[str] = None
