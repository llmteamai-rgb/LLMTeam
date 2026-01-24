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

    # Output
    output_key: Optional[str] = None  # Key for saving result
    output_format: str = "text"  # "text" | "json" | "structured"

    # Context
    use_context: bool = True  # Use context from RAG/KAG


@dataclass
class RAGAgentConfig(AgentConfig):
    """RAG agent configuration."""

    type: AgentType = AgentType.RAG
    role: str = "rag"

    # Mode
    mode: AgentMode = AgentMode.NATIVE

    # Native Mode
    vector_store: Optional[str] = None  # "chroma", "faiss", "pinecone", "qdrant"
    collection: str = "default"
    embedding_model: str = "text-embedding-3-small"

    # Proxy Mode
    proxy_endpoint: Optional[str] = None
    proxy_api_key: Optional[str] = None

    # Retrieval
    top_k: int = 5
    score_threshold: float = 0.7
    namespace: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)

    # Output
    include_sources: bool = True
    include_scores: bool = False

    # Delivery (for NOT_SHARED mode)
    deliver_to: Optional[str] = None  # Target agent ID


@dataclass
class KAGAgentConfig(AgentConfig):
    """KAG agent configuration."""

    type: AgentType = AgentType.KAG
    role: str = "kag"

    # Mode
    mode: AgentMode = AgentMode.NATIVE

    # Native Mode
    graph_store: Optional[str] = None  # "neo4j", "neptune", "memgraph"
    graph_uri: Optional[str] = None
    graph_user: Optional[str] = None
    graph_password: Optional[str] = None

    # Proxy Mode
    proxy_endpoint: Optional[str] = None
    proxy_api_key: Optional[str] = None

    # Query
    max_hops: int = 2  # Graph traversal depth
    max_entities: int = 10
    include_relations: bool = True

    # Entity Extraction
    extract_entities: bool = True  # Extract from query
    entity_types: List[str] = field(default_factory=list)

    # Delivery
    deliver_to: Optional[str] = None
