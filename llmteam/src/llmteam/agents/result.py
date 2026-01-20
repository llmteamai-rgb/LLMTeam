"""
Agent result dataclasses.

Defines output structures for all agent types.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llmteam.agents.types import AgentType


@dataclass
class AgentResult:
    """
    Result of agent execution.

    Returned from agent.process().
    """

    # Output
    output: Any  # Main result
    output_key: str = "result"  # Key for saving in context

    # Agent Info
    agent_id: str = ""
    agent_type: AgentType = AgentType.LLM

    # Status
    success: bool = True
    error: Optional[str] = None

    # Metadata
    tokens_used: int = 0
    latency_ms: int = 0
    model: Optional[str] = None  # Which model was used

    # RAG/KAG Specific
    sources: List[Dict[str, Any]] = field(default_factory=list)

    # Escalation
    should_escalate: bool = False
    escalation_reason: Optional[str] = None

    # Context Delivery (for RAG/KAG)
    context_payload: Optional[Dict[str, Any]] = None  # Data for mailbox

    def to_dict(self) -> Dict[str, Any]:
        """Serialize."""
        return {
            "output": self.output,
            "output_key": self.output_key,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "success": self.success,
            "error": self.error,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "sources": self.sources,
            "should_escalate": self.should_escalate,
            "escalation_reason": self.escalation_reason,
        }


@dataclass
class RAGResult(AgentResult):
    """RAG agent result."""

    agent_type: AgentType = AgentType.RAG

    # RAG Specific
    documents: List[Dict[str, Any]] = field(default_factory=list)
    query: str = ""
    total_found: int = 0

    def __post_init__(self):
        # Automatically form context_payload for mailbox
        if self.documents and self.context_payload is None:
            self.context_payload = {
                "_rag_context": self.documents,
                "rag_query": self.query,
                "rag_count": len(self.documents),
            }


@dataclass
class KAGResult(AgentResult):
    """KAG agent result."""

    agent_type: AgentType = AgentType.KAG

    # KAG Specific
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    query_entities: List[str] = field(default_factory=list)  # Searched entities

    def __post_init__(self):
        if self.entities and self.context_payload is None:
            self.context_payload = {
                "_kag_context": self.entities,
                "kag_relations": self.relations,
                "kag_entities": self.query_entities,
            }
