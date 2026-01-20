"""
Agent type definitions.

Enums for agent types, modes, and statuses.
"""

from enum import Enum


class AgentType(Enum):
    """Type of agent."""

    LLM = "llm"  # Text generation via LLM
    RAG = "rag"  # Retrieval from vector store
    KAG = "kag"  # Retrieval from knowledge graph


class AgentMode(Enum):
    """Operating mode for RAG/KAG agents."""

    NATIVE = "native"  # Direct database connection
    PROXY = "proxy"  # Via external API


class AgentStatus(Enum):
    """Agent execution status."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING = "waiting"  # Waiting for input (HITL)
