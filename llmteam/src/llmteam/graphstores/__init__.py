"""
Graph stores package.

Provides graph storage for KAG pipeline.
"""

from llmteam.graphstores.base import BaseGraphStore, Relation, Subgraph, Path
from llmteam.graphstores.memory import InMemoryGraphStore

# Neo4j is lazy-loaded to avoid import errors when neo4j is missing
# Use: from llmteam.graphstores.neo4j import Neo4jStore

__all__ = [
    "BaseGraphStore",
    "Relation",
    "Subgraph",
    "Path",
    "InMemoryGraphStore",
]
