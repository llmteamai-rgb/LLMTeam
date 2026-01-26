"""
Knowledge extraction module.

Provides relation extraction for KAG pipeline.
"""

from llmteam.knowledge.extractor import RelationExtractor
from llmteam.knowledge.builder import KnowledgeGraphBuilder

__all__ = [
    "RelationExtractor",
    "KnowledgeGraphBuilder",
]
