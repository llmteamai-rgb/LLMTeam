"""
Knowledge graph builder.

Builds knowledge graphs from text using NER and relation extraction.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from llmteam.ner.base import Entity
from llmteam.graphstores.base import Relation, Subgraph

if TYPE_CHECKING:
    from llmteam.ner.base import BaseNER
    from llmteam.graphstores.base import BaseGraphStore
    from llmteam.knowledge.extractor import RelationExtractor


class KnowledgeGraphBuilder:
    """
    Knowledge graph builder.

    Combines NER and relation extraction to build knowledge graphs.
    """

    def __init__(
        self,
        ner: "BaseNER",
        relation_extractor: "RelationExtractor",
        graph_store: "BaseGraphStore",
    ):
        """
        Initialize the builder.

        Args:
            ner: NER provider for entity extraction.
            relation_extractor: Relation extractor.
            graph_store: Graph store for persistence.
        """
        self._ner = ner
        self._relation_extractor = relation_extractor
        self._graph_store = graph_store

    async def build_from_text(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
    ) -> "BuildResult":
        """
        Build knowledge graph from text.

        Args:
            text: Text to process.
            entity_types: Optional entity types to extract.
            relation_types: Optional relation types to extract.

        Returns:
            BuildResult with entities and relations.
        """
        # Extract entities
        entities = await self._ner.extract(text, entity_types)

        # Extract relations
        relations = []
        if len(entities) >= 2:
            relations = await self._relation_extractor.extract(
                text, entities, relation_types
            )

        # Add to graph store
        for entity in entities:
            await self._graph_store.add_entity(entity)

        for relation in relations:
            await self._graph_store.add_relation(relation)

        return BuildResult(
            entities=entities,
            relations=relations,
            entity_count=len(entities),
            relation_count=len(relations),
        )

    async def build_from_texts(
        self,
        texts: List[str],
        entity_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
    ) -> "BuildResult":
        """
        Build knowledge graph from multiple texts.

        Args:
            texts: List of texts to process.
            entity_types: Optional entity types to extract.
            relation_types: Optional relation types to extract.

        Returns:
            Combined BuildResult.
        """
        all_entities: List[Entity] = []
        all_relations: List[Relation] = []

        for text in texts:
            result = await self.build_from_text(text, entity_types, relation_types)
            all_entities.extend(result.entities)
            all_relations.extend(result.relations)

        # Deduplicate entities
        unique_entities = list({e.name: e for e in all_entities}.values())

        return BuildResult(
            entities=unique_entities,
            relations=all_relations,
            entity_count=len(unique_entities),
            relation_count=len(all_relations),
        )

    def get_subgraph(self) -> Subgraph:
        """Get the current graph as a subgraph."""
        # This would need to be async in a full implementation
        # For now, return empty subgraph
        return Subgraph(entities=[], relations=[])


class BuildResult:
    """Result of knowledge graph building."""

    def __init__(
        self,
        entities: List[Entity],
        relations: List[Relation],
        entity_count: int,
        relation_count: int,
    ):
        """
        Initialize the build result.

        Args:
            entities: Extracted entities.
            relations: Extracted relations.
            entity_count: Total entity count.
            relation_count: Total relation count.
        """
        self.entities = entities
        self.relations = relations
        self.entity_count = entity_count
        self.relation_count = relation_count

    def to_subgraph(self) -> Subgraph:
        """Convert to Subgraph."""
        return Subgraph(entities=self.entities, relations=self.relations)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entities": [
                {"name": e.name, "type": e.type, "properties": e.properties}
                for e in self.entities
            ],
            "relations": [
                {
                    "source": r.source,
                    "target": r.target,
                    "type": r.type,
                    "properties": r.properties,
                }
                for r in self.relations
            ],
            "entity_count": self.entity_count,
            "relation_count": self.relation_count,
        }


__all__ = ["KnowledgeGraphBuilder", "BuildResult"]
