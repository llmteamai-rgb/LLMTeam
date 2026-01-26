"""
Base graph store protocol.

Defines the interface for all graph stores.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llmteam.ner.base import Entity


@dataclass
class Relation:
    """
    Relation between entities.

    Attributes:
        source: Source entity name.
        target: Target entity name.
        type: Relation type (e.g., WORKS_FOR, FOUNDED).
        properties: Additional properties.
    """

    source: str
    target: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.source, self.target, self.type))

    def __eq__(self, other):
        if not isinstance(other, Relation):
            return False
        return (
            self.source == other.source
            and self.target == other.target
            and self.type == other.type
        )


@dataclass
class Path:
    """
    Path between entities in the graph.

    Attributes:
        entities: List of entity names in the path.
        relations: List of relation types connecting them.
        length: Number of hops in the path.
    """

    entities: List[str]
    relations: List[str]
    length: int


@dataclass
class Subgraph:
    """
    Subgraph containing entities and relations.

    Attributes:
        entities: List of entities in the subgraph.
        relations: List of relations in the subgraph.
    """

    entities: List[Entity]
    relations: List[Relation]

    def to_mermaid(self) -> str:
        """Convert subgraph to Mermaid diagram code."""
        lines = ["graph LR"]

        # Create node definitions
        entity_ids = {}
        for i, entity in enumerate(self.entities):
            node_id = f"E{i}"
            entity_ids[entity.name] = node_id
            label = f"{entity.name}\\n({entity.type})"
            lines.append(f"    {node_id}[{label}]")

        # Create edges
        for relation in self.relations:
            source_id = entity_ids.get(relation.source, relation.source)
            target_id = entity_ids.get(relation.target, relation.target)
            lines.append(f"    {source_id} -->|{relation.type}| {target_id}")

        return "\n".join(lines)


class BaseGraphStore(ABC):
    """
    Abstract base class for graph stores.

    All graph stores must implement methods for
    adding, querying, and managing entities and relations.
    """

    @abstractmethod
    async def add_entity(self, entity: Entity) -> str:
        """
        Add an entity to the graph.

        Args:
            entity: Entity to add.

        Returns:
            Entity ID.
        """
        pass

    @abstractmethod
    async def add_relation(self, relation: Relation) -> str:
        """
        Add a relation to the graph.

        Args:
            relation: Relation to add.

        Returns:
            Relation ID.
        """
        pass

    @abstractmethod
    async def get_entity(self, name: str) -> Optional[Entity]:
        """
        Get an entity by name.

        Args:
            name: Entity name.

        Returns:
            Entity if found, None otherwise.
        """
        pass

    @abstractmethod
    async def find_entities(
        self,
        name: Optional[str] = None,
        type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> List[Entity]:
        """
        Find entities by criteria.

        Args:
            name: Optional name pattern.
            type: Optional entity type.
            properties: Optional property filters.

        Returns:
            List of matching entities.
        """
        pass

    @abstractmethod
    async def find_relations(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None,
        type: Optional[str] = None,
    ) -> List[Relation]:
        """
        Find relations by criteria.

        Args:
            source: Optional source entity name.
            target: Optional target entity name.
            type: Optional relation type.

        Returns:
            List of matching relations.
        """
        pass

    @abstractmethod
    async def get_neighbors(
        self,
        entity_name: str,
        hops: int = 1,
        relation_types: Optional[List[str]] = None,
    ) -> Subgraph:
        """
        Get neighboring entities within N hops.

        Args:
            entity_name: Starting entity name.
            hops: Number of hops to traverse.
            relation_types: Optional filter for relation types.

        Returns:
            Subgraph containing neighbors.
        """
        pass

    @abstractmethod
    async def find_path(
        self,
        source: str,
        target: str,
        max_hops: int = 3,
    ) -> List[Path]:
        """
        Find paths between two entities.

        Args:
            source: Source entity name.
            target: Target entity name.
            max_hops: Maximum path length.

        Returns:
            List of paths found.
        """
        pass

    @abstractmethod
    async def delete_entity(self, name: str) -> bool:
        """
        Delete an entity and its relations.

        Args:
            name: Entity name to delete.

        Returns:
            True if deleted, False if not found.
        """
        pass

    @abstractmethod
    async def clear(self) -> int:
        """
        Clear all entities and relations.

        Returns:
            Number of items cleared.
        """
        pass

    @abstractmethod
    def entity_count(self) -> int:
        """Get number of entities."""
        pass

    @abstractmethod
    def relation_count(self) -> int:
        """Get number of relations."""
        pass


__all__ = ["BaseGraphStore", "Relation", "Subgraph", "Path"]
