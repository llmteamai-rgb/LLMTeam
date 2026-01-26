"""
In-memory graph store using NetworkX.

Provides a simple graph store for development and testing.
"""

import json
import pickle
from collections import deque
from pathlib import Path as FilePath
from typing import Any, Dict, List, Optional, Set

from llmteam.ner.base import Entity
from llmteam.graphstores.base import BaseGraphStore, Relation, Subgraph, Path


class InMemoryGraphStore(BaseGraphStore):
    """
    In-memory graph store using NetworkX.

    Suitable for development, testing, and small graphs.

    Requires:
        pip install networkx
    """

    def __init__(self):
        """Initialize the in-memory graph store."""
        self._ensure_networkx()
        import networkx as nx

        self._graph = nx.DiGraph()
        self._entities: Dict[str, Entity] = {}

    def _ensure_networkx(self) -> None:
        """Ensure NetworkX is available."""
        try:
            import networkx  # noqa: F401
        except ImportError:
            raise ImportError(
                "In-memory graph store requires 'networkx'. "
                "Install with: pip install networkx"
            )

    async def add_entity(self, entity: Entity) -> str:
        """Add an entity to the graph."""
        self._entities[entity.name] = entity
        self._graph.add_node(
            entity.name,
            type=entity.type,
            properties=entity.properties,
        )
        return entity.name

    async def add_relation(self, relation: Relation) -> str:
        """Add a relation to the graph."""
        # Ensure source and target nodes exist
        if relation.source not in self._graph:
            self._graph.add_node(relation.source)
        if relation.target not in self._graph:
            self._graph.add_node(relation.target)

        self._graph.add_edge(
            relation.source,
            relation.target,
            type=relation.type,
            properties=relation.properties,
        )
        return f"{relation.source}-{relation.type}-{relation.target}"

    async def get_entity(self, name: str) -> Optional[Entity]:
        """Get an entity by name."""
        return self._entities.get(name)

    async def find_entities(
        self,
        name: Optional[str] = None,
        type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> List[Entity]:
        """Find entities by criteria."""
        results = []

        for entity in self._entities.values():
            # Filter by name (partial match)
            if name and name.lower() not in entity.name.lower():
                continue

            # Filter by type
            if type and entity.type != type:
                continue

            # Filter by properties
            if properties:
                match = True
                for key, value in properties.items():
                    if entity.properties.get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            results.append(entity)

        return results

    async def find_relations(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None,
        type: Optional[str] = None,
    ) -> List[Relation]:
        """Find relations by criteria."""
        results = []

        for u, v, data in self._graph.edges(data=True):
            # Filter by source
            if source and u != source:
                continue

            # Filter by target
            if target and v != target:
                continue

            # Filter by type
            rel_type = data.get("type", "RELATED_TO")
            if type and rel_type != type:
                continue

            results.append(
                Relation(
                    source=u,
                    target=v,
                    type=rel_type,
                    properties=data.get("properties", {}),
                )
            )

        return results

    async def get_neighbors(
        self,
        entity_name: str,
        hops: int = 1,
        relation_types: Optional[List[str]] = None,
    ) -> Subgraph:
        """Get neighboring entities within N hops."""
        if entity_name not in self._graph:
            return Subgraph(entities=[], relations=[])

        visited_nodes: Set[str] = {entity_name}
        visited_edges: Set[tuple] = set()
        queue = deque([(entity_name, 0)])

        while queue:
            current, depth = queue.popleft()

            if depth >= hops:
                continue

            # Get outgoing edges
            for neighbor in self._graph.successors(current):
                edge_data = self._graph.get_edge_data(current, neighbor)
                rel_type = edge_data.get("type", "RELATED_TO")

                # Filter by relation types
                if relation_types and rel_type not in relation_types:
                    continue

                visited_edges.add((current, neighbor, rel_type))
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    queue.append((neighbor, depth + 1))

            # Get incoming edges
            for neighbor in self._graph.predecessors(current):
                edge_data = self._graph.get_edge_data(neighbor, current)
                rel_type = edge_data.get("type", "RELATED_TO")

                if relation_types and rel_type not in relation_types:
                    continue

                visited_edges.add((neighbor, current, rel_type))
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    queue.append((neighbor, depth + 1))

        # Build subgraph
        entities = []
        for node_name in visited_nodes:
            if node_name in self._entities:
                entities.append(self._entities[node_name])
            else:
                # Node exists but no entity data
                node_data = self._graph.nodes.get(node_name, {})
                entities.append(
                    Entity(
                        name=node_name,
                        type=node_data.get("type", "Unknown"),
                        properties=node_data.get("properties", {}),
                    )
                )

        relations = []
        for source, target, rel_type in visited_edges:
            edge_data = self._graph.get_edge_data(source, target)
            relations.append(
                Relation(
                    source=source,
                    target=target,
                    type=rel_type,
                    properties=edge_data.get("properties", {}),
                )
            )

        return Subgraph(entities=entities, relations=relations)

    async def find_path(
        self,
        source: str,
        target: str,
        max_hops: int = 3,
    ) -> List[Path]:
        """Find paths between two entities."""
        import networkx as nx

        if source not in self._graph or target not in self._graph:
            return []

        paths = []
        try:
            # Find all simple paths up to max_hops length
            for path_nodes in nx.all_simple_paths(
                self._graph, source, target, cutoff=max_hops
            ):
                # Extract relation types along the path
                relations = []
                for i in range(len(path_nodes) - 1):
                    edge_data = self._graph.get_edge_data(
                        path_nodes[i], path_nodes[i + 1]
                    )
                    relations.append(edge_data.get("type", "RELATED_TO"))

                paths.append(
                    Path(
                        entities=path_nodes,
                        relations=relations,
                        length=len(path_nodes) - 1,
                    )
                )
        except nx.NetworkXNoPath:
            pass

        return paths

    async def delete_entity(self, name: str) -> bool:
        """Delete an entity and its relations."""
        if name not in self._entities:
            return False

        del self._entities[name]
        if name in self._graph:
            self._graph.remove_node(name)
        return True

    async def clear(self) -> int:
        """Clear all entities and relations."""
        count = len(self._entities) + self._graph.number_of_edges()
        self._entities.clear()
        self._graph.clear()
        return count

    def entity_count(self) -> int:
        """Get number of entities."""
        return len(self._entities)

    def relation_count(self) -> int:
        """Get number of relations."""
        return self._graph.number_of_edges()

    def save(self, path: str) -> None:
        """Save the graph to disk."""
        import networkx as nx

        path = FilePath(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save graph structure
        nx.write_gml(self._graph, f"{path}.gml")

        # Save entities
        entities_data = {
            name: {
                "name": e.name,
                "type": e.type,
                "properties": e.properties,
            }
            for name, e in self._entities.items()
        }
        with open(f"{path}.entities.json", "w") as f:
            json.dump(entities_data, f)

    def load(self, path: str) -> None:
        """Load the graph from disk."""
        import networkx as nx

        # Load graph structure
        self._graph = nx.read_gml(f"{path}.gml")

        # Load entities
        with open(f"{path}.entities.json", "r") as f:
            entities_data = json.load(f)

        self._entities = {
            name: Entity(
                name=data["name"],
                type=data["type"],
                properties=data.get("properties", {}),
            )
            for name, data in entities_data.items()
        }

    def export_graphml(self, path: str) -> None:
        """Export graph in GraphML format."""
        import networkx as nx

        nx.write_graphml(self._graph, path)

    def export_cypher(self) -> str:
        """Export as Cypher queries for Neo4j."""
        lines = []

        # Create nodes
        for name, entity in self._entities.items():
            props = json.dumps(entity.properties) if entity.properties else "{}"
            lines.append(
                f"CREATE (:{entity.type} {{name: '{name}', properties: {props}}})"
            )

        # Create relationships
        for u, v, data in self._graph.edges(data=True):
            rel_type = data.get("type", "RELATED_TO")
            props = json.dumps(data.get("properties", {}))
            lines.append(
                f"MATCH (a {{name: '{u}'}}), (b {{name: '{v}'}}) "
                f"CREATE (a)-[:{rel_type} {{properties: {props}}}]->(b)"
            )

        return ";\n".join(lines) + ";"


__all__ = ["InMemoryGraphStore"]
