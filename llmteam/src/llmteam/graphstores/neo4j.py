"""
Neo4j graph store.

Provides a production-ready graph store using Neo4j.
"""

from typing import Any, Dict, List, Optional

from llmteam.ner.base import Entity
from llmteam.graphstores.base import BaseGraphStore, Relation, Subgraph, Path


class Neo4jStore(BaseGraphStore):
    """
    Neo4j graph store for production deployments.

    Requires:
        pip install neo4j
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "",
        database: str = "neo4j",
    ):
        """
        Initialize the Neo4j store.

        Args:
            uri: Neo4j connection URI.
            user: Username.
            password: Password.
            database: Database name.
        """
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._driver = None

        self._ensure_driver()

    def _ensure_driver(self) -> None:
        """Ensure Neo4j driver is initialized."""
        if self._driver is not None:
            return

        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError(
                "Neo4j store requires 'neo4j'. Install with: pip install neo4j"
            )

        self._driver = GraphDatabase.driver(
            self._uri, auth=(self._user, self._password)
        )

    def _run_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict]:
        """Run a Cypher query and return results."""
        with self._driver.session(database=self._database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    async def add_entity(self, entity: Entity) -> str:
        """Add an entity to the graph."""
        query = f"""
        MERGE (e:{entity.type} {{name: $name}})
        SET e.properties = $properties
        RETURN e.name as name
        """
        results = self._run_query(
            query, {"name": entity.name, "properties": entity.properties}
        )
        return results[0]["name"] if results else entity.name

    async def add_relation(self, relation: Relation) -> str:
        """Add a relation to the graph."""
        query = f"""
        MATCH (a {{name: $source}}), (b {{name: $target}})
        MERGE (a)-[r:{relation.type}]->(b)
        SET r.properties = $properties
        RETURN type(r) as type
        """
        self._run_query(
            query,
            {
                "source": relation.source,
                "target": relation.target,
                "properties": relation.properties,
            },
        )
        return f"{relation.source}-{relation.type}-{relation.target}"

    async def get_entity(self, name: str) -> Optional[Entity]:
        """Get an entity by name."""
        query = """
        MATCH (e {name: $name})
        RETURN e.name as name, labels(e)[0] as type, e.properties as properties
        """
        results = self._run_query(query, {"name": name})
        if results:
            r = results[0]
            return Entity(
                name=r["name"],
                type=r.get("type", "Unknown"),
                properties=r.get("properties", {}),
            )
        return None

    async def find_entities(
        self,
        name: Optional[str] = None,
        type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> List[Entity]:
        """Find entities by criteria."""
        conditions = []
        params = {}

        if name:
            conditions.append("e.name CONTAINS $name")
            params["name"] = name

        if type:
            conditions.append(f"e:{type}")

        where_clause = " AND ".join(conditions) if conditions else "true"

        query = f"""
        MATCH (e)
        WHERE {where_clause}
        RETURN e.name as name, labels(e)[0] as type, e.properties as properties
        """

        results = self._run_query(query, params)
        return [
            Entity(
                name=r["name"],
                type=r.get("type", "Unknown"),
                properties=r.get("properties", {}),
            )
            for r in results
        ]

    async def find_relations(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None,
        type: Optional[str] = None,
    ) -> List[Relation]:
        """Find relations by criteria."""
        conditions = []
        params = {}

        if source:
            conditions.append("a.name = $source")
            params["source"] = source

        if target:
            conditions.append("b.name = $target")
            params["target"] = target

        where_clause = " AND ".join(conditions) if conditions else "true"
        rel_pattern = f"[r:{type}]" if type else "[r]"

        query = f"""
        MATCH (a)-{rel_pattern}->(b)
        WHERE {where_clause}
        RETURN a.name as source, b.name as target, type(r) as type, r.properties as properties
        """

        results = self._run_query(query, params)
        return [
            Relation(
                source=r["source"],
                target=r["target"],
                type=r["type"],
                properties=r.get("properties", {}),
            )
            for r in results
        ]

    async def get_neighbors(
        self,
        entity_name: str,
        hops: int = 1,
        relation_types: Optional[List[str]] = None,
    ) -> Subgraph:
        """Get neighboring entities within N hops."""
        rel_filter = "|".join(relation_types) if relation_types else ""
        rel_pattern = f"[*1..{hops}]" if not rel_filter else f"[:{rel_filter}*1..{hops}]"

        # Get nodes
        node_query = f"""
        MATCH (start {{name: $name}})-{rel_pattern}-(neighbor)
        RETURN DISTINCT neighbor.name as name, labels(neighbor)[0] as type,
               neighbor.properties as properties
        """
        node_results = self._run_query(node_query, {"name": entity_name})

        # Get relations
        rel_query = f"""
        MATCH (start {{name: $name}})-{rel_pattern}-(neighbor)
        MATCH (a)-[r]->(b)
        WHERE (a.name = start.name OR a.name IN [n.name for n in collect(neighbor)])
          AND (b.name = start.name OR b.name IN [n.name for n in collect(neighbor)])
        RETURN DISTINCT a.name as source, b.name as target, type(r) as type,
               r.properties as properties
        """
        rel_results = self._run_query(rel_query, {"name": entity_name})

        entities = [
            Entity(
                name=r["name"],
                type=r.get("type", "Unknown"),
                properties=r.get("properties", {}),
            )
            for r in node_results
        ]

        relations = [
            Relation(
                source=r["source"],
                target=r["target"],
                type=r["type"],
                properties=r.get("properties", {}),
            )
            for r in rel_results
        ]

        return Subgraph(entities=entities, relations=relations)

    async def find_path(
        self,
        source: str,
        target: str,
        max_hops: int = 3,
    ) -> List[Path]:
        """Find paths between two entities."""
        query = f"""
        MATCH path = shortestPath((a {{name: $source}})-[*1..{max_hops}]-(b {{name: $target}}))
        RETURN [n IN nodes(path) | n.name] as entities,
               [r IN relationships(path) | type(r)] as relations
        """
        results = self._run_query(query, {"source": source, "target": target})

        return [
            Path(
                entities=r["entities"],
                relations=r["relations"],
                length=len(r["relations"]),
            )
            for r in results
        ]

    async def delete_entity(self, name: str) -> bool:
        """Delete an entity and its relations."""
        query = """
        MATCH (e {name: $name})
        DETACH DELETE e
        RETURN count(e) as deleted
        """
        results = self._run_query(query, {"name": name})
        return results[0]["deleted"] > 0 if results else False

    async def clear(self) -> int:
        """Clear all entities and relations."""
        count_query = "MATCH (n) RETURN count(n) as count"
        count_results = self._run_query(count_query)
        count = count_results[0]["count"] if count_results else 0

        self._run_query("MATCH (n) DETACH DELETE n")
        return count

    def entity_count(self) -> int:
        """Get number of entities."""
        results = self._run_query("MATCH (n) RETURN count(n) as count")
        return results[0]["count"] if results else 0

    def relation_count(self) -> int:
        """Get number of relations."""
        results = self._run_query("MATCH ()-[r]->() RETURN count(r) as count")
        return results[0]["count"] if results else 0

    def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver:
            self._driver.close()
            self._driver = None


__all__ = ["Neo4jStore"]
