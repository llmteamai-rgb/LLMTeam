"""
KAG Agent implementation.

Knowledge Graph agent with "out of the box" support (RFC-025).
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from llmteam.agents.types import AgentType, AgentMode
from llmteam.agents.config import KAGAgentConfig
from llmteam.agents.result import KAGResult
from llmteam.agents.base import BaseAgent

if TYPE_CHECKING:
    from llmteam.team import LLMTeam
    from llmteam.ner.base import BaseNER, Entity
    from llmteam.graphstores.base import BaseGraphStore, Relation, Subgraph, Path
    from llmteam.knowledge.extractor import RelationExtractor


class KAGAgent(BaseAgent):
    """
    Knowledge Graph agent with "out of the box" support.

    RFC-025: Can be used with just text, no external setup needed:
        kag = KAGAgent(text="Tesla was founded by...")
        result = await kag.query("How is Elon Musk related to Tesla?")

    Features:
    - Automatic NER (LLM, SpaCy, or regex-based)
    - Automatic relation extraction
    - In-memory graph store (or Neo4j for production)
    - Path finding and graph traversal
    """

    agent_type = AgentType.KAG

    # Config fields
    mode: AgentMode
    documents: Optional[List[str]]
    text: Optional[str]
    texts: Optional[List[str]]
    entities_config: Optional[List[Dict[str, Any]]]
    relations_config: Optional[List[Dict[str, Any]]]
    ner_provider: str
    ner_model: Optional[str]
    entity_types: List[str]
    extract_properties: bool
    relation_extractor_type: str
    relation_types: List[str]
    graph_store_type: str
    graph_store_path: Optional[str]
    neo4j_uri: Optional[str]
    neo4j_user: Optional[str]
    neo4j_password: Optional[str]
    max_hops: int
    max_entities: int
    max_relations: int
    include_relations: bool
    query_mode: str
    extract_query_entities: bool
    return_subgraph: bool
    return_paths: bool
    context_template: str
    deliver_to: Optional[str]
    context_key: str

    # Internal state
    _initialized: bool
    _ner: Optional["BaseNER"]
    _relation_extractor: Optional["RelationExtractor"]
    _graph_store: Optional["BaseGraphStore"]

    def __init__(self, team: "LLMTeam", config: KAGAgentConfig):
        super().__init__(team, config)

        self.mode = config.mode
        self.documents = config.documents
        self.text = config.text
        self.texts = config.texts
        self.entities_config = config.entities
        self.relations_config = config.relations
        self.ner_provider = config.ner_provider
        self.ner_model = config.ner_model
        self.entity_types = config.entity_types
        self.extract_properties = config.extract_properties
        self.relation_extractor_type = config.relation_extractor
        self.relation_types = config.relation_types
        self.graph_store_type = config.graph_store
        self.graph_store_path = config.graph_store_path
        self.neo4j_uri = config.neo4j_uri or config.graph_uri
        self.neo4j_user = config.neo4j_user or config.graph_user
        self.neo4j_password = config.neo4j_password or config.graph_password
        self.max_hops = config.max_hops
        self.max_entities = config.max_entities
        self.max_relations = config.max_relations
        self.include_relations = config.include_relations
        self.query_mode = config.query_mode
        self.extract_query_entities = config.extract_query_entities
        self.return_subgraph = config.return_subgraph
        self.return_paths = config.return_paths
        self.context_template = config.context_template
        self.deliver_to = config.deliver_to
        self.context_key = config.context_key

        # Internal state
        self._initialized = False
        self._ner = None
        self._relation_extractor = None
        self._graph_store = None

    def _uses_builtin_store(self) -> bool:
        """Check if this KAGAgent uses the built-in store."""
        return bool(
            self.documents or self.text or self.texts
            or self.entities_config or self.relations_config
        )

    async def initialize(self) -> None:
        """
        Initialize the KAG agent with documents/text.

        This method is called automatically on first query() call.
        """
        if self._initialized:
            return

        if not self._uses_builtin_store():
            self._initialized = True
            return

        # Create NER provider
        self._ner = self._create_ner()

        # Create relation extractor
        self._relation_extractor = self._create_relation_extractor()

        # Create graph store
        self._graph_store = self._create_graph_store()

        # Build graph from sources
        await self._build_graph()

        self._initialized = True

    def _create_ner(self) -> "BaseNER":
        """Create NER provider based on config."""
        if self.ner_provider == "spacy":
            from llmteam.ner.spacy import SpacyNER
            return SpacyNER(model=self.ner_model or "en_core_web_sm")
        elif self.ner_provider == "simple":
            from llmteam.ner import SimpleNER
            return SimpleNER()
        else:  # Default to LLM
            from llmteam.ner import LLMNER
            return LLMNER(entity_types=self.entity_types)

    def _create_relation_extractor(self) -> "RelationExtractor":
        """Create relation extractor."""
        from llmteam.knowledge import RelationExtractor
        return RelationExtractor(relation_types=self.relation_types)

    def _create_graph_store(self) -> "BaseGraphStore":
        """Create graph store based on config."""
        if self.graph_store_type == "neo4j" and self.neo4j_uri:
            from llmteam.graphstores.neo4j import Neo4jStore
            return Neo4jStore(
                uri=self.neo4j_uri,
                user=self.neo4j_user or "neo4j",
                password=self.neo4j_password or "",
            )
        else:  # Default to memory
            from llmteam.graphstores import InMemoryGraphStore
            return InMemoryGraphStore()

    async def _build_graph(self) -> None:
        """Build the knowledge graph from sources."""
        if not self._ner or not self._relation_extractor or not self._graph_store:
            return

        from llmteam.ner.base import Entity
        from llmteam.graphstores.base import Relation

        # Collect all texts to process
        all_texts: List[str] = []

        if self.text:
            all_texts.append(self.text)

        if self.texts:
            all_texts.extend(self.texts)

        if self.documents:
            from llmteam.documents import AutoLoader
            loader = AutoLoader()
            for doc_path in self.documents:
                try:
                    docs = loader.load(doc_path)
                    for doc in docs:
                        all_texts.append(doc.content)
                except Exception as e:
                    import warnings
                    warnings.warn(f"Failed to load document {doc_path}: {e}")

        # Process texts to extract entities and relations
        for text in all_texts:
            # Extract entities
            entities = await self._ner.extract(text, self.entity_types)

            # Add entities to graph
            for entity in entities:
                await self._graph_store.add_entity(entity)

            # Extract relations
            if len(entities) >= 2:
                relations = await self._relation_extractor.extract(
                    text, entities, self.relation_types
                )
                for relation in relations:
                    await self._graph_store.add_relation(relation)

        # Add pre-defined entities
        if self.entities_config:
            for e_data in self.entities_config:
                entity = Entity(
                    name=e_data.get("name", ""),
                    type=e_data.get("type", "Entity"),
                    properties=e_data.get("properties", {}),
                )
                await self._graph_store.add_entity(entity)

        # Add pre-defined relations
        if self.relations_config:
            for r_data in self.relations_config:
                relation = Relation(
                    source=r_data.get("source", ""),
                    target=r_data.get("target", ""),
                    type=r_data.get("type", "RELATED_TO"),
                    properties=r_data.get("properties", {}),
                )
                await self._graph_store.add_relation(relation)

    async def query(
        self,
        question: str,
        mode: Optional[str] = None,
        max_hops: Optional[int] = None,
    ) -> KAGResult:
        """
        Query the knowledge graph.

        Args:
            question: The query string.
            mode: Query mode override.
            max_hops: Max hops override.

        Returns:
            KAGResult with entities, relations, and paths.
        """
        # Lazy initialization
        if not self._initialized:
            await self.initialize()

        query_mode = mode or self.query_mode
        hops = max_hops or self.max_hops

        # Extract query entities if enabled
        query_entities: List[str] = []
        if self.extract_query_entities and self._ner:
            entities = await self._ner.extract(question, self.entity_types)
            query_entities = [e.name for e in entities]

        # Use built-in store if available
        if self._uses_builtin_store() and self._graph_store:
            return await self._search_builtin(question, query_entities, query_mode, hops)
        else:
            # Delegate to _execute for external stores
            return await self._execute(
                {"query": question, "entities": query_entities},
                {"mode": query_mode, "max_hops": hops},
            )

    async def _search_builtin(
        self,
        query: str,
        query_entities: List[str],
        mode: str,
        max_hops: int,
    ) -> KAGResult:
        """Search using the built-in graph store."""
        if not self._graph_store:
            return KAGResult(
                output={},
                entities=[],
                relations=[],
                query_entities=query_entities,
                success=False,
                error="Graph store not initialized",
            )

        from llmteam.graphstores.base import Subgraph

        all_entities = []
        all_relations = []
        all_paths = []

        # Find matching entities
        for entity_name in query_entities:
            # Try exact match first
            entity = await self._graph_store.get_entity(entity_name)
            if entity:
                all_entities.append(entity)

            # Try fuzzy match
            matches = await self._graph_store.find_entities(name=entity_name)
            for match in matches[:self.max_entities]:
                if match not in all_entities:
                    all_entities.append(match)

        # Get neighbors based on mode
        if mode in ("entity", "hybrid") and query_entities:
            for entity_name in query_entities[:3]:  # Limit to first 3
                subgraph = await self._graph_store.get_neighbors(
                    entity_name, hops=max_hops
                )
                for e in subgraph.entities:
                    if e not in all_entities and len(all_entities) < self.max_entities:
                        all_entities.append(e)
                for r in subgraph.relations:
                    if r not in all_relations and len(all_relations) < self.max_relations:
                        all_relations.append(r)

        # Find paths between entities if requested
        if mode in ("path", "hybrid") and len(query_entities) >= 2:
            for i, source in enumerate(query_entities[:3]):
                for target in query_entities[i+1:4]:
                    paths = await self._graph_store.find_path(source, target, max_hops)
                    all_paths.extend(paths[:3])  # Limit paths

        # Get all relations for found entities
        if self.include_relations and mode != "path":
            for entity in all_entities:
                relations = await self._graph_store.find_relations(source=entity.name)
                for r in relations:
                    if r not in all_relations and len(all_relations) < self.max_relations:
                        all_relations.append(r)

        # Convert to dict format
        entities_dicts = [
            {"name": e.name, "type": e.type, "properties": e.properties}
            for e in all_entities
        ]
        relations_dicts = [
            {"source": r.source, "target": r.target, "type": r.type, "properties": r.properties}
            for r in all_relations
        ]
        paths_dicts = [
            {"entities": p.entities, "relations": p.relations, "length": p.length}
            for p in all_paths
        ]

        # Build context
        context = self._format_context(entities_dicts, relations_dicts)

        # Build subgraph if requested
        subgraph = None
        if self.return_subgraph:
            subgraph = Subgraph(entities=all_entities, relations=all_relations)

        return KAGResult(
            output={"entities": entities_dicts, "relations": relations_dicts},
            entities=entities_dicts,
            relations=relations_dicts,
            paths=paths_dicts if self.return_paths else [],
            query_entities=query_entities,
            subgraph=subgraph,
            context=context,
            success=True,
        )

    def _format_context(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
    ) -> str:
        """Format entities and relations as context string."""
        entities_str = ", ".join([f"{e['name']} ({e['type']})" for e in entities])
        relations_str = ", ".join([
            f"{r['source']} -[{r['type']}]-> {r['target']}"
            for r in relations
        ])

        return self.context_template.format(
            entities=entities_str,
            relations=relations_str,
        )

    async def _execute(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> KAGResult:
        """
        INTERNAL: Retrieval from knowledge graph.

        Do NOT call directly - use team.run() instead.
        """
        query = input_data.get("query", "")
        query_entities = input_data.get("entities", [])

        # RFC-025: Use built-in store if configured
        if self._uses_builtin_store():
            if not self._initialized:
                await self.initialize()
            return await self._search_builtin(
                query,
                query_entities,
                context.get("mode", self.query_mode),
                context.get("max_hops", self.max_hops),
            )

        # Get external graph store
        store = self._get_graph_store()

        if store is None:
            # Fallback: return mock results
            entities = [
                {"name": entity, "type": "Entity", "properties": {}}
                for entity in query_entities[:3]
            ] or [{"name": "MockEntity", "type": "Entity", "properties": {}}]

            relations = []
            if len(entities) > 1:
                relations = [
                    {
                        "source": entities[0]["name"],
                        "target": entities[1]["name"],
                        "type": "RELATED_TO",
                    }
                ]

            return KAGResult(
                output={"entities": entities, "relations": relations},
                entities=entities,
                relations=relations,
                query_entities=query_entities,
                success=True,
            )

        # External graph traversal
        subgraph = await self._native_traverse(store, query_entities)
        entities = subgraph.get("entities", [])
        relations = subgraph.get("relations", []) if self.include_relations else []

        return KAGResult(
            output={"entities": entities, "relations": relations},
            entities=entities,
            relations=relations,
            query_entities=query_entities,
            success=True,
        )

    def _get_graph_store(self):
        """Get graph store from runtime context."""
        if hasattr(self._team, "_runtime") and self._team._runtime:
            try:
                return self._team._runtime.get_store(
                    self.graph_store_type or "graph_store"
                )
            except Exception:
                pass
        return None

    async def _native_traverse(
        self, store, entities: List[str]
    ) -> Dict[str, List[Dict]]:
        """Perform native graph traversal."""
        try:
            result = await store.traverse(
                entities=entities,
                max_hops=self.max_hops,
                max_entities=self.max_entities,
            )
            return {
                "entities": result.get("entities", []),
                "relations": result.get("relations", []),
            }
        except Exception:
            return {"entities": [], "relations": []}

    # ═══════════════════════════════════════════════════════
    # Public API methods (RFC-025)
    # ═══════════════════════════════════════════════════════

    async def build_from_text(self, text: str) -> Dict[str, Any]:
        """
        Build graph from a single text.

        Args:
            text: Text to process.

        Returns:
            Dict with entity_count and relation_count.
        """
        if not self._initialized:
            await self.initialize()

        if not self._ner or not self._relation_extractor or not self._graph_store:
            raise ValueError("KAGAgent not properly initialized")

        # Extract entities
        entities = await self._ner.extract(text, self.entity_types)

        # Add entities to graph
        for entity in entities:
            await self._graph_store.add_entity(entity)

        # Extract relations
        relations = []
        if len(entities) >= 2:
            relations = await self._relation_extractor.extract(
                text, entities, self.relation_types
            )
            for relation in relations:
                await self._graph_store.add_relation(relation)

        return {
            "entity_count": len(entities),
            "relation_count": len(relations),
        }

    async def add_entities(self, entities: List[Dict[str, Any]]) -> int:
        """Add entities to the graph."""
        if not self._initialized:
            await self.initialize()

        if not self._graph_store:
            raise ValueError("Graph store not initialized")

        from llmteam.ner.base import Entity

        count = 0
        for e_data in entities:
            entity = Entity(
                name=e_data.get("name", ""),
                type=e_data.get("type", "Entity"),
                properties=e_data.get("properties", {}),
            )
            await self._graph_store.add_entity(entity)
            count += 1

        return count

    async def add_relations(self, relations: List[Dict[str, Any]]) -> int:
        """Add relations to the graph."""
        if not self._initialized:
            await self.initialize()

        if not self._graph_store:
            raise ValueError("Graph store not initialized")

        from llmteam.graphstores.base import Relation

        count = 0
        for r_data in relations:
            relation = Relation(
                source=r_data.get("source", ""),
                target=r_data.get("target", ""),
                type=r_data.get("type", "RELATED_TO"),
                properties=r_data.get("properties", {}),
            )
            await self._graph_store.add_relation(relation)
            count += 1

        return count

    async def find_entities(
        self,
        name: Optional[str] = None,
        type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Find entities by criteria."""
        if not self._initialized:
            await self.initialize()

        if not self._graph_store:
            return []

        entities = await self._graph_store.find_entities(name, type, properties)
        return [
            {"name": e.name, "type": e.type, "properties": e.properties}
            for e in entities
        ]

    async def find_relations(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None,
        type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find relations by criteria."""
        if not self._initialized:
            await self.initialize()

        if not self._graph_store:
            return []

        relations = await self._graph_store.find_relations(source, target, type)
        return [
            {"source": r.source, "target": r.target, "type": r.type, "properties": r.properties}
            for r in relations
        ]

    async def find_path(
        self,
        source: str,
        target: str,
        max_hops: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Find paths between entities."""
        if not self._initialized:
            await self.initialize()

        if not self._graph_store:
            return []

        hops = max_hops or self.max_hops
        paths = await self._graph_store.find_path(source, target, hops)
        return [
            {"entities": p.entities, "relations": p.relations, "length": p.length}
            for p in paths
        ]

    async def get_neighbors(
        self,
        entity: str,
        hops: int = 1,
        relation_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get neighboring entities."""
        if not self._initialized:
            await self.initialize()

        if not self._graph_store:
            return {"entities": [], "relations": []}

        subgraph = await self._graph_store.get_neighbors(entity, hops, relation_types)
        return {
            "entities": [
                {"name": e.name, "type": e.type, "properties": e.properties}
                for e in subgraph.entities
            ],
            "relations": [
                {"source": r.source, "target": r.target, "type": r.type}
                for r in subgraph.relations
            ],
        }

    async def delete_entity(self, name: str) -> bool:
        """Delete an entity and its relations."""
        if not self._graph_store:
            return False
        return await self._graph_store.delete_entity(name)

    async def clear(self) -> int:
        """Clear the entire graph."""
        if not self._graph_store:
            return 0
        return await self._graph_store.clear()

    def save(self, path: str) -> None:
        """Save the graph to disk."""
        if not self._graph_store:
            raise ValueError("Graph store not initialized")

        if hasattr(self._graph_store, "save"):
            self._graph_store.save(path)
        else:
            raise ValueError("Graph store does not support save")

    @property
    def entity_count(self) -> int:
        """Number of entities in the graph."""
        if self._graph_store:
            return self._graph_store.entity_count()
        return 0

    @property
    def relation_count(self) -> int:
        """Number of relations in the graph."""
        if self._graph_store:
            return self._graph_store.relation_count()
        return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "entity_count": self.entity_count,
            "relation_count": self.relation_count,
            "initialized": self._initialized,
            "ner_provider": self.ner_provider,
            "graph_store": self.graph_store_type,
        }

    def visualize(self) -> str:
        """Get Mermaid visualization of the graph."""
        if not self._graph_store:
            return "graph LR\n    Empty[No graph data]"

        if hasattr(self._graph_store, "_entities") and hasattr(self._graph_store, "_graph"):
            # InMemoryGraphStore
            from llmteam.graphstores.base import Subgraph
            entities = list(self._graph_store._entities.values())
            relations = []

            for u, v, data in self._graph_store._graph.edges(data=True):
                from llmteam.graphstores.base import Relation
                relations.append(
                    Relation(
                        source=u,
                        target=v,
                        type=data.get("type", "RELATED_TO"),
                    )
                )

            subgraph = Subgraph(entities=entities, relations=relations)
            return subgraph.to_mermaid()

        return "graph LR\n    External[External graph store]"

    def export_cypher(self) -> str:
        """Export graph as Cypher queries."""
        if self._graph_store and hasattr(self._graph_store, "export_cypher"):
            return self._graph_store.export_cypher()
        return ""
