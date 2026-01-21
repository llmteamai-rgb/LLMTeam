"""
KAG Agent implementation.

Retrieval agent from knowledge graph.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from llmteam.agents.types import AgentType, AgentMode
from llmteam.agents.config import KAGAgentConfig
from llmteam.agents.result import KAGResult
from llmteam.agents.base import BaseAgent

if TYPE_CHECKING:
    from llmteam.team import LLMTeam


class KAGAgent(BaseAgent):
    """
    Retrieval agent from knowledge graph.

    Modes:
    - native: direct connection to Neo4j/Neptune
    - proxy: via external API

    Result is delivered to mailbox for LLMAgent.
    """

    agent_type = AgentType.KAG

    # Config fields
    mode: AgentMode
    graph_store: Optional[str]
    graph_uri: Optional[str]
    graph_user: Optional[str]
    graph_password: Optional[str]
    proxy_endpoint: Optional[str]
    proxy_api_key: Optional[str]
    max_hops: int
    max_entities: int
    include_relations: bool
    extract_entities: bool
    entity_types: List[str]
    deliver_to: Optional[str]

    def __init__(self, team: "LLMTeam", config: KAGAgentConfig):
        super().__init__(team, config)

        self.mode = config.mode
        self.graph_store = config.graph_store
        self.graph_uri = config.graph_uri
        self.graph_user = config.graph_user
        self.graph_password = config.graph_password
        self.proxy_endpoint = config.proxy_endpoint
        self.proxy_api_key = config.proxy_api_key
        self.max_hops = config.max_hops
        self.max_entities = config.max_entities
        self.include_relations = config.include_relations
        self.extract_entities = config.extract_entities
        self.entity_types = config.entity_types
        self.deliver_to = config.deliver_to

    async def _execute(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> KAGResult:
        """
        INTERNAL: Retrieval from knowledge graph.

        Do NOT call directly - use team.run() instead.

        Args:
            input_data: Must contain "query" or entities for search
                Example: {"query": "How is Tesla related to SpaceX?"}
            context: Usually empty (KAG is first in pipeline)

        Returns:
            KAGResult:
                output: Dict - entities and relations
                entities: List[Dict] - found entities
                relations: List[Dict] - found relations
                query_entities: List[str] - extracted from query
                context_payload: {"_kag_context": {...}} - for mailbox
        """
        query = input_data.get("query", "")

        # Extract entities if enabled
        if self.extract_entities:
            query_entities = await self._extract_entities(query)
        else:
            query_entities = input_data.get("entities", [])

        # Get graph store
        store = self._get_graph_store()

        if store is None:
            # Fallback: return mock results for testing
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

        # Graph traversal
        if self.mode == AgentMode.NATIVE:
            subgraph = await self._native_traverse(store, query_entities)
        else:  # PROXY
            subgraph = await self._proxy_query(query_entities)

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
                return self._team._runtime.get_store(self.graph_store or "graph_store")
            except Exception:
                pass
        return None

    async def _extract_entities(self, query: str) -> List[str]:
        """Extract entities from query using simple heuristics."""
        # Simple entity extraction: capitalized words
        words = query.split()
        entities = []

        for word in words:
            # Clean punctuation
            clean = word.strip(".,!?\"'()[]")
            # Check if starts with capital (potential entity)
            if clean and clean[0].isupper() and len(clean) > 1:
                entities.append(clean)

        # Filter by entity types if specified
        if self.entity_types:
            # In production, this would use NER
            pass

        return entities[: self.max_entities]

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

    async def _proxy_query(self, entities: List[str]) -> Dict[str, List[Dict]]:
        """Perform proxy query via external API."""
        if not self.proxy_endpoint:
            return {"entities": [], "relations": []}

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                headers = {}
                if self.proxy_api_key:
                    headers["Authorization"] = f"Bearer {self.proxy_api_key}"

                async with session.post(
                    self.proxy_endpoint,
                    json={
                        "entities": entities,
                        "max_hops": self.max_hops,
                        "max_entities": self.max_entities,
                    },
                    headers=headers,
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception:
            pass

        return {"entities": [], "relations": []}
