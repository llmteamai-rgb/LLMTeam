"""
Relation extraction using LLM.

Extracts relations between entities from text.
"""

import json
import os
from typing import Any, Dict, List, Optional

from llmteam.ner.base import Entity
from llmteam.graphstores.base import Relation


# Default relation types
DEFAULT_RELATION_TYPES = [
    "WORKS_FOR",
    "FOUNDED",
    "CEO_OF",
    "LOCATED_IN",
    "OWNS",
    "PART_OF",
    "CREATED",
    "RELATED_TO",
    "SUBSIDIARY_OF",
    "ACQUIRED",
    "PARTNER_OF",
    "INVESTED_IN",
]


class RelationExtractor:
    """
    LLM-based relation extractor.

    Extracts relations between entities from text using LLM.

    Requires:
        pip install openai
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        relation_types: Optional[List[str]] = None,
        extract_properties: bool = True,
    ):
        """
        Initialize the relation extractor.

        Args:
            model: LLM model to use.
            api_key: OpenAI API key.
            relation_types: List of relation types to extract.
            extract_properties: Whether to extract relation properties.
        """
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._relation_types = relation_types or DEFAULT_RELATION_TYPES
        self._extract_properties = extract_properties
        self._client = None

    def _ensure_client(self) -> None:
        """Ensure OpenAI client is initialized."""
        if self._client is not None:
            return

        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "Relation extraction requires 'openai'. Install with: pip install openai"
            )

        if not self._api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY env var or pass api_key."
            )

        self._client = AsyncOpenAI(api_key=self._api_key)

    def _build_prompt(
        self, text: str, entities: List[Entity], relation_types: List[str]
    ) -> str:
        """Build the extraction prompt."""
        entities_str = ", ".join([f"{e.name} ({e.type})" for e in entities])
        types_str = ", ".join(relation_types)

        prompt = f"""Extract relations between the given entities from the text.

Entities found in text: {entities_str}

Relation types to consider: {types_str}

For each relation, provide:
- source: Source entity name (must be from the entities list)
- target: Target entity name (must be from the entities list)
- type: One of the specified relation types
- properties: Additional attributes (dates, context, etc.)

Return a JSON array of relations. Example:
[
    {{"source": "Elon Musk", "target": "Tesla", "type": "CEO_OF", "properties": {{"since": "2008"}}}},
    {{"source": "Tesla", "target": "Fremont", "type": "LOCATED_IN", "properties": {{}}}}
]

Rules:
1. Only create relations between entities in the provided list
2. Each relation should be directly supported by the text
3. Use the most specific relation type that applies
4. If no relations are found, return an empty array: []

Text to analyze:
\"\"\"
{text}
\"\"\"

JSON response:"""

        return prompt

    async def extract(
        self,
        text: str,
        entities: List[Entity],
        relation_types: Optional[List[str]] = None,
    ) -> List[Relation]:
        """
        Extract relations between entities.

        Args:
            text: Source text.
            entities: Entities to find relations between.
            relation_types: Optional list of relation types.

        Returns:
            List of extracted Relation objects.
        """
        if len(entities) < 2:
            return []  # Need at least 2 entities for relations

        self._ensure_client()

        types_to_use = relation_types or self._relation_types
        prompt = self._build_prompt(text, entities, types_to_use)

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a relation extraction expert. Extract relations accurately and return valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=2000,
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON response
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            relations_data = json.loads(content)

            # Validate entity names
            entity_names = {e.name for e in entities}

            # Convert to Relation objects
            relations = []
            for item in relations_data:
                if not isinstance(item, dict):
                    continue

                source = item.get("source", "")
                target = item.get("target", "")
                rel_type = item.get("type", "RELATED_TO")

                # Validate that source and target are in our entities
                if source not in entity_names or target not in entity_names:
                    continue

                # Validate relation type
                if rel_type not in types_to_use:
                    rel_type = "RELATED_TO"

                relations.append(
                    Relation(
                        source=source,
                        target=target,
                        type=rel_type,
                        properties=item.get("properties", {}),
                    )
                )

            return relations

        except json.JSONDecodeError:
            return []
        except Exception as e:
            import warnings
            warnings.warn(f"Relation extraction failed: {e}")
            return []


__all__ = ["RelationExtractor"]
