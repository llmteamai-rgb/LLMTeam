"""
LLM-based NER.

Uses LLM for high-quality named entity extraction.
"""

import json
import os
from typing import List, Optional

from llmteam.ner.base import BaseNER, Entity


# Default entity types
DEFAULT_ENTITY_TYPES = [
    "Person",
    "Organization",
    "Location",
    "Date",
    "Product",
    "Event",
    "Technology",
    "Money",
]


class LLMNER(BaseNER):
    """
    LLM-based named entity recognition.

    Uses LLM to extract entities with high quality.
    Best for complex texts and domain-specific entities.

    Requires:
        pip install openai
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        entity_types: Optional[List[str]] = None,
        extract_properties: bool = True,
        language: str = "auto",
    ):
        """
        Initialize the LLM NER.

        Args:
            model: LLM model to use.
            api_key: OpenAI API key.
            entity_types: Default entity types to extract.
            extract_properties: Whether to extract entity properties.
            language: Expected text language ("auto", "en", "ru", etc.).
        """
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._entity_types = entity_types or DEFAULT_ENTITY_TYPES
        self._extract_properties = extract_properties
        self._language = language
        self._client = None

    def _ensure_client(self) -> None:
        """Ensure OpenAI client is initialized."""
        if self._client is not None:
            return

        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "LLM NER requires 'openai'. Install with: pip install openai"
            )

        if not self._api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY env var or pass api_key."
            )

        self._client = AsyncOpenAI(api_key=self._api_key)

    @property
    def supported_types(self) -> List[str]:
        """Return list of supported entity types."""
        return self._entity_types

    def _build_prompt(self, text: str, entity_types: List[str]) -> str:
        """Build the extraction prompt."""
        types_str = ", ".join(entity_types)

        prompt = f"""Extract named entities from the following text.

Entity types to extract: {types_str}

For each entity, provide:
- name: The exact text of the entity
- type: One of the specified types
- properties: Any additional attributes (dates, numbers, descriptions)

Return a JSON array of entities. Example:
[
    {{"name": "Tesla", "type": "Organization", "properties": {{"industry": "electric vehicles"}}}},
    {{"name": "Elon Musk", "type": "Person", "properties": {{"role": "CEO"}}}},
    {{"name": "2003", "type": "Date", "properties": {{"context": "founding year"}}}}
]

If no entities are found, return an empty array: []

Text to analyze:
\"\"\"
{text}
\"\"\"

JSON response:"""

        return prompt

    async def extract(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
    ) -> List[Entity]:
        """
        Extract entities using LLM.

        Args:
            text: Text to extract entities from.
            entity_types: Optional list of types to extract.

        Returns:
            List of extracted Entity objects.
        """
        self._ensure_client()

        types_to_extract = entity_types or self._entity_types
        prompt = self._build_prompt(text, types_to_extract)

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a named entity recognition expert. Extract entities accurately and return valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=2000,
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON response
            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            entities_data = json.loads(content)

            # Convert to Entity objects
            entities = []
            for item in entities_data:
                if isinstance(item, dict) and "name" in item and "type" in item:
                    entities.append(
                        Entity(
                            name=item["name"],
                            type=item["type"],
                            properties=item.get("properties", {}),
                        )
                    )

            return entities

        except json.JSONDecodeError:
            # If JSON parsing fails, return empty list
            return []
        except Exception as e:
            # Log error but don't crash
            import warnings
            warnings.warn(f"LLM NER extraction failed: {e}")
            return []


__all__ = ["LLMNER"]
