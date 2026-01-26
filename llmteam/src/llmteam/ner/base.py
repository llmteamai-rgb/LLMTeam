"""
Base NER protocol.

Defines the interface for all NER providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Entity:
    """
    Extracted entity.

    Attributes:
        name: Entity text/name.
        type: Entity type (Person, Organization, etc.).
        properties: Additional properties extracted.
        start_char: Start position in source text.
        end_char: End position in source text.
    """

    name: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    def __hash__(self):
        return hash((self.name, self.type))

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.name == other.name and self.type == other.type


class BaseNER(ABC):
    """
    Abstract base class for NER providers.

    All NER providers must implement the extract method
    to extract named entities from text.
    """

    @property
    @abstractmethod
    def supported_types(self) -> List[str]:
        """Return list of supported entity types."""
        pass

    @abstractmethod
    async def extract(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
    ) -> List[Entity]:
        """
        Extract named entities from text.

        Args:
            text: Text to extract entities from.
            entity_types: Optional list of entity types to extract.
                If None, extracts all supported types.

        Returns:
            List of extracted Entity objects.
        """
        pass

    async def extract_batch(
        self,
        texts: List[str],
        entity_types: Optional[List[str]] = None,
    ) -> List[List[Entity]]:
        """
        Extract entities from multiple texts.

        Args:
            texts: List of texts to process.
            entity_types: Optional list of entity types.

        Returns:
            List of entity lists, one per input text.
        """
        results = []
        for text in texts:
            entities = await self.extract(text, entity_types)
            results.append(entities)
        return results


__all__ = ["BaseNER", "Entity"]
