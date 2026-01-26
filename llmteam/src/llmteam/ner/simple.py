"""
Simple regex-based NER.

Basic NER using pattern matching for common entity types.
"""

import re
from typing import List, Optional

from llmteam.ner.base import BaseNER, Entity


class SimpleNER(BaseNER):
    """
    Simple regex-based NER.

    Uses pattern matching for common entity types.
    Good for quick extraction without external dependencies.

    Supported types:
        - Date: Various date formats
        - Email: Email addresses
        - URL: Web URLs
        - Phone: Phone numbers
        - Money: Currency amounts
        - Percentage: Percentage values
        - Number: Large numbers
    """

    # Common patterns for entity extraction
    PATTERNS = {
        "Date": [
            r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b",  # 01/02/2023, 1-2-23
            r"\b\d{4}[/.-]\d{1,2}[/.-]\d{1,2}\b",  # 2023-01-02
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",  # January 1, 2023
            r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b",  # 1 January 2023
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        ],
        "Email": [
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        ],
        "URL": [
            r"https?://[^\s<>\"{}|\\^`\[\]]+",
            r"www\.[^\s<>\"{}|\\^`\[\]]+",
        ],
        "Phone": [
            r"\b\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            r"\b\+?\d{1,3}[-.\s]?\d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{2,4}\b",
        ],
        "Money": [
            r"\$\s?\d+(?:,\d{3})*(?:\.\d{2})?(?:\s?(?:million|billion|M|B|K))?\b",
            r"€\s?\d+(?:,\d{3})*(?:\.\d{2})?\b",
            r"£\s?\d+(?:,\d{3})*(?:\.\d{2})?\b",
            r"\b\d+(?:,\d{3})*(?:\.\d{2})?\s?(?:dollars?|euros?|pounds?|USD|EUR|GBP)\b",
        ],
        "Percentage": [
            r"\b\d+(?:\.\d+)?%\b",
            r"\b\d+(?:\.\d+)?\s+percent\b",
        ],
        "Number": [
            r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b",  # Large numbers with commas
        ],
    }

    def __init__(self, case_sensitive: bool = False):
        """
        Initialize the simple NER.

        Args:
            case_sensitive: Whether pattern matching is case sensitive.
        """
        self.case_sensitive = case_sensitive
        self._compiled_patterns = {}

        # Compile patterns
        flags = 0 if case_sensitive else re.IGNORECASE
        for entity_type, patterns in self.PATTERNS.items():
            self._compiled_patterns[entity_type] = [
                re.compile(p, flags) for p in patterns
            ]

    @property
    def supported_types(self) -> List[str]:
        """Return list of supported entity types."""
        return list(self.PATTERNS.keys())

    async def extract(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
    ) -> List[Entity]:
        """
        Extract entities using pattern matching.

        Args:
            text: Text to extract entities from.
            entity_types: Optional list of types to extract.

        Returns:
            List of extracted Entity objects.
        """
        types_to_extract = entity_types or self.supported_types
        entities: List[Entity] = []
        seen = set()

        for entity_type in types_to_extract:
            if entity_type not in self._compiled_patterns:
                continue

            for pattern in self._compiled_patterns[entity_type]:
                for match in pattern.finditer(text):
                    name = match.group().strip()
                    key = (name.lower(), entity_type)

                    if key not in seen:
                        seen.add(key)
                        entities.append(
                            Entity(
                                name=name,
                                type=entity_type,
                                start_char=match.start(),
                                end_char=match.end(),
                            )
                        )

        return entities


__all__ = ["SimpleNER"]
