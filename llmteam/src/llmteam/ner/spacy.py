"""
SpaCy-based NER.

Uses SpaCy for fast offline named entity extraction.
"""

from typing import List, Optional

from llmteam.ner.base import BaseNER, Entity


# SpaCy entity type mapping to our types
SPACY_TYPE_MAP = {
    "PERSON": "Person",
    "PER": "Person",
    "ORG": "Organization",
    "GPE": "Location",
    "LOC": "Location",
    "DATE": "Date",
    "TIME": "Date",
    "MONEY": "Money",
    "PERCENT": "Percentage",
    "PRODUCT": "Product",
    "EVENT": "Event",
    "WORK_OF_ART": "Work",
    "LAW": "Law",
    "LANGUAGE": "Language",
    "NORP": "Group",
    "FAC": "Facility",
    "CARDINAL": "Number",
    "ORDINAL": "Number",
    "QUANTITY": "Quantity",
}


class SpacyNER(BaseNER):
    """
    SpaCy-based named entity recognition.

    Uses SpaCy for fast offline NER. Good for high-volume processing.

    Requires:
        pip install spacy
        python -m spacy download en_core_web_lg  # or ru_core_news_lg

    Common models:
        - en_core_web_lg: English (best quality)
        - en_core_web_sm: English (fast)
        - ru_core_news_lg: Russian (best quality)
        - ru_core_news_sm: Russian (fast)
    """

    def __init__(
        self,
        model: str = "en_core_web_sm",
        type_mapping: Optional[dict] = None,
    ):
        """
        Initialize the SpaCy NER.

        Args:
            model: SpaCy model name to use.
            type_mapping: Optional custom mapping from SpaCy types to our types.
        """
        self._model_name = model
        self._type_mapping = type_mapping or SPACY_TYPE_MAP
        self._nlp = None

    def _ensure_model(self) -> None:
        """Ensure SpaCy model is loaded."""
        if self._nlp is not None:
            return

        try:
            import spacy
        except ImportError:
            raise ImportError(
                "SpaCy NER requires 'spacy'. Install with: pip install spacy"
            )

        try:
            self._nlp = spacy.load(self._model_name)
        except OSError:
            raise ImportError(
                f"SpaCy model '{self._model_name}' not found. "
                f"Install with: python -m spacy download {self._model_name}"
            )

    @property
    def supported_types(self) -> List[str]:
        """Return list of supported entity types."""
        return list(set(self._type_mapping.values()))

    async def extract(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
    ) -> List[Entity]:
        """
        Extract entities using SpaCy.

        Args:
            text: Text to extract entities from.
            entity_types: Optional list of types to filter.

        Returns:
            List of extracted Entity objects.
        """
        self._ensure_model()

        doc = self._nlp(text)
        entities: List[Entity] = []
        seen = set()

        for ent in doc.ents:
            # Map SpaCy type to our type
            entity_type = self._type_mapping.get(ent.label_, ent.label_)

            # Filter by requested types
            if entity_types and entity_type not in entity_types:
                continue

            # Deduplicate
            key = (ent.text.lower(), entity_type)
            if key in seen:
                continue
            seen.add(key)

            entities.append(
                Entity(
                    name=ent.text,
                    type=entity_type,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    properties={"spacy_label": ent.label_},
                )
            )

        return entities


__all__ = ["SpacyNER"]
