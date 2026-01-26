"""
Named Entity Recognition (NER) module.

Provides NER providers for KAG pipeline.
"""

from llmteam.ner.base import BaseNER, Entity
from llmteam.ner.simple import SimpleNER
from llmteam.ner.llm import LLMNER

# SpacyNER is lazy-loaded to avoid import errors when spacy is missing
# Use: from llmteam.ner.spacy import SpacyNER

__all__ = [
    "BaseNER",
    "Entity",
    "SimpleNER",
    "LLMNER",
]
