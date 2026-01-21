"""
Quality presets and model matrices (RFC-008).

Defines mappings from quality levels to models, parameters, and pipeline depth.
"""

from enum import Enum
from typing import Dict, Tuple


class QualityPreset(str, Enum):
    """Named quality presets."""

    DRAFT = "draft"           # quality=20
    ECONOMY = "economy"       # quality=30
    BALANCED = "balanced"     # quality=50
    PRODUCTION = "production" # quality=75
    BEST = "best"             # quality=95
    AUTO = "auto"             # Automatic based on budget


# Preset to numeric value mapping
PRESET_VALUES: Dict[str, int] = {
    "draft": 20,
    "economy": 30,
    "balanced": 50,
    "production": 75,
    "best": 95,
}


class TaskComplexity(str, Enum):
    """Task complexity levels for model selection."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class PipelineDepth(str, Enum):
    """Pipeline depth recommendations."""

    SHALLOW = "shallow"  # 1-2 agents
    MEDIUM = "medium"    # 2-4 agents
    DEEP = "deep"        # 3-6 agents


# Model selection matrix: [complexity][quality_range] -> model
# Quality ranges: (0, 30), (30, 70), (70, 100)
MODEL_MATRIX: Dict[str, Dict[Tuple[int, int], str]] = {
    "simple": {
        (0, 30): "gpt-4o-mini",
        (30, 70): "gpt-4o-mini",
        (70, 100): "gpt-4o",
    },
    "medium": {
        (0, 30): "gpt-4o-mini",
        (30, 70): "gpt-4o",
        (70, 100): "gpt-4-turbo",
    },
    "complex": {
        (0, 30): "gpt-4o",
        (30, 70): "gpt-4-turbo",
        (70, 100): "gpt-4-turbo",  # or claude-3-opus
    },
}


# Generation parameters by quality range
GENERATION_PARAMS: Dict[Tuple[int, int], Dict[str, any]] = {
    (0, 30): {
        "max_tokens": 500,
        "temperature": 0.3,
    },
    (30, 70): {
        "max_tokens": 1000,
        "temperature": 0.5,
    },
    (70, 100): {
        "max_tokens": 2000,
        "temperature": 0.7,
    },
}


# Pipeline depth by quality
PIPELINE_DEPTH: Dict[Tuple[int, int], PipelineDepth] = {
    (0, 30): PipelineDepth.SHALLOW,
    (30, 70): PipelineDepth.MEDIUM,
    (70, 100): PipelineDepth.DEEP,
}


# Agent count recommendations by pipeline depth
AGENT_COUNT_RANGE: Dict[PipelineDepth, Tuple[int, int]] = {
    PipelineDepth.SHALLOW: (1, 2),
    PipelineDepth.MEDIUM: (2, 4),
    PipelineDepth.DEEP: (3, 6),
}


# Base costs by task complexity (in USD)
BASE_COSTS: Dict[str, float] = {
    "simple": 0.01,
    "medium": 0.05,
    "complex": 0.15,
}


# Cost multipliers by quality
# Formula: 0.5 + (quality / 100) * 4.5
# quality=0 -> 0.5x, quality=100 -> 5.0x
COST_MULTIPLIERS: Dict[int, float] = {
    q: 0.5 + (q / 100) * 4.5 for q in range(0, 101, 10)
}


def get_quality_range(quality: int) -> Tuple[int, int]:
    """Get the quality range tuple for a given quality value."""
    if quality < 30:
        return (0, 30)
    elif quality < 70:
        return (30, 70)
    else:
        return (70, 100)
