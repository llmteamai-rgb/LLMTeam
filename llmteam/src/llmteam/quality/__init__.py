"""
Quality Slider module (RFC-008).

Provides a single parameter (0-100) to control quality/cost tradeoff.
"""

from llmteam.quality.manager import QualityManager
from llmteam.quality.presets import (
    QualityPreset,
    TaskComplexity,
    PipelineDepth,
    MODEL_MATRIX,
    GENERATION_PARAMS,
    COST_MULTIPLIERS,
)
from llmteam.quality.estimator import CostEstimate, CostEstimator

__all__ = [
    # Manager
    "QualityManager",
    # Presets
    "QualityPreset",
    "TaskComplexity",
    "PipelineDepth",
    "MODEL_MATRIX",
    "GENERATION_PARAMS",
    "COST_MULTIPLIERS",
    # Estimator
    "CostEstimate",
    "CostEstimator",
]
