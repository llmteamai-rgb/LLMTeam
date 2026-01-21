"""
QualityManager for RFC-008 Quality Slider.

Manages quality settings and provides model/parameter selection.
"""

from typing import Any, Dict, Optional, Union

from llmteam.quality.presets import (
    PRESET_VALUES,
    MODEL_MATRIX,
    GENERATION_PARAMS,
    PIPELINE_DEPTH,
    AGENT_COUNT_RANGE,
    BASE_COSTS,
    TaskComplexity,
    PipelineDepth,
    get_quality_range,
)


class QualityManager:
    """
    Manages quality slider settings (RFC-008).

    The quality slider is a single parameter (0-100) that controls
    the quality/cost tradeoff for LLM operations.

    Example:
        manager = QualityManager(quality=70)

        model = manager.get_model("medium")  # -> "gpt-4-turbo"
        params = manager.get_generation_params()  # -> {"max_tokens": 2000, ...}
        depth = manager.get_pipeline_depth()  # -> PipelineDepth.DEEP
    """

    def __init__(self, quality: Union[int, str] = 50):
        """
        Initialize QualityManager.

        Args:
            quality: Quality value (0-100) or preset name
                     ("draft", "economy", "balanced", "production", "best")
        """
        self._quality = self._normalize(quality)
        self._auto_mode = quality == "auto"
        self._daily_budget: Optional[float] = None
        self._spent_today: float = 0.0

    def _normalize(self, quality: Union[int, str]) -> int:
        """Normalize quality value to 0-100 range."""
        if isinstance(quality, str):
            if quality == "auto":
                return 50  # Default for auto mode
            return PRESET_VALUES.get(quality.lower(), 50)
        return max(0, min(100, int(quality)))

    @property
    def quality(self) -> int:
        """Get current quality value."""
        if self._auto_mode:
            return self._calculate_auto_quality()
        return self._quality

    @quality.setter
    def quality(self, value: Union[int, str]) -> None:
        """Set quality value."""
        self._quality = self._normalize(value)
        self._auto_mode = value == "auto"

    def _calculate_auto_quality(self) -> int:
        """Calculate quality based on remaining budget (for auto mode)."""
        if self._daily_budget is None or self._daily_budget <= 0:
            return self._quality

        remaining = self._daily_budget - self._spent_today
        ratio = remaining / self._daily_budget

        # Scale quality based on remaining budget
        # Full budget -> quality 70
        # Half budget -> quality 50
        # Low budget -> quality 30
        if ratio > 0.7:
            return 70
        elif ratio > 0.3:
            return 50
        elif ratio > 0.1:
            return 30
        else:
            return 20

    def set_daily_budget(self, budget: float) -> None:
        """Set daily budget for auto mode."""
        self._daily_budget = budget

    def record_spend(self, amount: float) -> None:
        """Record spending for auto mode budget tracking."""
        self._spent_today += amount

    def reset_daily_spend(self) -> None:
        """Reset daily spend counter."""
        self._spent_today = 0.0

    def get_model(self, task_complexity: Union[str, TaskComplexity]) -> str:
        """
        Get recommended model for task complexity.

        Args:
            task_complexity: "simple", "medium", or "complex"

        Returns:
            Model name (e.g., "gpt-4o-mini", "gpt-4o", "gpt-4-turbo")
        """
        complexity = task_complexity.value if isinstance(task_complexity, TaskComplexity) else task_complexity
        quality_range = get_quality_range(self.quality)

        matrix = MODEL_MATRIX.get(complexity, MODEL_MATRIX["medium"])
        return matrix.get(quality_range, "gpt-4o")

    def get_generation_params(self) -> Dict[str, Any]:
        """
        Get generation parameters for current quality.

        Returns:
            Dict with max_tokens, temperature, etc.
        """
        quality_range = get_quality_range(self.quality)
        return GENERATION_PARAMS.get(quality_range, GENERATION_PARAMS[(30, 70)]).copy()

    def get_pipeline_depth(self) -> PipelineDepth:
        """
        Get recommended pipeline depth.

        Returns:
            PipelineDepth (SHALLOW, MEDIUM, or DEEP)
        """
        quality_range = get_quality_range(self.quality)
        return PIPELINE_DEPTH.get(quality_range, PipelineDepth.MEDIUM)

    def get_agent_count_range(self) -> tuple[int, int]:
        """
        Get recommended agent count range.

        Returns:
            Tuple of (min_agents, max_agents)
        """
        depth = self.get_pipeline_depth()
        return AGENT_COUNT_RANGE.get(depth, (2, 4))

    def get_cost_multiplier(self) -> float:
        """
        Get cost multiplier for current quality.

        Returns:
            Multiplier (0.5 at quality=0, 5.0 at quality=100)
        """
        return 0.5 + (self.quality / 100) * 4.5

    def estimate_cost(
        self,
        task_complexity: Union[str, TaskComplexity] = "medium",
    ) -> tuple[float, float]:
        """
        Estimate cost range for a task.

        Args:
            task_complexity: "simple", "medium", or "complex"

        Returns:
            Tuple of (min_cost, max_cost) in USD
        """
        complexity = task_complexity.value if isinstance(task_complexity, TaskComplexity) else task_complexity
        base = BASE_COSTS.get(complexity, BASE_COSTS["medium"])
        multiplier = self.get_cost_multiplier()

        estimated = base * multiplier

        # Return range ±30%
        return (estimated * 0.7, estimated * 1.3)

    def with_importance(self, importance: str) -> int:
        """
        Adjust quality based on task importance.

        Args:
            importance: "high", "medium", or "low"

        Returns:
            Adjusted quality value
        """
        adjustments = {
            "high": 20,
            "medium": 0,
            "low": -20,
        }
        adjustment = adjustments.get(importance, 0)
        return max(0, min(100, self.quality + adjustment))

    def to_dict(self) -> Dict[str, Any]:
        """Export settings to dictionary."""
        return {
            "quality": self.quality,
            "auto_mode": self._auto_mode,
            "daily_budget": self._daily_budget,
            "spent_today": self._spent_today,
            "model_simple": self.get_model("simple"),
            "model_medium": self.get_model("medium"),
            "model_complex": self.get_model("complex"),
            "pipeline_depth": self.get_pipeline_depth().value,
            "cost_multiplier": self.get_cost_multiplier(),
        }

    def __repr__(self) -> str:
        mode = "auto" if self._auto_mode else "fixed"
        return f"<QualityManager quality={self.quality} mode={mode}>"
