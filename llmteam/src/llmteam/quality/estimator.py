"""
Cost estimation for RFC-008 Quality Slider.

Provides cost estimates based on quality level and task complexity.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llmteam.quality.presets import BASE_COSTS, TaskComplexity


@dataclass
class CostEstimate:
    """Cost estimate result."""

    min_cost: float
    """Minimum estimated cost in USD."""

    max_cost: float
    """Maximum estimated cost in USD."""

    quality: int
    """Quality level used for estimate."""

    task_complexity: str
    """Task complexity level."""

    breakdown: Optional[Dict[str, float]] = None
    """Optional cost breakdown by component."""

    @property
    def average(self) -> float:
        """Average of min and max cost."""
        return (self.min_cost + self.max_cost) / 2

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return {
            "min_cost": self.min_cost,
            "max_cost": self.max_cost,
            "average_cost": self.average,
            "quality": self.quality,
            "task_complexity": self.task_complexity,
            "breakdown": self.breakdown,
        }

    def __str__(self) -> str:
        return f"${self.min_cost:.2f} - ${self.max_cost:.2f}"


class CostEstimator:
    """
    Estimates costs for LLMTeam runs based on quality.

    Example:
        estimator = CostEstimator()
        estimate = estimator.estimate(quality=70, complexity="medium")
        print(f"Estimated cost: {estimate}")  # "$0.12 - $0.22"
    """

    # Model pricing per 1K tokens (approximate, USD)
    MODEL_PRICING = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
    }

    # Average tokens per agent call
    AVG_INPUT_TOKENS = 500
    AVG_OUTPUT_TOKENS = 300

    def estimate(
        self,
        quality: int,
        complexity: str = "medium",
        agent_count: Optional[int] = None,
    ) -> CostEstimate:
        """
        Estimate cost for a run.

        Args:
            quality: Quality level (0-100)
            complexity: Task complexity ("simple", "medium", "complex")
            agent_count: Optional specific agent count (otherwise estimated)

        Returns:
            CostEstimate with min/max range
        """
        # Calculate cost multiplier from quality
        multiplier = 0.5 + (quality / 100) * 4.5

        # Get base cost for complexity
        base = BASE_COSTS.get(complexity, BASE_COSTS["medium"])

        # Calculate estimate
        estimated = base * multiplier

        # Return range ±30%
        return CostEstimate(
            min_cost=round(estimated * 0.7, 4),
            max_cost=round(estimated * 1.3, 4),
            quality=quality,
            task_complexity=complexity,
        )

    def estimate_detailed(
        self,
        quality: int,
        agents: List[Dict[str, Any]],
        avg_input_tokens: int = 500,
        avg_output_tokens: int = 300,
    ) -> CostEstimate:
        """
        Detailed cost estimate based on agent configuration.

        Args:
            quality: Quality level (0-100)
            agents: List of agent configurations with 'model' key
            avg_input_tokens: Average input tokens per call
            avg_output_tokens: Average output tokens per call

        Returns:
            CostEstimate with breakdown by agent
        """
        breakdown = {}
        total_min = 0.0
        total_max = 0.0

        for agent in agents:
            model = agent.get("model", "gpt-4o")
            role = agent.get("role", "agent")

            pricing = self.MODEL_PRICING.get(
                model,
                self.MODEL_PRICING["gpt-4o"]
            )

            # Calculate cost per call
            input_cost = (avg_input_tokens / 1000) * pricing["input"]
            output_cost = (avg_output_tokens / 1000) * pricing["output"]
            agent_cost = input_cost + output_cost

            breakdown[role] = agent_cost
            total_min += agent_cost * 0.7
            total_max += agent_cost * 1.3

        return CostEstimate(
            min_cost=round(total_min, 4),
            max_cost=round(total_max, 4),
            quality=quality,
            task_complexity="custom",
            breakdown=breakdown,
        )

    def suggest_quality_for_budget(
        self,
        budget: float,
        complexity: str = "medium",
    ) -> int:
        """
        Suggest quality level for a given budget.

        Args:
            budget: Maximum budget in USD
            complexity: Task complexity

        Returns:
            Suggested quality level (0-100)
        """
        base = BASE_COSTS.get(complexity, BASE_COSTS["medium"])

        # Solve for quality: budget = base * (0.5 + quality/100 * 4.5) * 1.3
        # quality = ((budget / base / 1.3) - 0.5) * 100 / 4.5
        try:
            max_multiplier = budget / base / 1.3  # Use max estimate
            quality = ((max_multiplier - 0.5) * 100) / 4.5
            return max(0, min(100, int(quality)))
        except ZeroDivisionError:
            return 50
