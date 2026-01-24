"""
Pricing registry for LLM models.

RFC-010: Cost Tracking & Budget Management.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelPricing:
    """Pricing for a single model (per 1K tokens)."""

    input_per_1k: float
    output_per_1k: float

    def calculate(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD."""
        input_cost = (input_tokens / 1000) * self.input_per_1k
        output_cost = (output_tokens / 1000) * self.output_per_1k
        return input_cost + output_cost


class PricingRegistry:
    """
    Registry of model pricing.

    Provides cost calculation for token usage.
    Users can register custom model pricing.
    """

    # Default pricing (per 1K tokens, USD)
    DEFAULT_PRICING: Dict[str, ModelPricing] = {
        # OpenAI
        "gpt-4o": ModelPricing(input_per_1k=0.0025, output_per_1k=0.01),
        "gpt-4o-mini": ModelPricing(input_per_1k=0.00015, output_per_1k=0.0006),
        "gpt-4-turbo": ModelPricing(input_per_1k=0.01, output_per_1k=0.03),
        "gpt-4": ModelPricing(input_per_1k=0.03, output_per_1k=0.06),
        "gpt-3.5-turbo": ModelPricing(input_per_1k=0.0005, output_per_1k=0.0015),
        "o1": ModelPricing(input_per_1k=0.015, output_per_1k=0.06),
        "o1-mini": ModelPricing(input_per_1k=0.003, output_per_1k=0.012),
        # Anthropic
        "claude-3-5-sonnet-20241022": ModelPricing(input_per_1k=0.003, output_per_1k=0.015),
        "claude-3-5-haiku-20241022": ModelPricing(input_per_1k=0.001, output_per_1k=0.005),
        "claude-3-opus-20240229": ModelPricing(input_per_1k=0.015, output_per_1k=0.075),
        "claude-3-sonnet-20240229": ModelPricing(input_per_1k=0.003, output_per_1k=0.015),
        "claude-3-haiku-20240307": ModelPricing(input_per_1k=0.00025, output_per_1k=0.00125),
        # Google
        "gemini-1.5-pro": ModelPricing(input_per_1k=0.00125, output_per_1k=0.005),
        "gemini-1.5-flash": ModelPricing(input_per_1k=0.000075, output_per_1k=0.0003),
    }

    def __init__(self, custom_pricing: Optional[Dict[str, ModelPricing]] = None):
        self._pricing: Dict[str, ModelPricing] = dict(self.DEFAULT_PRICING)
        if custom_pricing:
            self._pricing.update(custom_pricing)

    def register(self, model: str, pricing: ModelPricing) -> None:
        """Register pricing for a model."""
        self._pricing[model] = pricing

    def get_pricing(self, model: str) -> Optional[ModelPricing]:
        """Get pricing for a model. Returns None if unknown."""
        # Try exact match first
        if model in self._pricing:
            return self._pricing[model]

        # Try prefix match (e.g., "gpt-4o-2024-08-06" matches "gpt-4o")
        for known_model, pricing in self._pricing.items():
            if model.startswith(known_model):
                return pricing

        return None

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculate cost for token usage.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD (0.0 if model pricing unknown)
        """
        pricing = self.get_pricing(model)
        if pricing is None:
            return 0.0
        return pricing.calculate(input_tokens, output_tokens)

    def list_models(self) -> list:
        """List all registered model names."""
        return list(self._pricing.keys())
