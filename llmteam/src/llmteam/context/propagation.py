"""
Context propagation configuration for llmteam.

This module defines how context data flows between hierarchy levels:
- Propagation up: agent → pipeline → group
- Propagation down: group → pipeline → agent
- Aggregation rules for combining child data

Quick Start:
    from llmteam.context import ContextPropagationConfig

    # Default configuration
    config = ContextPropagationConfig()

    # Custom configuration
    config = ContextPropagationConfig(
        propagate_up=["status", "confidence", "error_count"],
        propagate_down=["global_config", "timeout"],
        aggregation_rules={
            "confidence": "avg",
            "error_count": "sum",
            "status": "worst",
        },
    )
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ContextPropagationConfig:
    """
    Configuration for context data propagation.

    Attributes:
        propagate_up: Fields to propagate from child to parent
        propagate_down: Fields to propagate from parent to child
        aggregation_rules: Rules for aggregating child values

    Aggregation Rules:
        - "sum": Sum all child values (numeric)
        - "avg": Average of child values (numeric)
        - "max": Maximum child value (numeric)
        - "min": Minimum child value (numeric)
        - "worst": Worst status (idle < running < error < failed)
        - "last": Last value wins (default)

    Example:
        config = ContextPropagationConfig(
            propagate_up=["status", "confidence"],
            propagate_down=["timeout", "global_state"],
            aggregation_rules={
                "confidence": "avg",
                "status": "worst",
            },
        )

        manager = ContextManager(config)
    """

    # Fields to propagate up the hierarchy (agent → pipeline → group)
    propagate_up: List[str] = field(default_factory=lambda: [
        "status",
        "confidence",
        "error_count",
    ])

    # Fields to propagate down the hierarchy (group → pipeline → agent)
    propagate_down: List[str] = field(default_factory=lambda: [
        "global_config",
        "shared_state",
    ])

    # Aggregation rules when propagating up
    # Maps field name to aggregation strategy
    aggregation_rules: Dict[str, str] = field(default_factory=lambda: {
        "confidence": "avg",       # Average confidence
        "error_count": "sum",      # Total errors
        "status": "worst",         # Worst status among children
    })

    def add_propagate_up(self, field: str, aggregation: str = "last") -> None:
        """
        Add a field to propagate up.

        Args:
            field: Field name to propagate
            aggregation: Aggregation rule ("sum", "avg", "max", "min", "worst", "last")
        """
        if field not in self.propagate_up:
            self.propagate_up.append(field)
        self.aggregation_rules[field] = aggregation

    def add_propagate_down(self, field: str) -> None:
        """
        Add a field to propagate down.

        Args:
            field: Field name to propagate
        """
        if field not in self.propagate_down:
            self.propagate_down.append(field)

    def remove_propagate_up(self, field: str) -> None:
        """
        Remove a field from upward propagation.

        Args:
            field: Field name to remove
        """
        if field in self.propagate_up:
            self.propagate_up.remove(field)
        if field in self.aggregation_rules:
            del self.aggregation_rules[field]

    def remove_propagate_down(self, field: str) -> None:
        """
        Remove a field from downward propagation.

        Args:
            field: Field name to remove
        """
        if field in self.propagate_down:
            self.propagate_down.remove(field)

    def set_aggregation_rule(self, field: str, rule: str) -> None:
        """
        Set aggregation rule for a field.

        Args:
            field: Field name
            rule: Aggregation rule
        """
        self.aggregation_rules[field] = rule

    def get_aggregation_rule(self, field: str) -> str:
        """
        Get aggregation rule for a field.

        Args:
            field: Field name

        Returns:
            Aggregation rule (default: "last")
        """
        return self.aggregation_rules.get(field, "last")
