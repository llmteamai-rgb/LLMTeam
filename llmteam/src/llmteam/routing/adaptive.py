"""
Adaptive Routing Types.

RFC-022: Dataclasses for adaptive routing in hybrid workflows.

AdaptiveStep enables rule-based routing with LLM fallback:
1. First tries deterministic rules (cheap, fast)
2. Falls back to LLM decision if no rules match
3. Uses default route if nothing else works
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json


class RoutingMethod(str, Enum):
    """Method used for routing decision."""

    RULE = "rule"       # Deterministic rule matched
    LLM = "llm"         # LLM made the decision
    DEFAULT = "default" # Default route used


@dataclass
class RouteOption:
    """
    Possible route for LLM to choose.

    Used in LLMFallbackConfig to describe available routing targets.
    """

    target: str
    """Target step ID to route to."""

    description: str
    """Human-readable description of when to use this route."""

    when: str = ""
    """Optional hint about conditions for this route."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target": self.target,
            "description": self.description,
            "when": self.when,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RouteOption":
        """Create from dictionary."""
        return cls(
            target=data["target"],
            description=data.get("description", ""),
            when=data.get("when", ""),
        )


@dataclass
class RoutingRule:
    """
    Rule-based routing condition.

    Evaluated against step output to determine routing.
    Uses safe expression evaluation (no arbitrary code).
    """

    condition: str
    """
    Condition expression evaluated against step output.

    Supported formats:
    - "output.field == 'value'"
    - "output.score > 80"
    - "output.status in ['ready', 'approved']"
    """

    target: str
    """Target step ID if condition is true."""

    description: str = ""
    """Human-readable description of this rule."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "condition": self.condition,
            "target": self.target,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoutingRule":
        """Create from dictionary."""
        return cls(
            condition=data["condition"],
            target=data["target"],
            description=data.get("description", ""),
        )


@dataclass
class LLMFallbackConfig:
    """
    LLM-based routing when rules don't match.

    Used as fallback when deterministic rules can't decide.
    """

    model: str = "gpt-4o-mini"
    """Model for routing decision. Use fast/cheap model."""

    prompt: str = ""
    """
    System prompt for routing decision.

    Should explain the routing context and available options.
    """

    routes: List[RouteOption] = field(default_factory=list)
    """Available routes to choose from."""

    max_tokens: int = 100
    """Max tokens for decision response. Keep small."""

    temperature: float = 0.0
    """Temperature for decision. Use 0 for determinism."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "prompt": self.prompt,
            "routes": [r.to_dict() for r in self.routes],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMFallbackConfig":
        """Create from dictionary."""
        routes = [RouteOption.from_dict(r) for r in data.get("routes", [])]
        return cls(
            model=data.get("model", "gpt-4o-mini"),
            prompt=data.get("prompt", ""),
            routes=routes,
            max_tokens=data.get("max_tokens", 100),
            temperature=data.get("temperature", 0.0),
        )


@dataclass
class AdaptiveStepConfig:
    """
    Configuration for adaptive routing step.

    Defines rules-first routing with optional LLM fallback.
    """

    decision_id: str
    """Unique identifier for this decision point."""

    rules: List[RoutingRule] = field(default_factory=list)
    """
    Rule-based routing conditions.

    Evaluated in order. First matching rule determines route.
    """

    llm_fallback: Optional[LLMFallbackConfig] = None
    """
    LLM fallback configuration.

    Used when no rules match. If None, uses default_route.
    """

    default_route: Optional[str] = None
    """
    Default route if nothing else matches.

    Used when:
    - No rules match AND
    - No LLM fallback configured OR LLM fails
    """

    checkpoint_before: bool = True
    """
    Create checkpoint before making decision.

    Allows resuming from this point if execution fails later.
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decision_id": self.decision_id,
            "rules": [r.to_dict() for r in self.rules],
            "llm_fallback": self.llm_fallback.to_dict() if self.llm_fallback else None,
            "default_route": self.default_route,
            "checkpoint_before": self.checkpoint_before,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdaptiveStepConfig":
        """Create from dictionary."""
        rules = [RoutingRule.from_dict(r) for r in data.get("rules", [])]
        llm_fallback = None
        if data.get("llm_fallback"):
            llm_fallback = LLMFallbackConfig.from_dict(data["llm_fallback"])

        return cls(
            decision_id=data["decision_id"],
            rules=rules,
            llm_fallback=llm_fallback,
            default_route=data.get("default_route"),
            checkpoint_before=data.get("checkpoint_before", True),
        )


@dataclass
class RoutingDecision:
    """
    Result of a routing decision.

    Contains target, method used, and optional LLM reasoning.
    """

    target: str
    """Target step ID to route to."""

    method: RoutingMethod
    """Method used for decision (rule/llm/default)."""

    # Rule-specific fields
    rule_condition: Optional[str] = None
    """Condition that matched (if method == RULE)."""

    rule_description: Optional[str] = None
    """Rule description (if method == RULE)."""

    # LLM-specific fields
    reasoning: Optional[str] = None
    """LLM reasoning (if method == LLM)."""

    confidence: Optional[float] = None
    """LLM confidence 0-1 (if method == LLM)."""

    # Metadata
    decision_id: str = ""
    """Decision point ID."""

    duration_ms: int = 0
    """Time taken to make decision."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target": self.target,
            "method": self.method.value,
            "rule_condition": self.rule_condition,
            "rule_description": self.rule_description,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "decision_id": self.decision_id,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoutingDecision":
        """Create from dictionary."""
        return cls(
            target=data["target"],
            method=RoutingMethod(data["method"]),
            rule_condition=data.get("rule_condition"),
            rule_description=data.get("rule_description"),
            reasoning=data.get("reasoning"),
            confidence=data.get("confidence"),
            decision_id=data.get("decision_id", ""),
            duration_ms=data.get("duration_ms", 0),
        )

    @classmethod
    def from_json(cls, json_str: str, decision_id: str = "") -> "RoutingDecision":
        """
        Parse LLM response JSON into RoutingDecision.

        Expected format:
        {"target": "<step_id>", "reasoning": "<why>", "confidence": 0.0-1.0}
        """
        try:
            # Try to extract JSON from response
            json_str = json_str.strip()
            # Handle markdown code blocks
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            data = json.loads(json_str)
            return cls(
                target=data.get("target", ""),
                method=RoutingMethod.LLM,
                reasoning=data.get("reasoning", ""),
                confidence=data.get("confidence"),
                decision_id=decision_id,
            )
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            # Return empty decision on parse error
            return cls(
                target="",
                method=RoutingMethod.LLM,
                reasoning=f"Failed to parse LLM response: {e}",
                confidence=0.0,
                decision_id=decision_id,
            )


@dataclass
class AdaptiveDecisionEvent:
    """
    Event emitted when adaptive routing decision is made.

    Used for audit logging and debugging.
    """

    event_type: str = "adaptive_decision"

    # Decision metadata
    step_id: str = ""
    """Step ID where decision was made."""

    decision_id: str = ""
    """Decision point identifier."""

    # Routing result
    target: str = ""
    """Target step routed to."""

    method: str = ""
    """Method used: rule/llm/default."""

    # Rule details (if method == rule)
    rule_condition: Optional[str] = None
    """Matched rule condition."""

    # LLM details (if method == llm)
    llm_reasoning: Optional[str] = None
    """LLM reasoning for decision."""

    llm_confidence: Optional[float] = None
    """LLM confidence score."""

    # Timing
    timestamp: datetime = field(default_factory=datetime.now)
    """When decision was made."""

    duration_ms: int = 0
    """Time taken to make decision."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "event_type": self.event_type,
            "step_id": self.step_id,
            "decision_id": self.decision_id,
            "target": self.target,
            "method": self.method,
            "rule_condition": self.rule_condition,
            "llm_reasoning": self.llm_reasoning,
            "llm_confidence": self.llm_confidence,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_decision(
        cls,
        decision: RoutingDecision,
        step_id: str = "",
    ) -> "AdaptiveDecisionEvent":
        """Create event from RoutingDecision."""
        return cls(
            step_id=step_id,
            decision_id=decision.decision_id,
            target=decision.target,
            method=decision.method.value,
            rule_condition=decision.rule_condition,
            llm_reasoning=decision.reasoning,
            llm_confidence=decision.confidence,
            duration_ms=decision.duration_ms,
        )


@dataclass
class CheckpointConfig:
    """
    Checkpoint configuration for adaptive routing.

    Controls when and how checkpoints are created.
    """

    before_adaptive: bool = True
    """Create checkpoint before each AdaptiveStep."""

    after_expensive: bool = True
    """Create checkpoint after expensive operations."""

    periodic_steps: int = 0
    """Create checkpoint every N steps (0 = disabled)."""

    ttl_hours: int = 24
    """Retention period for checkpoints in hours."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "before_adaptive": self.before_adaptive,
            "after_expensive": self.after_expensive,
            "periodic_steps": self.periodic_steps,
            "ttl_hours": self.ttl_hours,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointConfig":
        """Create from dictionary."""
        return cls(
            before_adaptive=data.get("before_adaptive", True),
            after_expensive=data.get("after_expensive", True),
            periodic_steps=data.get("periodic_steps", 0),
            ttl_hours=data.get("ttl_hours", 24),
        )
