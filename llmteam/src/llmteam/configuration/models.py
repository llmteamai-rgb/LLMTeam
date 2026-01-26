"""
Configuration models for CONFIGURATOR mode (RFC-005, RFC-023).

Provides data classes for team configuration sessions.

RFC-023: Added decision point analysis and routing configuration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class SessionState(Enum):
    """State of a configuration session."""

    CREATED = "created"
    ANALYZING = "analyzing"
    SUGGESTING = "suggesting"
    ROUTING_CONFIG = "routing_config"  # RFC-023: Routing preferences
    CONFIGURING = "configuring"
    TESTING = "testing"
    READY = "ready"
    APPLIED = "applied"


@dataclass
class AgentSuggestion:
    """
    Agent suggestion from CONFIGURATOR.

    Represents a proposed agent configuration based on task analysis.
    """

    role: str
    """Unique role name for the agent."""

    type: str
    """Agent type: 'llm', 'rag', or 'kag'."""

    purpose: str
    """What the agent does."""

    prompt_template: str
    """Initial prompt template for the agent."""

    reasoning: str
    """Why this agent is needed."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "type": self.type,
            "purpose": self.purpose,
            "prompt_template": self.prompt_template,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentSuggestion":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            type=data["type"],
            purpose=data.get("purpose", ""),
            prompt_template=data.get("prompt_template", ""),
            reasoning=data.get("reasoning", ""),
        )


@dataclass
class TestRunResult:
    """
    Result of a test run during configuration.

    Contains both execution results and LLM analysis.
    """

    test_id: str
    """Unique test run identifier."""

    input_data: Dict[str, Any]
    """Input data used for the test."""

    output: Dict[str, Any]
    """Final output from the team."""

    agent_outputs: Dict[str, Any]
    """Outputs from individual agents."""

    duration_ms: int
    """Execution duration in milliseconds."""

    success: bool
    """Whether the test run succeeded."""

    # LLM analysis
    analysis: str = ""
    """LLM analysis of the test run."""

    issues: List[str] = field(default_factory=list)
    """Issues found during analysis."""

    recommendations: List[str] = field(default_factory=list)
    """Recommendations for improvement."""

    ready_for_production: bool = False
    """Whether the configuration is ready for production."""

    created_at: datetime = field(default_factory=datetime.utcnow)
    """When the test was run."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "input_data": self.input_data,
            "output": self.output,
            "agent_outputs": self.agent_outputs,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "analysis": self.analysis,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "ready_for_production": self.ready_for_production,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestRunResult":
        """Create from dictionary."""
        result = cls(
            test_id=data["test_id"],
            input_data=data.get("input_data", {}),
            output=data.get("output", {}),
            agent_outputs=data.get("agent_outputs", {}),
            duration_ms=data.get("duration_ms", 0),
            success=data.get("success", False),
            analysis=data.get("analysis", ""),
            issues=data.get("issues", []),
            recommendations=data.get("recommendations", []),
            ready_for_production=data.get("ready_for_production", False),
        )
        if data.get("created_at"):
            result.created_at = datetime.fromisoformat(data["created_at"])
        return result


@dataclass
class TaskAnalysis:
    """
    Analysis of a user's task.

    Extracted by LLM from the task description.
    RFC-023: Added decision_points field.
    """

    main_goal: str
    """Primary goal of the task."""

    input_type: str
    """Type of input expected."""

    output_type: str
    """Type of output expected."""

    sub_tasks: List[str]
    """Sub-tasks needed to complete the goal."""

    complexity: str
    """Complexity level: 'simple', 'moderate', or 'complex'."""

    raw_analysis: str = ""
    """Raw LLM analysis text."""

    # RFC-023: Decision points
    domain: str = ""
    """Task domain (e.g., 'content creation', 'data analysis')."""

    required_capabilities: List[str] = field(default_factory=list)
    """Required capabilities (e.g., 'research', 'writing')."""

    decision_points: List["DecisionPointAnalysis"] = field(default_factory=list)
    """Identified decision points where routing is needed."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "main_goal": self.main_goal,
            "input_type": self.input_type,
            "output_type": self.output_type,
            "sub_tasks": self.sub_tasks,
            "complexity": self.complexity,
            "raw_analysis": self.raw_analysis,
            "domain": self.domain,
            "required_capabilities": self.required_capabilities,
            "decision_points": [dp.to_dict() for dp in self.decision_points],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskAnalysis":
        """Create from dictionary."""
        # Import here to avoid circular import
        decision_points = []
        for dp_data in data.get("decision_points", []):
            decision_points.append(DecisionPointAnalysis.from_dict(dp_data))

        return cls(
            main_goal=data.get("main_goal", ""),
            input_type=data.get("input_type", ""),
            output_type=data.get("output_type", ""),
            sub_tasks=data.get("sub_tasks", []),
            complexity=data.get("complexity", "moderate"),
            raw_analysis=data.get("raw_analysis", ""),
            domain=data.get("domain", ""),
            required_capabilities=data.get("required_capabilities", []),
            decision_points=decision_points,
        )


@dataclass
class PipelinePreview:
    """
    Preview of pipeline configuration (RFC-008).

    Shows estimated cost and quality for current configuration.
    """

    quality: int
    """Quality level (0-100)."""

    agents: List[Dict[str, Any]]
    """Agent configurations."""

    flow: Optional[str]
    """Flow definition."""

    estimated_cost_min: float
    """Minimum estimated cost in USD."""

    estimated_cost_max: float
    """Maximum estimated cost in USD."""

    quality_stars: int
    """Quality rating 1-5 stars."""

    quality_label: str
    """Quality label (e.g., "Good", "Excellent")."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "quality": self.quality,
            "agents": self.agents,
            "flow": self.flow,
            "estimated_cost": {
                "min": self.estimated_cost_min,
                "max": self.estimated_cost_max,
            },
            "quality_rating": {
                "stars": self.quality_stars,
                "label": self.quality_label,
            },
        }

    @property
    def estimated_cost(self) -> str:
        """Formatted estimated cost string."""
        return f"${self.estimated_cost_min:.2f} - ${self.estimated_cost_max:.2f}"

    def __str__(self) -> str:
        """Human-readable preview."""
        stars = "⭐" * self.quality_stars
        agents_str = "\n".join(
            f"  {i+1}. {a['role']} ({a.get('model', 'default')}) — {a.get('prompt', '')[:50]}..."
            for i, a in enumerate(self.agents)
        )
        return f"""Pipeline Preview (quality={self.quality}):

Agents:
{agents_str}

Flow: {self.flow or 'sequential'}

Estimated cost: {self.estimated_cost} per run
Estimated quality: {stars} ({self.quality_label})
"""


# =============================================================================
# RFC-023: Decision Point Models
# =============================================================================


@dataclass
class DecisionPointAnalysis:
    """
    Analysis of a decision point in the task (RFC-023).

    Identifies where routing decisions need to be made.
    """

    after_step: str
    """Step after which this decision occurs."""

    decision_type: str
    """Type: 'complexity', 'quality_gate', 'branch', 'retry'."""

    description: str
    """Human-readable description of the decision."""

    rule_feasible: bool = True
    """Can this be decided by deterministic rules?"""

    suggested_rules: List[str] = field(default_factory=list)
    """Suggested rule conditions."""

    needs_llm_fallback: bool = True
    """Does this need LLM fallback?"""

    possible_routes: List[str] = field(default_factory=list)
    """Possible target steps."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "after_step": self.after_step,
            "decision_type": self.decision_type,
            "description": self.description,
            "rule_feasible": self.rule_feasible,
            "suggested_rules": self.suggested_rules,
            "needs_llm_fallback": self.needs_llm_fallback,
            "possible_routes": self.possible_routes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionPointAnalysis":
        """Create from dictionary."""
        return cls(
            after_step=data.get("after_step", ""),
            decision_type=data.get("decision_type", "branch"),
            description=data.get("description", ""),
            rule_feasible=data.get("rule_feasible", True),
            suggested_rules=data.get("suggested_rules", []),
            needs_llm_fallback=data.get("needs_llm_fallback", True),
            possible_routes=data.get("possible_routes", []),
        )


@dataclass
class RoutingRuleConfig:
    """
    Rule for deterministic routing (RFC-023).

    Evaluated against step output.
    """

    condition: str
    """Condition expression: 'output.score >= 80'."""

    target: str
    """Target step if condition matches."""

    description: str = ""
    """Human-readable description."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "condition": self.condition,
            "target": self.target,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoutingRuleConfig":
        """Create from dictionary."""
        return cls(
            condition=data.get("condition", ""),
            target=data.get("target", ""),
            description=data.get("description", ""),
        )


@dataclass
class RouteConfig:
    """
    Possible route from a decision point (RFC-023).

    Describes when to use this route.
    """

    target: str
    """Target step ID."""

    description: str
    """Human-readable description."""

    when: str = ""
    """Hint when this route is appropriate."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target": self.target,
            "description": self.description,
            "when": self.when,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RouteConfig":
        """Create from dictionary."""
        return cls(
            target=data.get("target", ""),
            description=data.get("description", ""),
            when=data.get("when", ""),
        )


@dataclass
class LLMFallbackConfigData:
    """
    LLM fallback configuration for routing (RFC-023).

    Used when rules can't decide.
    """

    model: str = "gpt-4o-mini"
    """Model for routing decision."""

    prompt: str = ""
    """System prompt for routing."""

    routes: List[RouteConfig] = field(default_factory=list)
    """Available routes to choose from."""

    max_tokens: int = 100
    """Max tokens for decision."""

    temperature: float = 0.0
    """Temperature (0 for determinism)."""

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
    def from_dict(cls, data: Dict[str, Any]) -> "LLMFallbackConfigData":
        """Create from dictionary."""
        routes = [RouteConfig.from_dict(r) for r in data.get("routes", [])]
        return cls(
            model=data.get("model", "gpt-4o-mini"),
            prompt=data.get("prompt", ""),
            routes=routes,
            max_tokens=data.get("max_tokens", 100),
            temperature=data.get("temperature", 0.0),
        )


@dataclass
class DecisionPointConfig:
    """
    Configuration for an adaptive routing point (RFC-023).

    Defines rules-first routing with optional LLM fallback.
    """

    decision_id: str
    """Unique identifier for this decision point."""

    after_step: str
    """Step after which this decision occurs."""

    decision_type: str
    """Type: 'complexity', 'quality_gate', 'branch', 'retry'."""

    rules: List[RoutingRuleConfig] = field(default_factory=list)
    """Rule-based routing conditions."""

    llm_fallback: Optional[LLMFallbackConfigData] = None
    """LLM fallback configuration."""

    routes: List[RouteConfig] = field(default_factory=list)
    """All possible routes."""

    default_route: Optional[str] = None
    """Default route if nothing matches."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision_id": self.decision_id,
            "after_step": self.after_step,
            "decision_type": self.decision_type,
            "rules": [r.to_dict() for r in self.rules],
            "llm_fallback": self.llm_fallback.to_dict() if self.llm_fallback else None,
            "routes": [r.to_dict() for r in self.routes],
            "default_route": self.default_route,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionPointConfig":
        """Create from dictionary."""
        rules = [RoutingRuleConfig.from_dict(r) for r in data.get("rules", [])]
        llm_fallback = None
        if data.get("llm_fallback"):
            llm_fallback = LLMFallbackConfigData.from_dict(data["llm_fallback"])
        routes = [RouteConfig.from_dict(r) for r in data.get("routes", [])]

        return cls(
            decision_id=data.get("decision_id", ""),
            after_step=data.get("after_step", ""),
            decision_type=data.get("decision_type", "branch"),
            rules=rules,
            llm_fallback=llm_fallback,
            routes=routes,
            default_route=data.get("default_route"),
        )


@dataclass
class ConfiguratorCostEstimate:
    """
    Cost estimate from Configurator (RFC-023).

    Includes adaptive decision costs.
    """

    min_cost: float
    """Minimum estimated cost."""

    max_cost: float
    """Maximum estimated cost."""

    expected_cost: float
    """Expected cost."""

    breakdown: Dict[str, float] = field(default_factory=dict)
    """Cost breakdown by component."""

    adaptive_decisions: int = 0
    """Expected LLM routing calls."""

    adaptive_cost: float = 0.0
    """Cost of routing decisions."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_cost": self.min_cost,
            "max_cost": self.max_cost,
            "expected_cost": self.expected_cost,
            "breakdown": self.breakdown,
            "adaptive_decisions": self.adaptive_decisions,
            "adaptive_cost": self.adaptive_cost,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfiguratorCostEstimate":
        """Create from dictionary."""
        return cls(
            min_cost=data.get("min_cost", 0.0),
            max_cost=data.get("max_cost", 0.0),
            expected_cost=data.get("expected_cost", 0.0),
            breakdown=data.get("breakdown", {}),
            adaptive_decisions=data.get("adaptive_decisions", 0),
            adaptive_cost=data.get("adaptive_cost", 0.0),
        )


@dataclass
class ConfiguratorOutput:
    """
    Output from Configurator (RFC-023).

    Complete team configuration with decision points.
    """

    analysis: "TaskAnalysis"
    """Task analysis."""

    agents: List[Dict[str, Any]]
    """Agent configurations."""

    flow: str
    """Flow notation."""

    decision_points: List[DecisionPointConfig] = field(default_factory=list)
    """Decision points for adaptive routing."""

    orchestration: Dict[str, Any] = field(default_factory=dict)
    """Orchestration configuration."""

    estimated_cost: Optional[ConfiguratorCostEstimate] = None
    """Cost estimate."""

    reasoning: str = ""
    """Reasoning behind the configuration."""

    warnings: List[str] = field(default_factory=list)
    """Warnings about the configuration."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "analysis": self.analysis.to_dict() if self.analysis else {},
            "agents": self.agents,
            "flow": self.flow,
            "decision_points": [dp.to_dict() for dp in self.decision_points],
            "orchestration": self.orchestration,
            "estimated_cost": self.estimated_cost.to_dict() if self.estimated_cost else None,
            "reasoning": self.reasoning,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfiguratorOutput":
        """Create from dictionary."""
        analysis = TaskAnalysis.from_dict(data.get("analysis", {}))
        decision_points = [
            DecisionPointConfig.from_dict(dp)
            for dp in data.get("decision_points", [])
        ]
        estimated_cost = None
        if data.get("estimated_cost"):
            estimated_cost = ConfiguratorCostEstimate.from_dict(data["estimated_cost"])

        return cls(
            analysis=analysis,
            agents=data.get("agents", []),
            flow=data.get("flow", ""),
            decision_points=decision_points,
            orchestration=data.get("orchestration", {}),
            estimated_cost=estimated_cost,
            reasoning=data.get("reasoning", ""),
            warnings=data.get("warnings", []),
        )
