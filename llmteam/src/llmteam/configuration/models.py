"""
Configuration models for CONFIGURATOR mode (RFC-005).

Provides data classes for team configuration sessions.
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "main_goal": self.main_goal,
            "input_type": self.input_type,
            "output_type": self.output_type,
            "sub_tasks": self.sub_tasks,
            "complexity": self.complexity,
            "raw_analysis": self.raw_analysis,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskAnalysis":
        """Create from dictionary."""
        return cls(
            main_goal=data.get("main_goal", ""),
            input_type=data.get("input_type", ""),
            output_type=data.get("output_type", ""),
            sub_tasks=data.get("sub_tasks", []),
            complexity=data.get("complexity", "moderate"),
            raw_analysis=data.get("raw_analysis", ""),
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
