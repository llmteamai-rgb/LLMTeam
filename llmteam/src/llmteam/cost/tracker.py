"""
Cost tracker for real-time token usage and cost monitoring.

RFC-010: Cost Tracking & Budget Management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from llmteam.cost.pricing import PricingRegistry


@dataclass
class TokenUsage:
    """Token usage for a single LLM call."""

    input_tokens: int
    output_tokens: int
    model: str
    cost_usd: float
    agent_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "cost_usd": round(self.cost_usd, 6),
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RunCost:
    """Aggregated cost for a single run."""

    run_id: str
    team_id: str
    total_cost: float = 0.0
    token_usage: List[TokenUsage] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    @property
    def total_tokens(self) -> int:
        """Total tokens across all calls."""
        return sum(u.total_tokens for u in self.token_usage)

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens."""
        return sum(u.input_tokens for u in self.token_usage)

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens."""
        return sum(u.output_tokens for u in self.token_usage)

    @property
    def calls_count(self) -> int:
        """Number of LLM calls."""
        return len(self.token_usage)

    @property
    def duration_ms(self) -> Optional[int]:
        """Run duration in milliseconds."""
        if self.completed_at and self.started_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "run_id": self.run_id,
            "team_id": self.team_id,
            "total_cost": round(self.total_cost, 6),
            "total_tokens": self.total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "calls_count": self.calls_count,
            "duration_ms": self.duration_ms,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class CostTracker:
    """
    Real-time cost tracking during execution.

    Tracks token usage and costs per run, maintains history.

    Example:
        tracker = CostTracker()
        tracker.start_run("run-1", "my_team")
        tracker.record_usage("gpt-4o", input_tokens=500, output_tokens=200)
        tracker.record_usage("gpt-4o-mini", input_tokens=100, output_tokens=50)
        run_cost = tracker.end_run()
        print(f"Total cost: ${run_cost.total_cost:.4f}")
    """

    def __init__(
        self,
        pricing: Optional[PricingRegistry] = None,
        on_usage: Optional[Callable[["TokenUsage"], None]] = None,
    ):
        """
        Initialize cost tracker.

        Args:
            pricing: Custom pricing registry (default: built-in pricing)
            on_usage: Optional callback fired on each token usage record
        """
        self._pricing = pricing or PricingRegistry()
        self._on_usage = on_usage
        self._current_run: Optional[RunCost] = None
        self._history: List[RunCost] = []

    @property
    def current_cost(self) -> float:
        """Get current run cost."""
        return self._current_run.total_cost if self._current_run else 0.0

    @property
    def current_run(self) -> Optional[RunCost]:
        """Get current RunCost (None if no active run)."""
        return self._current_run

    @property
    def history(self) -> List[RunCost]:
        """Get completed run history."""
        return self._history.copy()

    @property
    def total_spent(self) -> float:
        """Total cost across all completed runs."""
        return sum(r.total_cost for r in self._history)

    def start_run(self, run_id: str, team_id: str) -> None:
        """
        Start tracking a new run.

        Args:
            run_id: Run identifier
            team_id: Team identifier
        """
        self._current_run = RunCost(
            run_id=run_id,
            team_id=team_id,
        )

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        agent_id: Optional[str] = None,
    ) -> TokenUsage:
        """
        Record token usage from an LLM call.

        Args:
            model: Model name (e.g., "gpt-4o")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            agent_id: Optional agent ID that made the call

        Returns:
            TokenUsage record with calculated cost
        """
        cost = self._pricing.calculate_cost(model, input_tokens, output_tokens)

        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            cost_usd=cost,
            agent_id=agent_id,
        )

        if self._current_run:
            self._current_run.token_usage.append(usage)
            self._current_run.total_cost += cost

        # Fire callback
        if self._on_usage:
            self._on_usage(usage)

        return usage

    def end_run(self) -> RunCost:
        """
        End current run and return cost summary.

        Returns:
            RunCost with aggregated data

        Raises:
            RuntimeError: If no active run
        """
        if not self._current_run:
            raise RuntimeError("No active run to end")

        self._current_run.completed_at = datetime.utcnow()
        result = self._current_run
        self._history.append(result)
        self._current_run = None
        return result

    def reset_history(self) -> None:
        """Clear run history."""
        self._history.clear()
