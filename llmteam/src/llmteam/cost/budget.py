"""
Budget management for cost control.

RFC-010: Cost Tracking & Budget Management.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class BudgetStatus(Enum):
    """Budget check result."""

    OK = "ok"  # Under budget
    ALERT = "alert"  # Approaching limit (threshold reached)
    EXCEEDED = "exceeded"  # Over budget (hard limit)
    WARNING = "warning"  # Over budget (soft limit, no hard stop)


class BudgetPeriod(Enum):
    """Budget period type."""

    RUN = "run"  # Per single run
    HOUR = "hour"  # Per hour
    DAY = "day"  # Per day
    MONTH = "month"  # Per month


@dataclass
class Budget:
    """
    Budget configuration.

    Example:
        budget = Budget(max_cost=5.0, period=BudgetPeriod.RUN)
        budget = Budget(max_cost=100.0, period=BudgetPeriod.DAY, alert_threshold=0.8)
    """

    max_cost: float  # Maximum cost in USD
    period: BudgetPeriod = BudgetPeriod.RUN
    alert_threshold: float = 0.8  # Alert at 80% of max_cost
    hard_limit: bool = True  # If True, stop execution on exceed

    def __post_init__(self):
        if self.max_cost <= 0:
            raise ValueError("max_cost must be positive")
        if not (0.0 < self.alert_threshold <= 1.0):
            raise ValueError("alert_threshold must be between 0 and 1")


class BudgetExceededError(Exception):
    """Raised when budget hard limit is exceeded."""

    def __init__(self, current_cost: float, max_cost: float):
        self.current_cost = current_cost
        self.max_cost = max_cost
        super().__init__(
            f"Budget exceeded: ${current_cost:.4f} / ${max_cost:.4f}"
        )


class BudgetManager:
    """
    Manages cost budgets and alerts.

    Checks current spending against budget limits.
    Fires alert callbacks when thresholds are reached.

    Example:
        budget = Budget(max_cost=5.0)
        manager = BudgetManager(budget)
        manager.on_alert(lambda cost, max_c: print(f"Alert: ${cost:.2f}/${max_c:.2f}"))

        status = manager.check(current_cost=4.5)
        # status == BudgetStatus.ALERT
    """

    def __init__(self, budget: Budget):
        self._budget = budget
        self._alert_callbacks: List[Callable[[float, float], None]] = []
        self._alerted = False  # Track if alert already fired

    @property
    def budget(self) -> Budget:
        """Get budget configuration."""
        return self._budget

    @property
    def max_cost(self) -> float:
        """Maximum allowed cost."""
        return self._budget.max_cost

    def on_alert(self, callback: Callable[[float, float], None]) -> None:
        """
        Register alert callback.

        Callback receives (current_cost, max_cost) when threshold is reached.

        Args:
            callback: Alert function
        """
        self._alert_callbacks.append(callback)

    def check(self, current_cost: float) -> BudgetStatus:
        """
        Check if budget allows continuation.

        Args:
            current_cost: Current accumulated cost

        Returns:
            BudgetStatus indicating budget state
        """
        if current_cost <= 0:
            return BudgetStatus.OK

        ratio = current_cost / self._budget.max_cost

        if ratio >= 1.0:
            if self._budget.hard_limit:
                return BudgetStatus.EXCEEDED
            return BudgetStatus.WARNING

        if ratio >= self._budget.alert_threshold:
            if not self._alerted:
                self._alerted = True
                for callback in self._alert_callbacks:
                    callback(current_cost, self._budget.max_cost)
            return BudgetStatus.ALERT

        return BudgetStatus.OK

    def check_or_raise(self, current_cost: float) -> BudgetStatus:
        """
        Check budget and raise if exceeded (hard limit).

        Args:
            current_cost: Current accumulated cost

        Returns:
            BudgetStatus

        Raises:
            BudgetExceededError: If hard limit exceeded
        """
        status = self.check(current_cost)
        if status == BudgetStatus.EXCEEDED:
            raise BudgetExceededError(current_cost, self._budget.max_cost)
        return status

    def reset(self) -> None:
        """Reset alert state (for new period)."""
        self._alerted = False

    def remaining(self, current_cost: float) -> float:
        """
        Get remaining budget.

        Args:
            current_cost: Current accumulated cost

        Returns:
            Remaining budget in USD (can be negative)
        """
        return self._budget.max_cost - current_cost

    def to_dict(self, current_cost: float = 0.0) -> Dict[str, Any]:
        """Serialize budget state."""
        return {
            "max_cost": self._budget.max_cost,
            "period": self._budget.period.value,
            "current_cost": round(current_cost, 6),
            "remaining": round(self.remaining(current_cost), 6),
            "status": self.check(current_cost).value,
            "hard_limit": self._budget.hard_limit,
            "alert_threshold": self._budget.alert_threshold,
        }
