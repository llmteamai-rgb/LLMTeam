"""
Tests for RFC-019: Budget Per-Period.
Tests for RFC-020: Retry-After from Providers.
"""

import time
import pytest
from unittest.mock import AsyncMock

from llmteam.cost.budget import (
    Budget,
    BudgetPeriod,
    BudgetStatus,
    BudgetExceededError,
    PeriodRecord,
    PeriodBudgetManager,
)


# ===== RFC-019: Period Budget Manager =====


class TestPeriodRecord:
    """Tests for PeriodRecord dataclass."""

    def test_creation(self):
        record = PeriodRecord(cost=1.5, timestamp=time.time(), run_id="run-1")
        assert record.cost == 1.5
        assert record.run_id == "run-1"

    def test_default_run_id(self):
        record = PeriodRecord(cost=0.5, timestamp=time.time())
        assert record.run_id is None


class TestPeriodBudgetManager:
    """Tests for PeriodBudgetManager."""

    def test_creation_empty(self):
        manager = PeriodBudgetManager()
        assert manager.budgets == {}

    def test_creation_with_budgets(self):
        budgets = {
            BudgetPeriod.HOUR: Budget(max_cost=50.0, period=BudgetPeriod.HOUR),
            BudgetPeriod.DAY: Budget(max_cost=200.0, period=BudgetPeriod.DAY),
        }
        manager = PeriodBudgetManager(budgets=budgets)
        assert len(manager.budgets) == 2

    def test_add_budget(self):
        manager = PeriodBudgetManager()
        manager.add_budget(Budget(max_cost=100.0, period=BudgetPeriod.HOUR))
        assert BudgetPeriod.HOUR in manager.budgets

    def test_record_cost(self):
        manager = PeriodBudgetManager(budgets={
            BudgetPeriod.HOUR: Budget(max_cost=10.0, period=BudgetPeriod.HOUR),
        })
        manager.record(cost=3.0, run_id="run-1")
        manager.record(cost=2.0, run_id="run-2")

        assert manager.get_period_cost(BudgetPeriod.HOUR) == 5.0

    def test_check_period_ok(self):
        manager = PeriodBudgetManager(budgets={
            BudgetPeriod.HOUR: Budget(max_cost=100.0, period=BudgetPeriod.HOUR),
        })
        manager.record(cost=10.0)

        assert manager.check_period(BudgetPeriod.HOUR) == BudgetStatus.OK

    def test_check_period_alert(self):
        manager = PeriodBudgetManager(budgets={
            BudgetPeriod.HOUR: Budget(max_cost=10.0, period=BudgetPeriod.HOUR, alert_threshold=0.8),
        })
        manager.record(cost=9.0)

        assert manager.check_period(BudgetPeriod.HOUR) == BudgetStatus.ALERT

    def test_check_period_exceeded_hard(self):
        manager = PeriodBudgetManager(budgets={
            BudgetPeriod.HOUR: Budget(max_cost=10.0, period=BudgetPeriod.HOUR, hard_limit=True),
        })
        manager.record(cost=11.0)

        assert manager.check_period(BudgetPeriod.HOUR) == BudgetStatus.EXCEEDED

    def test_check_period_exceeded_soft(self):
        manager = PeriodBudgetManager(budgets={
            BudgetPeriod.HOUR: Budget(max_cost=10.0, period=BudgetPeriod.HOUR, hard_limit=False),
        })
        manager.record(cost=11.0)

        assert manager.check_period(BudgetPeriod.HOUR) == BudgetStatus.WARNING

    def test_check_period_unknown_returns_ok(self):
        manager = PeriodBudgetManager()
        assert manager.check_period(BudgetPeriod.DAY) == BudgetStatus.OK

    def test_check_all(self):
        manager = PeriodBudgetManager(budgets={
            BudgetPeriod.HOUR: Budget(max_cost=100.0, period=BudgetPeriod.HOUR),
            BudgetPeriod.DAY: Budget(max_cost=500.0, period=BudgetPeriod.DAY),
        })
        manager.record(cost=5.0)

        result = manager.check_all()
        assert result[BudgetPeriod.HOUR] == BudgetStatus.OK
        assert result[BudgetPeriod.DAY] == BudgetStatus.OK

    def test_check_all_or_raise(self):
        manager = PeriodBudgetManager(budgets={
            BudgetPeriod.HOUR: Budget(max_cost=5.0, period=BudgetPeriod.HOUR, hard_limit=True),
        })
        manager.record(cost=10.0)

        with pytest.raises(BudgetExceededError):
            manager.check_all_or_raise()

    def test_remaining(self):
        manager = PeriodBudgetManager(budgets={
            BudgetPeriod.HOUR: Budget(max_cost=50.0, period=BudgetPeriod.HOUR),
        })
        manager.record(cost=20.0)

        assert manager.remaining(BudgetPeriod.HOUR) == 30.0

    def test_remaining_no_budget(self):
        manager = PeriodBudgetManager()
        assert manager.remaining(BudgetPeriod.DAY) == float("inf")

    def test_reset(self):
        manager = PeriodBudgetManager(budgets={
            BudgetPeriod.HOUR: Budget(max_cost=10.0, period=BudgetPeriod.HOUR),
        })
        manager.record(cost=5.0)
        assert manager.get_period_cost(BudgetPeriod.HOUR) == 5.0

        manager.reset()
        assert manager.get_period_cost(BudgetPeriod.HOUR) == 0.0

    def test_alert_callback(self):
        alerts = []
        manager = PeriodBudgetManager(budgets={
            BudgetPeriod.HOUR: Budget(max_cost=10.0, period=BudgetPeriod.HOUR, alert_threshold=0.5),
        })
        manager.on_alert(lambda period, cost, max_c: alerts.append((period, cost, max_c)))

        manager.record(cost=6.0)
        manager.check_period(BudgetPeriod.HOUR)

        assert len(alerts) == 1
        assert alerts[0][0] == BudgetPeriod.HOUR
        assert alerts[0][1] == 6.0

    def test_alert_fires_once(self):
        alerts = []
        manager = PeriodBudgetManager(budgets={
            BudgetPeriod.HOUR: Budget(max_cost=10.0, period=BudgetPeriod.HOUR, alert_threshold=0.5),
        })
        manager.on_alert(lambda p, c, m: alerts.append(True))

        manager.record(cost=6.0)
        manager.check_period(BudgetPeriod.HOUR)
        manager.check_period(BudgetPeriod.HOUR)

        assert len(alerts) == 1  # Only fired once

    def test_to_dict(self):
        manager = PeriodBudgetManager(budgets={
            BudgetPeriod.HOUR: Budget(max_cost=50.0, period=BudgetPeriod.HOUR),
        })
        manager.record(cost=10.0)

        data = manager.to_dict()
        assert "hour" in data
        assert data["hour"]["max_cost"] == 50.0
        assert data["hour"]["current_cost"] == 10.0
        assert data["hour"]["remaining"] == 40.0
        assert data["hour"]["status"] == "ok"

    def test_run_period_returns_zero(self):
        """RUN period cost is handled separately, returns 0."""
        manager = PeriodBudgetManager(budgets={
            BudgetPeriod.RUN: Budget(max_cost=5.0, period=BudgetPeriod.RUN),
        })
        manager.record(cost=3.0)
        assert manager.get_period_cost(BudgetPeriod.RUN) == 0.0

    def test_old_records_excluded(self):
        """Records older than period window should be excluded."""
        manager = PeriodBudgetManager(budgets={
            BudgetPeriod.HOUR: Budget(max_cost=100.0, period=BudgetPeriod.HOUR),
        })
        # Add old record (2 hours ago)
        old_record = PeriodRecord(cost=50.0, timestamp=time.time() - 7200)
        manager._records.append(old_record)

        # Add recent record
        manager.record(cost=10.0)

        assert manager.get_period_cost(BudgetPeriod.HOUR) == 10.0


# ===== RFC-020: Retry-After from Providers =====


class TestRetryAfterFromProviders:
    """Tests for RFC-020: retry_after support in AgentRetryExecutor."""

    async def test_retry_after_used_when_present(self):
        """When exception has retry_after, it should be used as delay."""
        from llmteam.agents.retry import AgentRetryExecutor, RetryPolicy
        from llmteam.providers.base import LLMRateLimitError

        executor = AgentRetryExecutor(
            agent_id="test",
            retry_policy=RetryPolicy(
                max_retries=2,
                backoff="exponential",
                base_delay=1.0,
                retryable_exceptions=(LLMRateLimitError,),
            ),
        )

        call_count = 0

        async def failing_func(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise LLMRateLimitError(
                    message="Rate limited",
                    provider="openai",
                    retry_after=0.01,  # Very short for testing
                )
            return "success"

        import asyncio
        start = asyncio.get_event_loop().time()
        result, metrics = await executor.execute(failing_func)
        elapsed = asyncio.get_event_loop().time() - start

        assert result == "success"
        assert metrics.total_attempts == 3
        # Delay should be based on retry_after (0.01s), not exponential (1.0s)
        assert elapsed < 1.0  # Would be >2s with exponential backoff

    async def test_retry_after_none_uses_strategy(self):
        """When retry_after is None, normal backoff strategy is used."""
        from llmteam.agents.retry import AgentRetryExecutor, RetryPolicy

        executor = AgentRetryExecutor(
            agent_id="test",
            retry_policy=RetryPolicy(
                max_retries=1,
                backoff="constant",
                base_delay=0.01,
                retryable_exceptions=(ValueError,),
            ),
        )

        call_count = 0

        async def failing_func(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("fail")  # No retry_after attribute
            return "ok"

        result, metrics = await executor.execute(failing_func)
        assert result == "ok"
        assert metrics.total_attempts == 2

    async def test_retry_after_zero_ignored(self):
        """retry_after=0 should be ignored (use normal backoff)."""
        from llmteam.agents.retry import AgentRetryExecutor, RetryPolicy
        from llmteam.providers.base import LLMRateLimitError

        executor = AgentRetryExecutor(
            agent_id="test",
            retry_policy=RetryPolicy(
                max_retries=1,
                backoff="constant",
                base_delay=0.01,
                retryable_exceptions=(LLMRateLimitError,),
            ),
        )

        call_count = 0

        async def failing_func(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise LLMRateLimitError(retry_after=0)  # Zero should be ignored
            return "ok"

        result, metrics = await executor.execute(failing_func)
        assert result == "ok"

    async def test_retry_after_negative_ignored(self):
        """Negative retry_after should be ignored."""
        from llmteam.agents.retry import AgentRetryExecutor, RetryPolicy
        from llmteam.providers.base import LLMRateLimitError

        executor = AgentRetryExecutor(
            agent_id="test",
            retry_policy=RetryPolicy(
                max_retries=1,
                backoff="constant",
                base_delay=0.01,
                retryable_exceptions=(LLMRateLimitError,),
            ),
        )

        call_count = 0

        async def failing_func(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise LLMRateLimitError(retry_after=-5.0)
            return "ok"

        result, metrics = await executor.execute(failing_func)
        assert result == "ok"
