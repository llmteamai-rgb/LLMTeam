"""
Tests for RFC-012: Per-Agent Retry Policies & Circuit Breaker Integration.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from llmteam import (
    LLMTeam,
    AgentConfig,
    LLMAgentConfig,
    RetryPolicy,
    CircuitBreakerPolicy,
    RetryMetrics,
    AgentRetryExecutor,
)
from llmteam.agents.retry import RetryPolicy, CircuitBreakerPolicy, RetryMetrics
from llmteam.clients.retry import ExponentialBackoff, LinearBackoff, ConstantBackoff
from llmteam.clients.circuit_breaker import CircuitBreakerOpen, CircuitState


class TestRetryPolicy:
    """Tests for RetryPolicy dataclass."""

    def test_defaults(self):
        """RetryPolicy should have sensible defaults."""
        policy = RetryPolicy()

        assert policy.max_retries == 3
        assert policy.backoff == "exponential"
        assert policy.base_delay == 1.0
        assert policy.max_delay == 30.0
        assert policy.multiplier == 2.0
        assert policy.jitter == 0.1
        assert policy.retryable_exceptions == (Exception,)
        assert policy.on_retry is None

    def test_custom_values(self):
        """RetryPolicy should accept custom values."""
        policy = RetryPolicy(
            max_retries=5,
            backoff="linear",
            base_delay=2.0,
            max_delay=60.0,
            jitter=0.2,
            retryable_exceptions=(TimeoutError, ConnectionError),
        )

        assert policy.max_retries == 5
        assert policy.backoff == "linear"
        assert policy.base_delay == 2.0
        assert policy.max_delay == 60.0
        assert policy.jitter == 0.2
        assert policy.retryable_exceptions == (TimeoutError, ConnectionError)

    def test_build_exponential_strategy(self):
        """build_strategy should create ExponentialBackoff."""
        policy = RetryPolicy(backoff="exponential", base_delay=1.0, multiplier=2.0)
        strategy = policy.build_strategy()

        assert isinstance(strategy, ExponentialBackoff)
        assert strategy.base_delay == 1.0
        assert strategy.multiplier == 2.0

    def test_build_linear_strategy(self):
        """build_strategy should create LinearBackoff."""
        policy = RetryPolicy(backoff="linear", base_delay=2.0)
        strategy = policy.build_strategy()

        assert isinstance(strategy, LinearBackoff)
        assert strategy.base_delay == 2.0

    def test_build_constant_strategy(self):
        """build_strategy should create ConstantBackoff."""
        policy = RetryPolicy(backoff="constant", base_delay=5.0)
        strategy = policy.build_strategy()

        assert isinstance(strategy, ConstantBackoff)
        assert strategy.delay == 5.0


class TestCircuitBreakerPolicy:
    """Tests for CircuitBreakerPolicy dataclass."""

    def test_defaults(self):
        """CircuitBreakerPolicy should have sensible defaults."""
        policy = CircuitBreakerPolicy()

        assert policy.failure_threshold == 5
        assert policy.recovery_timeout_seconds == 30.0
        assert policy.success_threshold == 2
        assert policy.failure_window_seconds == 60.0
        assert policy.failure_exceptions == (Exception,)
        assert policy.on_state_change is None

    def test_custom_values(self):
        """CircuitBreakerPolicy should accept custom values."""
        policy = CircuitBreakerPolicy(
            failure_threshold=3,
            recovery_timeout_seconds=10.0,
            success_threshold=1,
        )

        assert policy.failure_threshold == 3
        assert policy.recovery_timeout_seconds == 10.0
        assert policy.success_threshold == 1

    def test_build_circuit_breaker(self):
        """build_circuit_breaker should create CircuitBreaker instance."""
        policy = CircuitBreakerPolicy(
            failure_threshold=3,
            recovery_timeout_seconds=15.0,
        )

        cb = policy.build_circuit_breaker(name="test")

        assert cb is not None
        assert cb.state == CircuitState.CLOSED
        assert cb.config.failure_threshold == 3

    def test_on_state_change_callback(self):
        """CircuitBreakerPolicy should pass callback to CircuitBreaker."""
        changes = []

        def on_change(old, new):
            changes.append((old, new))

        policy = CircuitBreakerPolicy(on_state_change=on_change)
        cb = policy.build_circuit_breaker()

        assert cb.config.on_state_change is on_change


class TestRetryMetrics:
    """Tests for RetryMetrics dataclass."""

    def test_defaults(self):
        """RetryMetrics should have zero defaults."""
        metrics = RetryMetrics()

        assert metrics.total_attempts == 0
        assert metrics.successful_attempt == 0
        assert metrics.retries_performed == 0
        assert metrics.total_retry_delay_ms == 0.0
        assert metrics.last_error is None
        assert metrics.circuit_breaker_state is None
        assert metrics.circuit_breaker_blocked is False

    def test_to_dict(self):
        """RetryMetrics.to_dict should serialize all fields."""
        metrics = RetryMetrics(
            total_attempts=3,
            successful_attempt=2,
            retries_performed=2,
            total_retry_delay_ms=3500.0,
            last_error="Connection timeout",
            circuit_breaker_state="closed",
            circuit_breaker_blocked=False,
        )

        data = metrics.to_dict()

        assert data["total_attempts"] == 3
        assert data["successful_attempt"] == 2
        assert data["retries_performed"] == 2
        assert data["total_retry_delay_ms"] == 3500.0
        assert data["last_error"] == "Connection timeout"
        assert data["circuit_breaker_state"] == "closed"
        assert data["circuit_breaker_blocked"] is False


class TestAgentRetryExecutor:
    """Tests for AgentRetryExecutor."""

    def test_init_with_retry_only(self):
        """Executor should work with retry policy only."""
        executor = AgentRetryExecutor(
            agent_id="test",
            retry_policy=RetryPolicy(max_retries=3),
        )

        assert executor.has_retry is True
        assert executor.has_circuit_breaker is False

    def test_init_with_circuit_breaker_only(self):
        """Executor should work with circuit breaker only."""
        executor = AgentRetryExecutor(
            agent_id="test",
            circuit_breaker_policy=CircuitBreakerPolicy(),
        )

        assert executor.has_retry is False
        assert executor.has_circuit_breaker is True

    def test_init_with_both(self):
        """Executor should work with both policies."""
        executor = AgentRetryExecutor(
            agent_id="test",
            retry_policy=RetryPolicy(max_retries=2),
            circuit_breaker_policy=CircuitBreakerPolicy(),
        )

        assert executor.has_retry is True
        assert executor.has_circuit_breaker is True

    async def test_execute_success_no_retry(self):
        """Successful execution should return immediately."""
        executor = AgentRetryExecutor(
            agent_id="test",
            retry_policy=RetryPolicy(max_retries=3),
        )

        async def success_func(*args, **kwargs):
            return "success"

        result, metrics = await executor.execute(success_func)

        assert result == "success"
        assert metrics.total_attempts == 1
        assert metrics.successful_attempt == 0
        assert metrics.retries_performed == 0

    async def test_execute_retry_then_success(self):
        """Should retry on failure then succeed."""
        executor = AgentRetryExecutor(
            agent_id="test",
            retry_policy=RetryPolicy(
                max_retries=3,
                backoff="constant",
                base_delay=0.01,  # Fast for testing
            ),
        )

        call_count = 0

        async def fail_then_succeed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "recovered"

        result, metrics = await executor.execute(fail_then_succeed)

        assert result == "recovered"
        assert call_count == 3
        assert metrics.total_attempts == 3
        assert metrics.successful_attempt == 2
        assert metrics.retries_performed == 2

    async def test_execute_all_retries_exhausted(self):
        """Should raise after all retries exhausted."""
        executor = AgentRetryExecutor(
            agent_id="test",
            retry_policy=RetryPolicy(
                max_retries=2,
                backoff="constant",
                base_delay=0.01,
            ),
        )

        async def always_fail(*args, **kwargs):
            raise TimeoutError("Always fails")

        with pytest.raises(TimeoutError, match="Always fails"):
            await executor.execute(always_fail)

    async def test_execute_non_retryable_exception(self):
        """Should not retry on non-retryable exceptions."""
        executor = AgentRetryExecutor(
            agent_id="test",
            retry_policy=RetryPolicy(
                max_retries=3,
                retryable_exceptions=(ConnectionError,),
                backoff="constant",
                base_delay=0.01,
            ),
        )

        call_count = 0

        async def raise_value_error(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError, match="Not retryable"):
            await executor.execute(raise_value_error)

        assert call_count == 1  # No retries

    async def test_execute_with_circuit_breaker(self):
        """Circuit breaker should track successes."""
        executor = AgentRetryExecutor(
            agent_id="test",
            circuit_breaker_policy=CircuitBreakerPolicy(
                failure_threshold=3,
            ),
        )

        async def success_func(*args, **kwargs):
            return "ok"

        result, metrics = await executor.execute(success_func)

        assert result == "ok"
        assert metrics.circuit_breaker_state == "closed"
        assert metrics.circuit_breaker_blocked is False

    async def test_circuit_breaker_opens_after_failures(self):
        """Circuit breaker should open after failure_threshold failures."""
        executor = AgentRetryExecutor(
            agent_id="test",
            retry_policy=RetryPolicy(
                max_retries=0,  # No retry - let it fail immediately
            ),
            circuit_breaker_policy=CircuitBreakerPolicy(
                failure_threshold=3,
                recovery_timeout_seconds=60.0,
            ),
        )

        async def always_fail(*args, **kwargs):
            raise RuntimeError("Fail")

        # Fail 3 times to trip the circuit breaker
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await executor.execute(always_fail)

        # Now circuit should be open
        assert executor.circuit_breaker.state == CircuitState.OPEN

        # Next call should raise CircuitBreakerOpen
        with pytest.raises(CircuitBreakerOpen):
            await executor.execute(always_fail)

    async def test_on_retry_callback(self):
        """on_retry callback should be called on each retry."""
        retry_log = []

        def on_retry(attempt, exc, delay):
            retry_log.append((attempt, str(exc), delay))

        executor = AgentRetryExecutor(
            agent_id="test",
            retry_policy=RetryPolicy(
                max_retries=2,
                backoff="constant",
                base_delay=0.01,
                on_retry=on_retry,
            ),
        )

        call_count = 0

        async def fail_twice(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError(f"Fail #{call_count}")
            return "done"

        result, _ = await executor.execute(fail_twice)

        assert result == "done"
        assert len(retry_log) == 2
        assert retry_log[0][0] == 1
        assert retry_log[1][0] == 2

    async def test_get_circuit_breaker_metrics(self):
        """get_circuit_breaker_metrics should return breaker state."""
        executor = AgentRetryExecutor(
            agent_id="test",
            circuit_breaker_policy=CircuitBreakerPolicy(),
        )

        metrics = executor.get_circuit_breaker_metrics()

        assert metrics is not None
        assert metrics["state"] == "closed"
        assert metrics["failure_count"] == 0

    async def test_reset_circuit_breaker(self):
        """reset_circuit_breaker should close the circuit."""
        executor = AgentRetryExecutor(
            agent_id="test",
            retry_policy=RetryPolicy(max_retries=0),
            circuit_breaker_policy=CircuitBreakerPolicy(
                failure_threshold=2,
                recovery_timeout_seconds=60.0,
            ),
        )

        async def fail(*args, **kwargs):
            raise RuntimeError("fail")

        # Trip the breaker
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await executor.execute(fail)

        assert executor.circuit_breaker.state == CircuitState.OPEN

        # Reset
        await executor.reset_circuit_breaker()

        assert executor.circuit_breaker.state == CircuitState.CLOSED

    def test_no_retry_no_circuit_breaker(self):
        """Executor with no policies should not have retry/cb."""
        executor = AgentRetryExecutor(agent_id="test")

        assert executor.has_retry is False
        assert executor.has_circuit_breaker is False

    async def test_no_policy_passthrough(self):
        """With no policies, execute should pass through directly."""
        executor = AgentRetryExecutor(agent_id="test")

        async def simple(*args, **kwargs):
            return "direct"

        result, metrics = await executor.execute(simple)

        assert result == "direct"
        assert metrics.total_attempts == 1


class TestAgentConfigRetryFields:
    """Tests for AgentConfig retry_policy and circuit_breaker fields."""

    def test_agent_config_defaults(self):
        """AgentConfig should have None defaults for retry fields."""
        config = AgentConfig(role="test")

        assert config.retry_policy is None
        assert config.circuit_breaker is None

    def test_agent_config_with_retry_policy(self):
        """AgentConfig should accept RetryPolicy."""
        policy = RetryPolicy(max_retries=5)
        config = AgentConfig(role="test", retry_policy=policy)

        assert config.retry_policy is policy
        assert config.retry_policy.max_retries == 5

    def test_agent_config_with_circuit_breaker(self):
        """AgentConfig should accept CircuitBreakerPolicy."""
        cb = CircuitBreakerPolicy(failure_threshold=3)
        config = AgentConfig(role="test", circuit_breaker=cb)

        assert config.circuit_breaker is cb
        assert config.circuit_breaker.failure_threshold == 3

    def test_llm_agent_config_inherits_retry(self):
        """LLMAgentConfig should inherit retry fields."""
        policy = RetryPolicy(max_retries=2)
        config = LLMAgentConfig(
            role="test",
            prompt="test prompt",
            retry_policy=policy,
        )

        assert config.retry_policy is policy


class TestLLMTeamRetryIntegration:
    """Tests for LLMTeam integration with per-agent retry."""

    def test_add_agent_with_retry_policy_dict(self):
        """add_agent should accept retry_policy as dict."""
        team = LLMTeam(team_id="test")

        agent = team.add_agent({
            "type": "llm",
            "role": "retrier",
            "prompt": "test",
            "retry_policy": {
                "max_retries": 5,
                "backoff": "linear",
                "base_delay": 2.0,
            },
        })

        assert agent.config.retry_policy is not None
        assert agent.config.retry_policy.max_retries == 5
        assert agent.config.retry_policy.backoff == "linear"

    def test_add_agent_with_circuit_breaker_dict(self):
        """add_agent should accept circuit_breaker as dict."""
        team = LLMTeam(team_id="test")

        agent = team.add_agent({
            "type": "llm",
            "role": "breaker",
            "prompt": "test",
            "circuit_breaker": {
                "failure_threshold": 3,
                "recovery_timeout_seconds": 10.0,
            },
        })

        assert agent.config.circuit_breaker is not None
        assert agent.config.circuit_breaker.failure_threshold == 3

    def test_add_agent_with_both_policies(self):
        """add_agent should accept both retry and circuit breaker."""
        team = LLMTeam(team_id="test")

        agent = team.add_agent({
            "type": "llm",
            "role": "resilient",
            "prompt": "test",
            "retry_policy": {"max_retries": 3},
            "circuit_breaker": {"failure_threshold": 5},
        })

        assert agent.config.retry_policy is not None
        assert agent.config.circuit_breaker is not None
        assert agent.retry_executor is not None
        assert agent.retry_executor.has_retry is True
        assert agent.retry_executor.has_circuit_breaker is True

    def test_add_agent_without_retry(self):
        """Agent without retry should have no executor."""
        team = LLMTeam(team_id="test")

        agent = team.add_agent({
            "type": "llm",
            "role": "normal",
            "prompt": "test",
        })

        assert agent.retry_executor is None

    def test_add_agent_with_retry_policy_object(self):
        """add_agent should accept RetryPolicy object via AgentConfig."""
        team = LLMTeam(team_id="test")

        config = LLMAgentConfig(
            role="custom",
            prompt="test",
            retry_policy=RetryPolicy(max_retries=4, backoff="constant"),
            circuit_breaker=CircuitBreakerPolicy(failure_threshold=2),
        )

        agent = team.add_agent(config)

        assert agent.retry_executor is not None
        assert agent.retry_executor.has_retry is True
        assert agent.retry_executor.has_circuit_breaker is True


class TestAgentProcessWithRetry:
    """Tests for BaseAgent._process() with retry integration."""

    async def test_process_with_retry_success(self):
        """_process with retry should succeed on first try."""
        team = LLMTeam(team_id="test")

        agent = team.add_agent({
            "type": "llm",
            "role": "retrier",
            "prompt": "test",
            "retry_policy": {"max_retries": 3, "base_delay": 0.01},
        })

        # Mock _execute to succeed
        async def mock_execute(input_data, context):
            from llmteam import AgentResult
            return AgentResult(output={"result": "ok"})

        agent._execute = mock_execute

        result = await agent._process(
            input_data={"query": "test"},
            context={},
            run_id="run-1",
        )

        assert result.success is True
        assert result.output == {"result": "ok"}
        assert result.context_payload is not None
        assert "retry_metrics" in result.context_payload
        assert result.context_payload["retry_metrics"]["total_attempts"] == 1

    async def test_process_with_retry_recovers(self):
        """_process should retry and recover from transient failures."""
        team = LLMTeam(team_id="test")

        agent = team.add_agent({
            "type": "llm",
            "role": "retrier",
            "prompt": "test",
            "retry_policy": {
                "max_retries": 3,
                "backoff": "constant",
                "base_delay": 0.01,
            },
        })

        call_count = 0

        async def mock_execute(input_data, context):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient failure")
            from llmteam import AgentResult
            return AgentResult(output={"recovered": True})

        agent._execute = mock_execute

        result = await agent._process(
            input_data={"query": "test"},
            context={},
            run_id="run-1",
        )

        assert result.success is True
        assert result.output == {"recovered": True}
        assert result.context_payload["retry_metrics"]["retries_performed"] == 2

    async def test_process_with_retry_exhausted(self):
        """_process should return error result when retries exhausted."""
        team = LLMTeam(team_id="test")

        agent = team.add_agent({
            "type": "llm",
            "role": "retrier",
            "prompt": "test",
            "retry_policy": {
                "max_retries": 2,
                "backoff": "constant",
                "base_delay": 0.01,
            },
        })

        async def mock_execute(input_data, context):
            raise TimeoutError("Always timeout")

        agent._execute = mock_execute

        result = await agent._process(
            input_data={"query": "test"},
            context={},
            run_id="run-1",
        )

        assert result.success is False
        assert "Always timeout" in result.error

    async def test_process_circuit_breaker_escalation(self):
        """Circuit breaker open should trigger escalation flag."""
        team = LLMTeam(team_id="test")

        agent = team.add_agent({
            "type": "llm",
            "role": "breaker",
            "prompt": "test",
            "retry_policy": {"max_retries": 0, "base_delay": 0.01},
            "circuit_breaker": {
                "failure_threshold": 2,
                "recovery_timeout_seconds": 60.0,
            },
        })

        async def mock_execute(input_data, context):
            raise RuntimeError("Service unavailable")

        agent._execute = mock_execute

        # Trip the circuit breaker
        for _ in range(2):
            await agent._process({"q": "test"}, {}, "run-1")

        # Now circuit is open - next call should flag escalation
        result = await agent._process({"q": "test"}, {}, "run-2")

        assert result.success is False
        assert result.should_escalate is True
        assert "Circuit breaker open" in (result.escalation_reason or "")

    async def test_process_without_retry_unchanged(self):
        """Agent without retry should behave as before."""
        team = LLMTeam(team_id="test")

        agent = team.add_agent({
            "type": "llm",
            "role": "normal",
            "prompt": "test",
        })

        async def mock_execute(input_data, context):
            from llmteam import AgentResult
            return AgentResult(output={"normal": True})

        agent._execute = mock_execute

        result = await agent._process(
            input_data={"query": "test"},
            context={},
            run_id="run-1",
        )

        assert result.success is True
        assert result.output == {"normal": True}
        # No retry metrics when no policy configured
        assert result.context_payload is None
