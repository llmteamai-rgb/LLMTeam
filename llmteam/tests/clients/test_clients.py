"""Tests for clients module."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from llmteam.clients import (
    HTTPClientConfig,
    HTTPResponse,
    RetryConfig,
    ExponentialBackoff,
    LinearBackoff,
    ConstantBackoff,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitState,
)


class TestHTTPClientConfig:
    """Tests for HTTPClientConfig."""

    def test_default_values(self):
        """Config has sensible defaults."""
        config = HTTPClientConfig()

        assert config.timeout_seconds == 30.0
        assert config.connect_timeout_seconds == 10.0
        assert config.verify_ssl is True
        assert config.auth_type == "Bearer"

    def test_custom_values(self):
        """Custom values are respected."""
        config = HTTPClientConfig(
            base_url="https://api.example.com",
            timeout_seconds=60.0,
            auth_token="my-token",
        )

        assert config.base_url == "https://api.example.com"
        assert config.timeout_seconds == 60.0
        assert config.auth_token == "my-token"


class TestHTTPResponse:
    """Tests for HTTPResponse."""

    def test_create_response(self):
        """Create response with required fields."""
        response = HTTPResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body=b'{"result": "ok"}',
            elapsed_ms=100,
        )

        assert response.status_code == 200
        assert response.elapsed_ms == 100

    def test_text_property(self):
        """Text property decodes body."""
        response = HTTPResponse(
            status_code=200,
            headers={},
            body=b"Hello World",
            elapsed_ms=100,
        )

        assert response.text == "Hello World"

    def test_json_property(self):
        """JSON property parses body."""
        response = HTTPResponse(
            status_code=200,
            headers={},
            body=b'{"key": "value"}',
            elapsed_ms=100,
        )

        assert response.json == {"key": "value"}

    def test_ok_property_success(self):
        """OK property true for 2xx status."""
        for status in [200, 201, 204]:
            response = HTTPResponse(
                status_code=status,
                headers={},
                body=b"",
                elapsed_ms=100,
            )
            assert response.ok is True

    def test_ok_property_failure(self):
        """OK property false for non-2xx status."""
        for status in [400, 401, 404, 500]:
            response = HTTPResponse(
                status_code=status,
                headers={},
                body=b"",
                elapsed_ms=100,
            )
            assert response.ok is False


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_values(self):
        """Config has sensible defaults."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert 429 in config.retry_status_codes
        assert 500 in config.retry_status_codes
        assert config.jitter == 0.1

    def test_custom_status_codes(self):
        """Custom retry status codes are respected."""
        config = RetryConfig(retry_status_codes={500, 502})

        assert config.retry_status_codes == {500, 502}


class TestExponentialBackoff:
    """Tests for ExponentialBackoff."""

    def test_default_values(self):
        """Strategy has sensible defaults."""
        strategy = ExponentialBackoff()

        assert strategy.base_delay == 1.0
        assert strategy.multiplier == 2.0
        assert strategy.max_delay == 60.0

    def test_delay_increases_exponentially(self):
        """Delay increases exponentially with attempts."""
        strategy = ExponentialBackoff(base_delay=1.0, multiplier=2.0)

        assert strategy.get_delay(0) == 1.0
        assert strategy.get_delay(1) == 2.0
        assert strategy.get_delay(2) == 4.0
        assert strategy.get_delay(3) == 8.0

    def test_delay_capped_at_max(self):
        """Delay is capped at max_delay."""
        strategy = ExponentialBackoff(base_delay=1.0, multiplier=2.0, max_delay=5.0)

        assert strategy.get_delay(10) == 5.0

    def test_jitter_applied(self):
        """Jitter randomizes delay."""
        strategy = ExponentialBackoff(base_delay=10.0)

        delays = [strategy.apply_jitter(10.0, 0.1) for _ in range(10)]

        # All delays should be within jitter range
        for delay in delays:
            assert 9.0 <= delay <= 11.0

        # Delays should not all be the same
        assert len(set(delays)) > 1


class TestLinearBackoff:
    """Tests for LinearBackoff."""

    def test_default_values(self):
        """Strategy has sensible defaults."""
        strategy = LinearBackoff()

        assert strategy.base_delay == 1.0
        assert strategy.increment == 1.0
        assert strategy.max_delay == 60.0

    def test_delay_increases_linearly(self):
        """Delay increases linearly with attempts."""
        strategy = LinearBackoff(base_delay=1.0, increment=2.0)

        assert strategy.get_delay(0) == 1.0
        assert strategy.get_delay(1) == 3.0
        assert strategy.get_delay(2) == 5.0
        assert strategy.get_delay(3) == 7.0


class TestConstantBackoff:
    """Tests for ConstantBackoff."""

    def test_constant_delay(self):
        """Delay is constant regardless of attempt."""
        strategy = ConstantBackoff(delay=5.0)

        assert strategy.get_delay(0) == 5.0
        assert strategy.get_delay(1) == 5.0
        assert strategy.get_delay(10) == 5.0


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_values(self):
        """Config has sensible defaults."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.recovery_timeout == timedelta(seconds=30)


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initial_state_closed(self):
        """Circuit breaker starts in closed state."""
        breaker = CircuitBreaker()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed is True
        assert breaker.is_open is False

    async def test_record_success(self):
        """Recording success keeps circuit closed."""
        breaker = CircuitBreaker()

        await breaker._record_success()

        assert breaker.is_closed is True

    async def test_opens_after_threshold(self):
        """Circuit opens after failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(config)

        for _ in range(3):
            await breaker._record_failure(Exception("test"))

        assert breaker.is_open is True

    async def test_half_open_after_recovery(self):
        """Circuit becomes half-open after recovery timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=timedelta(milliseconds=10),
        )
        breaker = CircuitBreaker(config)

        # Open the circuit
        await breaker._record_failure(Exception("test"))
        assert breaker.is_open is True

        # Wait for recovery
        import asyncio
        await asyncio.sleep(0.02)

        # Check state should transition to half-open
        await breaker._check_state()
        assert breaker.is_half_open is True

    async def test_closes_after_success_threshold(self):
        """Circuit closes after success threshold in half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=2,
            recovery_timeout=timedelta(milliseconds=10),
        )
        breaker = CircuitBreaker(config)

        # Open and transition to half-open
        await breaker._record_failure(Exception("test"))
        import asyncio
        await asyncio.sleep(0.02)
        await breaker._check_state()

        # Record successes
        await breaker._record_success()
        assert breaker.is_half_open is True

        await breaker._record_success()
        assert breaker.is_closed is True

    async def test_reopens_on_failure_in_half_open(self):
        """Circuit reopens on failure in half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=timedelta(milliseconds=10),
        )
        breaker = CircuitBreaker(config)

        # Open and transition to half-open
        await breaker._record_failure(Exception("test"))
        import asyncio
        await asyncio.sleep(0.02)
        await breaker._check_state()
        assert breaker.is_half_open is True

        # Record failure
        await breaker._record_failure(Exception("test"))
        assert breaker.is_open is True

    async def test_raises_when_open(self):
        """Raises CircuitBreakerOpen when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker(config)

        await breaker._record_failure(Exception("test"))

        with pytest.raises(CircuitBreakerOpen):
            await breaker._check_state()

    async def test_reset_closes_circuit(self):
        """Reset manually closes circuit."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker(config)

        await breaker._record_failure(Exception("test"))
        assert breaker.is_open is True

        await breaker.reset()
        assert breaker.is_closed is True

    async def test_trip_opens_circuit(self):
        """Trip manually opens circuit."""
        breaker = CircuitBreaker()

        await breaker.trip()

        assert breaker.is_open is True

    def test_get_metrics(self):
        """Get metrics returns circuit state info."""
        breaker = CircuitBreaker()

        metrics = breaker.get_metrics()

        assert "state" in metrics
        assert metrics["state"] == "closed"
        assert "failure_count" in metrics
        assert "success_count" in metrics

    async def test_context_manager_success(self):
        """Context manager records success."""
        breaker = CircuitBreaker()

        async with breaker:
            pass  # Success

        # Circuit should still be closed
        assert breaker.is_closed is True

    async def test_context_manager_failure(self):
        """Context manager records failure."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker(config)

        with pytest.raises(ValueError):
            async with breaker:
                raise ValueError("test")

        # Circuit should be open
        assert breaker.is_open is True

    async def test_call_method(self):
        """Call method executes function with circuit breaker."""
        breaker = CircuitBreaker()

        async def my_func():
            return "result"

        result = await breaker.call(my_func)

        assert result == "result"
