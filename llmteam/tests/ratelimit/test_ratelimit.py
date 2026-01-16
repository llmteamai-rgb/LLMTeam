"""
Tests for ratelimit module.
"""

import pytest
import asyncio
from datetime import timedelta

from llmteam.ratelimit import (
    RateLimiter,
    RateLimitConfig,
    RateLimitStrategy,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    RateLimitedExecutor,
    RateLimitExceeded,
    CircuitOpenError,
)


class TestRateLimitConfig:
    """Tests for RateLimitConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RateLimitConfig()
        
        assert config.requests_per_second == 10.0
        assert config.requests_per_minute == 100.0
        assert config.burst_size == 10
        assert config.strategy == RateLimitStrategy.WAIT
    
    def test_min_interval(self):
        """Test min interval calculation."""
        config = RateLimitConfig(requests_per_second=5.0)
        
        assert config.get_min_interval() == 0.2
    
    def test_retry_delay_linear(self):
        """Test linear retry delay."""
        config = RateLimitConfig(
            retry_base_delay=1.0,
            retry_exponential=False,
        )
        
        assert config.get_retry_delay(0) == 1.0
        assert config.get_retry_delay(1) == 1.0
        assert config.get_retry_delay(2) == 1.0
    
    def test_retry_delay_exponential(self):
        """Test exponential retry delay."""
        config = RateLimitConfig(
            retry_base_delay=1.0,
            retry_exponential=True,
            retry_max_delay=10.0,
        )
        
        assert config.get_retry_delay(0) == 1.0
        assert config.get_retry_delay(1) == 2.0
        assert config.get_retry_delay(2) == 4.0
        assert config.get_retry_delay(10) == 10.0  # Capped


class TestRateLimiter:
    """Tests for RateLimiter."""
    
    @pytest.mark.asyncio
    async def test_basic_acquire_release(self):
        """Test basic acquire and release."""
        config = RateLimitConfig(burst_size=5)
        limiter = RateLimiter("test", config)
        
        result = await limiter.acquire()
        
        assert result is True
        limiter.release()
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using as context manager."""
        config = RateLimitConfig(burst_size=5)
        limiter = RateLimiter("test", config)
        
        async with limiter:
            # Inside context, have acquired
            pass
        
        # Outside context, released
    
    @pytest.mark.asyncio
    async def test_burst_limit(self):
        """Test burst limit is enforced."""
        config = RateLimitConfig(
            burst_size=2,
            strategy=RateLimitStrategy.REJECT,
            requests_per_second=100,  # High rate to not hit per-second limit
        )
        limiter = RateLimiter("test", config)
        
        # Acquire all burst capacity
        await limiter.acquire()
        await limiter.acquire()
        
        # Third should fail
        with pytest.raises(RateLimitExceeded):
            await limiter.acquire()
    
    @pytest.mark.asyncio
    async def test_reject_strategy(self):
        """Test REJECT strategy."""
        config = RateLimitConfig(
            burst_size=1,
            strategy=RateLimitStrategy.REJECT,
            requests_per_second=1,
        )
        limiter = RateLimiter("test", config)
        
        await limiter.acquire()
        
        with pytest.raises(RateLimitExceeded):
            await limiter.acquire()
    
    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        """Test statistics are tracked."""
        config = RateLimitConfig(burst_size=5)
        limiter = RateLimiter("test", config)
        
        await limiter.acquire()
        limiter.release()
        await limiter.acquire()
        limiter.release()
        
        stats = limiter.get_stats()
        
        assert stats.total_requests == 2
        assert stats.successful_requests == 2


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""
    
    @pytest.mark.asyncio
    async def test_starts_closed(self):
        """Test circuit starts in closed state."""
        config = CircuitBreakerConfig()
        breaker = CircuitBreaker("test", config)
        
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
    
    @pytest.mark.asyncio
    async def test_opens_after_failures(self):
        """Test circuit opens after threshold failures."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
        )
        breaker = CircuitBreaker("test", config)
        
        # Record failures
        for _ in range(3):
            await breaker.allow_request()
            await breaker.record_failure()
        
        assert breaker.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_rejects_when_open(self):
        """Test requests are rejected when open."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            open_timeout=timedelta(seconds=10),
        )
        breaker = CircuitBreaker("test", config)
        
        # Open the circuit
        await breaker.allow_request()
        await breaker.record_failure()
        
        # Next request should fail
        with pytest.raises(CircuitOpenError):
            await breaker.allow_request()
    
    @pytest.mark.asyncio
    async def test_success_resets_failures(self):
        """Test success resets failure count."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
        )
        breaker = CircuitBreaker("test", config)
        
        # Two failures
        await breaker.allow_request()
        await breaker.record_failure()
        await breaker.allow_request()
        await breaker.record_failure()
        
        # Success resets
        await breaker.allow_request()
        await breaker.record_success()
        
        # Can have two more failures without opening
        await breaker.allow_request()
        await breaker.record_failure()
        await breaker.allow_request()
        await breaker.record_failure()
        
        assert breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self):
        """Test circuit goes half-open after timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            open_timeout=timedelta(milliseconds=50),
        )
        breaker = CircuitBreaker("test", config)
        
        # Open
        await breaker.allow_request()
        await breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        
        # Wait for timeout
        await asyncio.sleep(0.1)
        
        # Should transition to half-open
        await breaker.allow_request()
        assert breaker.state == CircuitState.HALF_OPEN
    
    @pytest.mark.asyncio
    async def test_closes_after_successful_test(self):
        """Test circuit closes after successful half-open test."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            open_timeout=timedelta(milliseconds=50),
            success_threshold=1,
        )
        breaker = CircuitBreaker("test", config)
        
        # Open
        await breaker.allow_request()
        await breaker.record_failure()
        
        # Wait and go half-open
        await asyncio.sleep(0.1)
        await breaker.allow_request()
        
        # Success closes
        await breaker.record_success()
        
        assert breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_reopens_on_half_open_failure(self):
        """Test circuit reopens if half-open test fails."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            open_timeout=timedelta(milliseconds=50),
        )
        breaker = CircuitBreaker("test", config)
        
        # Open
        await breaker.allow_request()
        await breaker.record_failure()
        
        # Wait and go half-open
        await asyncio.sleep(0.1)
        await breaker.allow_request()
        
        # Failure reopens
        await breaker.record_failure()
        
        assert breaker.state == CircuitState.OPEN


class TestRateLimitedExecutor:
    """Tests for RateLimitedExecutor."""
    
    @pytest.fixture
    def executor(self):
        """Create an executor."""
        return RateLimitedExecutor()
    
    @pytest.mark.asyncio
    async def test_execute_success(self, executor):
        """Test successful execution."""
        executor.register(
            "test_api",
            RateLimitConfig(burst_size=10),
        )
        
        async def success_func():
            return "result"
        
        result = await executor.execute("test_api", success_func)
        
        assert result.success is True
        assert result.value == "result"
        assert result.attempts == 1
    
    @pytest.mark.asyncio
    async def test_execute_failure_with_retry(self, executor):
        """Test execution with retry on failure."""
        executor.register(
            "test_api",
            RateLimitConfig(
                retry_count=2,
                retry_base_delay=0.01,
            ),
        )
        
        attempts = 0
        
        async def failing_func():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("Transient error")
            return "success"
        
        result = await executor.execute("test_api", failing_func)
        
        assert result.success is True
        assert result.value == "success"
        assert result.attempts == 3
    
    @pytest.mark.asyncio
    async def test_execute_all_retries_fail(self, executor):
        """Test when all retries fail."""
        executor.register(
            "test_api",
            RateLimitConfig(
                retry_count=2,
                retry_base_delay=0.01,
            ),
        )
        
        async def always_fail():
            raise ValueError("Permanent error")
        
        result = await executor.execute("test_api", always_fail)
        
        assert result.success is False
        assert result.attempts == 3
        assert isinstance(result.error, ValueError)
    
    @pytest.mark.asyncio
    async def test_execute_with_fallback(self, executor):
        """Test fallback execution."""
        executor.register(
            "test_api",
            RateLimitConfig(
                burst_size=0,  # Immediately rate limited
                strategy=RateLimitStrategy.FALLBACK,
                fallback_value="fallback_result",
            ),
        )
        
        async def main_func():
            return "main_result"
        
        result = await executor.execute("test_api", main_func)
        
        assert result.success is True
        assert result.value == "fallback_result"
        assert result.used_fallback is True
    
    @pytest.mark.asyncio
    async def test_execute_with_circuit_breaker(self, executor):
        """Test execution with circuit breaker."""
        executor.register(
            "test_api",
            RateLimitConfig(retry_count=0),
            CircuitBreakerConfig(
                failure_threshold=2,
                open_timeout=timedelta(seconds=10),
            ),
        )
        
        async def failing_func():
            raise ValueError("Error")
        
        # Fail twice to open circuit
        await executor.execute("test_api", failing_func)
        await executor.execute("test_api", failing_func)
        
        # Third should be circuit open
        result = await executor.execute("test_api", failing_func)
        
        assert result.success is False
        assert result.circuit_open is True
    
    @pytest.mark.asyncio
    async def test_health_status(self, executor):
        """Test health status reporting."""
        executor.register(
            "api1",
            RateLimitConfig(),
            CircuitBreakerConfig(),
        )
        executor.register(
            "api2",
            RateLimitConfig(),
        )
        
        status = executor.get_health_status()
        
        assert "api1" in status
        assert "api2" in status
        assert status["api1"]["healthy"] is True
    
    @pytest.mark.asyncio
    async def test_sync_function_execution(self, executor):
        """Test executing sync functions."""
        executor.register("test_api", RateLimitConfig())
        
        def sync_func(x, y):
            return x + y
        
        result = await executor.execute("test_api", sync_func, 1, 2)
        
        assert result.success is True
        assert result.value == 3
