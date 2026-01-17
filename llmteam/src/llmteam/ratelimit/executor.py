"""
Rate limited executor implementation.

This module provides RateLimitedExecutor - combines rate limiting
and circuit breaking for executing external calls.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional, TypeVar, Generic

from llmteam.ratelimit.config import (
    RateLimitConfig,
    CircuitBreakerConfig,
    RateLimitStrategy,
    RateLimitExceeded,
    CircuitOpenError,
)
from llmteam.ratelimit.limiter import RateLimiter, RateLimiterStats
from llmteam.ratelimit.circuit import CircuitBreaker, CircuitBreakerStats


T = TypeVar('T')


@dataclass
class ExecutionResult(Generic[T]):
    """Result of an execution attempt."""
    
    success: bool
    value: Optional[T] = None
    error: Optional[Exception] = None
    attempts: int = 1
    total_time_ms: int = 0
    rate_limited: bool = False
    circuit_open: bool = False
    used_fallback: bool = False


@dataclass
class EndpointStats:
    """Combined statistics for an endpoint."""
    
    rate_limiter: RateLimiterStats = field(default_factory=RateLimiterStats)
    circuit_breaker: CircuitBreakerStats = field(default_factory=CircuitBreakerStats)
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    fallback_executions: int = 0
    total_retries: int = 0


from llmteam.licensing import professional_only


@professional_only
class RateLimitedExecutor:
    """
    Executor that combines rate limiting and circuit breaking.
    
    Provides:
    - Rate limiting per endpoint
    - Circuit breaker per endpoint
    - Automatic retry with exponential backoff
    - Fallback support
    
    Example:
        executor = RateLimitedExecutor()
        
        # Register endpoint
        executor.register(
            "payment_api",
            RateLimitConfig(
                requests_per_minute=100,
                strategy=RateLimitStrategy.QUEUE,
            ),
            CircuitBreakerConfig(
                failure_threshold=5,
            ),
        )
        
        # Execute with protection
        result = await executor.execute(
            "payment_api",
            process_payment,
            amount=100,
        )
    """
    
    def __init__(self, default_config: RateLimitConfig = None):
        """
        Initialize RateLimitedExecutor.
        
        Args:
            default_config: Default rate limit config for unregistered endpoints
        """
        self.default_config = default_config or RateLimitConfig()
        
        self._rate_limiters: Dict[str, RateLimiter] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._configs: Dict[str, RateLimitConfig] = {}
        self._stats: Dict[str, EndpointStats] = {}
    
    def register(
        self,
        name: str,
        rate_config: RateLimitConfig = None,
        circuit_config: CircuitBreakerConfig = None,
    ) -> None:
        """
        Register an endpoint for rate limiting.
        
        Args:
            name: Endpoint name
            rate_config: Rate limiting configuration
            circuit_config: Circuit breaker configuration
        """
        config = rate_config or self.default_config
        self._configs[name] = config
        self._rate_limiters[name] = RateLimiter(name, config)
        
        if circuit_config:
            self._circuit_breakers[name] = CircuitBreaker(name, circuit_config)
        
        self._stats[name] = EndpointStats()
    
    def unregister(self, name: str) -> None:
        """
        Unregister an endpoint.
        
        Args:
            name: Endpoint name
        """
        self._rate_limiters.pop(name, None)
        self._circuit_breakers.pop(name, None)
        self._configs.pop(name, None)
        self._stats.pop(name, None)
    
    async def execute(
        self,
        name: str,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> ExecutionResult[T]:
        """
        Execute a function with rate limiting and circuit breaking.
        
        Args:
            name: Endpoint name
            func: Function to execute (sync or async)
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            ExecutionResult with success status and value or error
        """
        # Get or create limiters
        limiter = self._rate_limiters.get(name)
        breaker = self._circuit_breakers.get(name)
        config = self._configs.get(name, self.default_config)
        
        if name not in self._stats:
            self._stats[name] = EndpointStats()
        stats = self._stats[name]
        
        stats.total_executions += 1
        start_time = asyncio.get_event_loop().time()
        
        # Try with retries
        last_error: Optional[Exception] = None
        attempts = 0
        
        for attempt in range(config.retry_count + 1):
            attempts = attempt + 1
            
            try:
                # Check circuit breaker
                if breaker:
                    try:
                        allowed = await breaker.allow_request()
                        if not allowed:
                            return ExecutionResult(
                                success=False,
                                error=CircuitOpenError(name),
                                attempts=attempts,
                                circuit_open=True,
                            )
                    except CircuitOpenError as e:
                        # Try fallback
                        if config.strategy == RateLimitStrategy.FALLBACK:
                            return await self._execute_fallback(name, config, attempts, start_time)
                        return ExecutionResult(
                            success=False,
                            error=e,
                            attempts=attempts,
                            circuit_open=True,
                        )
                
                # Apply rate limiting
                if limiter:
                    try:
                        allowed = await limiter.acquire()
                        if not allowed:
                            # Fallback strategy
                            return await self._execute_fallback(name, config, attempts, start_time)
                    except RateLimitExceeded as e:
                        if config.strategy == RateLimitStrategy.FALLBACK:
                            return await self._execute_fallback(name, config, attempts, start_time)
                        raise
                
                try:
                    # Execute the function
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    
                    # Record success
                    if breaker:
                        await breaker.record_success()
                    
                    stats.successful_executions += 1
                    
                    elapsed = int((asyncio.get_event_loop().time() - start_time) * 1000)
                    
                    return ExecutionResult(
                        success=True,
                        value=result,
                        attempts=attempts,
                        total_time_ms=elapsed,
                    )
                    
                finally:
                    if limiter:
                        limiter.release()
                
            except RateLimitExceeded as e:
                last_error = e
                stats.rate_limiter.rejected_requests += 1
                
                # Don't retry rate limit errors
                return ExecutionResult(
                    success=False,
                    error=e,
                    attempts=attempts,
                    rate_limited=True,
                    total_time_ms=int((asyncio.get_event_loop().time() - start_time) * 1000),
                )
                
            except Exception as e:
                last_error = e
                
                # Record failure for circuit breaker
                if breaker:
                    await breaker.record_failure()
                
                # Retry with backoff
                if attempt < config.retry_count:
                    stats.total_retries += 1
                    delay = config.get_retry_delay(attempt)
                    await asyncio.sleep(delay)
                    continue
                
                break
        
        # All retries exhausted
        stats.failed_executions += 1
        elapsed = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        return ExecutionResult(
            success=False,
            error=last_error,
            attempts=attempts,
            total_time_ms=elapsed,
        )
    
    async def _execute_fallback(
        self,
        name: str,
        config: RateLimitConfig,
        attempts: int,
        start_time: float,
    ) -> ExecutionResult:
        """Execute fallback handler or return fallback value."""
        stats = self._stats[name]
        stats.fallback_executions += 1
        
        elapsed = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        if config.fallback_handler:
            try:
                if asyncio.iscoroutinefunction(config.fallback_handler):
                    value = await config.fallback_handler()
                else:
                    value = config.fallback_handler()
                
                return ExecutionResult(
                    success=True,
                    value=value,
                    attempts=attempts,
                    total_time_ms=elapsed,
                    used_fallback=True,
                )
            except Exception as e:
                return ExecutionResult(
                    success=False,
                    error=e,
                    attempts=attempts,
                    total_time_ms=elapsed,
                    used_fallback=True,
                )
        
        return ExecutionResult(
            success=True,
            value=config.fallback_value,
            attempts=attempts,
            total_time_ms=elapsed,
            used_fallback=True,
        )
    
    def get_stats(self, name: str = None) -> Dict[str, EndpointStats]:
        """
        Get statistics for endpoints.
        
        Args:
            name: Specific endpoint name, or None for all
            
        Returns:
            Dictionary of endpoint name to stats
        """
        if name:
            if name in self._stats:
                return {name: self._stats[name]}
            return {}
        return dict(self._stats)
    
    def reset_stats(self, name: str = None) -> None:
        """
        Reset statistics.
        
        Args:
            name: Specific endpoint name, or None for all
        """
        if name:
            if name in self._stats:
                self._stats[name] = EndpointStats()
            if name in self._rate_limiters:
                self._rate_limiters[name].reset_stats()
            if name in self._circuit_breakers:
                self._circuit_breakers[name].reset_stats()
        else:
            for n in self._stats:
                self._stats[n] = EndpointStats()
            for limiter in self._rate_limiters.values():
                limiter.reset_stats()
            for breaker in self._circuit_breakers.values():
                breaker.reset_stats()
    
    def is_healthy(self, name: str) -> bool:
        """
        Check if an endpoint is healthy.
        
        Returns True if circuit is closed and not rate limited.
        """
        if name in self._circuit_breakers:
            if not self._circuit_breakers[name].is_closed:
                return False
        
        if name in self._rate_limiters:
            if self._rate_limiters[name].is_limited:
                return False
        
        return True
    
    def get_health_status(self) -> Dict[str, dict]:
        """Get health status for all endpoints."""
        result = {}
        
        for name in set(self._rate_limiters.keys()) | set(self._circuit_breakers.keys()):
            result[name] = {
                "healthy": self.is_healthy(name),
                "circuit_state": (
                    self._circuit_breakers[name].state.value
                    if name in self._circuit_breakers
                    else "n/a"
                ),
                "rate_limited": (
                    self._rate_limiters[name].is_limited
                    if name in self._rate_limiters
                    else False
                ),
            }
        
        return result
