"""
Circuit breaker implementation.

This module provides CircuitBreaker - prevents cascading failures
by stopping requests to failing services.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Optional, Tuple

from llmteam.ratelimit.config import (
    CircuitBreakerConfig,
    CircuitState,
    CircuitOpenError,
)


@dataclass
class CircuitBreakerStats:
    """Statistics for a circuit breaker."""
    
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0  # Rejected due to open circuit
    state_changes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    time_in_open: float = 0.0
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        total = self.successful_requests + self.failed_requests
        if total == 0:
            return 0.0
        return self.failed_requests / total


class CircuitBreaker:
    """
    Circuit breaker for preventing cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Requests are blocked after too many failures
    - HALF_OPEN: Testing if service has recovered
    
    Example:
        config = CircuitBreakerConfig(
            failure_threshold=5,
            open_timeout=timedelta(seconds=30),
        )
        
        breaker = CircuitBreaker("my_api", config)
        
        if breaker.allow_request():
            try:
                result = await call_api()
                breaker.record_success()
            except Exception as e:
                breaker.record_failure()
                raise
    """
    
    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig,
    ):
        """
        Initialize CircuitBreaker.
        
        Args:
            name: Name for this circuit breaker (for logging/errors)
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config
        
        # State
        self._state = CircuitState.CLOSED
        self._state_changed_at = time.monotonic()
        
        # Tracking
        self._recent_results: Deque[Tuple[float, bool]] = deque(
            maxlen=config.sample_size
        )
        self._half_open_requests = 0
        self._half_open_successes = 0
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        # Stats
        self._stats = CircuitBreakerStats()
        self._open_started_at: Optional[float] = None
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (allowing requests)."""
        return self._state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self._state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing)."""
        return self._state == CircuitState.HALF_OPEN
    
    async def allow_request(self) -> bool:
        """
        Check if a request should be allowed.
        
        Returns:
            True if request is allowed
            
        Raises:
            CircuitOpenError: If circuit is open and not ready to test
        """
        async with self._lock:
            self._stats.total_requests += 1
            
            if self._state == CircuitState.CLOSED:
                return True
            
            elif self._state == CircuitState.OPEN:
                # Check if we should transition to half-open
                elapsed = time.monotonic() - self._state_changed_at
                if elapsed >= self.config.open_timeout.total_seconds():
                    self._transition_to(CircuitState.HALF_OPEN)
                    self._half_open_requests = 1
                    self._half_open_successes = 0
                    return True
                
                # Still open
                self._stats.rejected_requests += 1
                retry_after = self.config.open_timeout.total_seconds() - elapsed
                raise CircuitOpenError(self.name, retry_after=retry_after)
            
            elif self._state == CircuitState.HALF_OPEN:
                # Allow limited requests in half-open state
                if self._half_open_requests < self.config.half_open_max_requests:
                    self._half_open_requests += 1
                    return True
                
                # Already at max test requests, reject
                self._stats.rejected_requests += 1
                raise CircuitOpenError(self.name, retry_after=1.0)
        
        return False
    
    async def record_success(self) -> None:
        """Record a successful request."""
        async with self._lock:
            now = time.monotonic()
            self._recent_results.append((now, True))
            self._stats.successful_requests += 1
            self._stats.last_success_time = now

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1

                # Check if we should close
                if self._half_open_successes >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

            elif self._state == CircuitState.CLOSED:
                # Reset failure tracking on success
                self._recent_results.clear()
                self._recent_results.append((now, True))
    
    async def record_failure(self) -> None:
        """Record a failed request."""
        async with self._lock:
            now = time.monotonic()
            self._recent_results.append((now, False))
            self._stats.failed_requests += 1
            self._stats.last_failure_time = now
            
            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._transition_to(CircuitState.OPEN)
                return
            
            if self._state == CircuitState.CLOSED:
                # Check if we should open
                failures, total = self._count_recent_failures()
                if self.config.should_open(failures, total):
                    self._transition_to(CircuitState.OPEN)
    
    def _count_recent_failures(self) -> Tuple[int, int]:
        """Count recent failures within sample window."""
        now = time.monotonic()
        cutoff = now - 60.0  # Last minute
        
        recent = [(t, s) for t, s in self._recent_results if t > cutoff]
        total = len(recent)
        failures = sum(1 for _, success in recent if not success)
        
        return failures, total
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        
        if old_state == new_state:
            return
        
        # Track time in open state
        if old_state == CircuitState.OPEN and self._open_started_at:
            self._stats.time_in_open += time.monotonic() - self._open_started_at
            self._open_started_at = None
        
        if new_state == CircuitState.OPEN:
            self._open_started_at = time.monotonic()
        
        self._state = new_state
        self._state_changed_at = time.monotonic()
        self._stats.state_changes += 1
        
        # Reset half-open counters
        if new_state == CircuitState.HALF_OPEN:
            self._half_open_requests = 0
            self._half_open_successes = 0
    
    async def force_open(self) -> None:
        """Force circuit to open state."""
        async with self._lock:
            self._transition_to(CircuitState.OPEN)
    
    async def force_close(self) -> None:
        """Force circuit to closed state."""
        async with self._lock:
            self._transition_to(CircuitState.CLOSED)
    
    async def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._state_changed_at = time.monotonic()
            self._recent_results.clear()
            self._half_open_requests = 0
            self._half_open_successes = 0
    
    def get_stats(self) -> CircuitBreakerStats:
        """Get current statistics."""
        return self._stats
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = CircuitBreakerStats()
    
    def get_state_duration(self) -> float:
        """Get how long we've been in current state (seconds)."""
        return time.monotonic() - self._state_changed_at
    
    def get_retry_after(self) -> Optional[float]:
        """Get seconds until circuit might close (if open)."""
        if self._state != CircuitState.OPEN:
            return None
        
        elapsed = time.monotonic() - self._state_changed_at
        remaining = self.config.open_timeout.total_seconds() - elapsed
        return max(0.0, remaining)
