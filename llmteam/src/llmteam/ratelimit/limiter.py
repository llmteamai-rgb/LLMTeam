"""
Rate limiter implementation.

This module provides RateLimiter - a token bucket rate limiter
with support for multiple time windows.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Optional

from llmteam.ratelimit.config import (
    RateLimitConfig,
    RateLimitStrategy,
    RateLimitExceeded,
    QueueFullError,
)


@dataclass
class RateLimiterStats:
    """Statistics for a rate limiter."""
    
    total_requests: int = 0
    successful_requests: int = 0
    rejected_requests: int = 0
    queued_requests: int = 0
    total_wait_time: float = 0.0
    current_queue_size: int = 0
    
    @property
    def rejection_rate(self) -> float:
        """Calculate rejection rate."""
        if self.total_requests == 0:
            return 0.0
        return self.rejected_requests / self.total_requests
    
    @property
    def average_wait_time(self) -> float:
        """Calculate average wait time."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_wait_time / self.successful_requests


class RateLimiter:
    """
    Token bucket rate limiter.
    
    Provides rate limiting with configurable strategies:
    - WAIT: Block until request can be made
    - REJECT: Immediately reject if rate exceeded
    - QUEUE: Add to queue for later processing
    - FALLBACK: Return fallback value
    
    Example:
        config = RateLimitConfig(
            requests_per_minute=100,
            burst_size=10,
            strategy=RateLimitStrategy.WAIT,
        )
        
        limiter = RateLimiter("my_api", config)
        
        async with limiter:
            # Make API call
            result = await call_api()
    """
    
    def __init__(
        self,
        name: str,
        config: RateLimitConfig,
    ):
        """
        Initialize RateLimiter.
        
        Args:
            name: Name for this rate limiter (for logging/errors)
            config: Rate limiting configuration
        """
        self.name = name
        self.config = config
        
        # Token bucket
        self._tokens = float(config.burst_size)
        self._last_update = time.monotonic()
        
        # Request tracking (for per-minute/hour limits)
        self._requests_second: Deque[float] = deque()
        self._requests_minute: Deque[float] = deque()
        self._requests_hour: Deque[float] = deque()
        
        # Queue (for QUEUE strategy)
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=config.queue_size)
        
        # Concurrency control
        self._semaphore = asyncio.Semaphore(config.burst_size)
        self._lock = asyncio.Lock()
        
        # Stats
        self._stats = RateLimiterStats()
    
    async def acquire(self) -> bool:
        """
        Acquire permission to make a request.
        
        Returns:
            True if permission granted, False if using fallback
            
        Raises:
            RateLimitExceeded: If strategy is REJECT and limit exceeded
            QueueFullError: If strategy is QUEUE and queue is full
        """
        self._stats.total_requests += 1
        
        async with self._lock:
            self._update_tokens()
            self._cleanup_request_history()
            
            # Check if we have capacity
            if not self._has_capacity():
                return await self._handle_limit_exceeded()
        
        # Try to acquire semaphore (for concurrency limit)
        start_wait = time.monotonic()
        
        try:
            if self.config.strategy == RateLimitStrategy.WAIT:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=self.config.max_wait_seconds,
                )
            elif self.config.strategy == RateLimitStrategy.REJECT:
                if self._semaphore.locked():
                    self._stats.rejected_requests += 1
                    raise RateLimitExceeded(self.name)
                else:
                    await self._semaphore.acquire()
            else:
                await self._semaphore.acquire()
        except asyncio.TimeoutError:
            self._stats.rejected_requests += 1
            raise RateLimitExceeded(
                self.name, 
                retry_after=self._estimate_wait_time(),
            )
        
        wait_time = time.monotonic() - start_wait
        self._stats.total_wait_time += wait_time
        self._stats.successful_requests += 1
        
        # Record request
        now = time.monotonic()
        self._requests_second.append(now)
        self._requests_minute.append(now)
        self._requests_hour.append(now)
        
        # Consume token
        async with self._lock:
            self._tokens = max(0, self._tokens - 1)
        
        return True
    
    def release(self) -> None:
        """Release the semaphore after request completes."""
        self._semaphore.release()
    
    async def __aenter__(self) -> "RateLimiter":
        """Async context manager entry."""
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        self.release()
    
    def _update_tokens(self) -> None:
        """Update token count based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now
        
        # Add tokens based on rate
        tokens_to_add = elapsed * self.config.requests_per_second
        self._tokens = min(
            self.config.burst_size,
            self._tokens + tokens_to_add,
        )
    
    def _cleanup_request_history(self) -> None:
        """Remove old requests from tracking."""
        now = time.monotonic()
        
        # Keep last second
        while self._requests_second and (now - self._requests_second[0]) > 1.0:
            self._requests_second.popleft()
        
        # Keep last minute
        while self._requests_minute and (now - self._requests_minute[0]) > 60.0:
            self._requests_minute.popleft()
        
        # Keep last hour
        while self._requests_hour and (now - self._requests_hour[0]) > 3600.0:
            self._requests_hour.popleft()
    
    def _has_capacity(self) -> bool:
        """Check if we have capacity for another request."""
        # Check token bucket
        if self._tokens < 1:
            return False
        
        # Check per-second limit
        if len(self._requests_second) >= self.config.requests_per_second:
            return False
        
        # Check per-minute limit
        if len(self._requests_minute) >= self.config.requests_per_minute:
            return False
        
        # Check per-hour limit
        if len(self._requests_hour) >= self.config.requests_per_hour:
            return False
        
        return True
    
    async def _handle_limit_exceeded(self) -> bool:
        """Handle rate limit exceeded based on strategy."""
        strategy = self.config.strategy
        
        if strategy == RateLimitStrategy.REJECT:
            self._stats.rejected_requests += 1
            raise RateLimitExceeded(
                self.name,
                retry_after=self._estimate_wait_time(),
            )
        
        elif strategy == RateLimitStrategy.WAIT:
            # Wait for capacity
            wait_time = self._estimate_wait_time()
            if wait_time > self.config.max_wait_seconds:
                self._stats.rejected_requests += 1
                raise RateLimitExceeded(self.name, retry_after=wait_time)
            
            await asyncio.sleep(wait_time)
            return True
        
        elif strategy == RateLimitStrategy.QUEUE:
            if self._queue.full():
                self._stats.rejected_requests += 1
                raise QueueFullError(self.name, self.config.queue_size)
            
            self._stats.queued_requests += 1
            self._stats.current_queue_size = self._queue.qsize() + 1
            
            # Wait in queue (simplified - actual implementation would be more complex)
            await asyncio.sleep(self._estimate_wait_time())
            return True
        
        elif strategy == RateLimitStrategy.FALLBACK:
            return False
        
        return True
    
    def _estimate_wait_time(self) -> float:
        """Estimate how long until capacity is available."""
        # Simple estimate based on token refill
        if self._tokens >= 1:
            return 0.0
        
        tokens_needed = 1 - self._tokens
        return tokens_needed / self.config.requests_per_second
    
    def get_stats(self) -> RateLimiterStats:
        """Get current statistics."""
        return self._stats
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = RateLimiterStats()
    
    @property
    def available_tokens(self) -> float:
        """Get current available tokens."""
        return self._tokens
    
    @property
    def is_limited(self) -> bool:
        """Check if currently rate limited."""
        return not self._has_capacity()
