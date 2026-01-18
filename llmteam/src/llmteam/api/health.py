"""
Health check endpoints for Kubernetes/load balancers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, Dict, Any
import asyncio

from llmteam import __version__


@dataclass
class HealthStatus:
    """Health check result."""
    status: str  # "healthy", "degraded", "unhealthy"
    version: str
    uptime_seconds: float
    checks: Dict[str, bool]
    timestamp: str
    
    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "version": self.version,
            "uptime_seconds": self.uptime_seconds,
            "checks": self.checks,
            "timestamp": self.timestamp,
        }


class HealthChecker:
    """
    Health checker for llmteam service.
    
    Usage with FastAPI:
        checker = HealthChecker()
        
        @app.get("/health")
        async def health():
            return (await checker.check()).to_dict()
        
        @app.get("/health/live")
        async def liveness():
            return {"status": "ok"}
        
        @app.get("/health/ready")
        async def readiness():
            result = await checker.check()
            if result.status == "unhealthy":
                raise HTTPException(503, result.to_dict())
            return result.to_dict()
    """
    
    def __init__(self) -> None:
        self._start_time = datetime.now()
        self._checks: Dict[str, Callable] = {}
        self._version = __version__
    
    def register_check(self, name: str, check_fn: Callable) -> None:
        """Register a health check function."""
        self._checks[name] = check_fn
    
    async def check(self) -> HealthStatus:
        """Run all health checks."""
        checks = {}
        all_healthy = True
        
        for name, check_fn in self._checks.items():
            try:
                # Handle both sync and async check functions
                if asyncio.iscoroutinefunction(check_fn):
                    result = await check_fn()
                else:
                    result = check_fn()
                    
                is_healthy = bool(result)
                checks[name] = is_healthy
                if not is_healthy:
                    all_healthy = False
            except Exception:
                checks[name] = False
                all_healthy = False
        
        uptime = (datetime.now() - self._start_time).total_seconds()
        
        return HealthStatus(
            status="healthy" if all_healthy else "unhealthy",
            version=self._version,
            uptime_seconds=uptime,
            checks=checks,
            timestamp=datetime.now().isoformat(),
        )
