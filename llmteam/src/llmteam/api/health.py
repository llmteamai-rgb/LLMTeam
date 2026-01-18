"""
Health Check Module.

Provides health check functionality for the LLMTeam API.

Usage:
    from llmteam.api.health import HealthChecker, HealthStatus

    checker = HealthChecker()
    checker.add_check("database", db_health_check)
    checker.add_check("redis", redis_health_check)

    status = await checker.check_all()
    print(status.is_healthy)  # True/False
"""

import asyncio
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Optional
from datetime import datetime, timezone


class ComponentStatus(Enum):
    """Health status of a component."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health check result for a single component."""
    name: str
    status: ComponentStatus
    message: Optional[str] = None
    latency_ms: Optional[float] = None
    details: dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "details": self.details,
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class HealthStatus:
    """Overall health status."""
    status: ComponentStatus
    version: str
    uptime_seconds: float
    components: list[ComponentHealth] = field(default_factory=list)
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_healthy(self) -> bool:
        """Check if overall status is healthy."""
        return self.status == ComponentStatus.HEALTHY

    @property
    def is_degraded(self) -> bool:
        """Check if any component is degraded."""
        return self.status == ComponentStatus.DEGRADED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "version": self.version,
            "uptime_seconds": self.uptime_seconds,
            "components": [c.to_dict() for c in self.components],
            "checked_at": self.checked_at.isoformat(),
        }


# Type for health check functions
HealthCheckFunc = Callable[[], Awaitable[ComponentHealth]]


class HealthChecker:
    """
    Health checker for monitoring system components.

    Supports:
    - Multiple component checks
    - Timeout handling
    - Degraded status detection
    - Latency measurement
    - Async health checks
    """

    def __init__(
        self,
        version: str = "unknown",
        default_timeout: float = 5.0,
    ):
        """
        Initialize health checker.

        Args:
            version: Application version string
            default_timeout: Default timeout for health checks in seconds
        """
        self._version = version
        self._default_timeout = default_timeout
        self._checks: dict[str, HealthCheckFunc] = {}
        self._start_time = datetime.now(timezone.utc)

    @property
    def uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        return (datetime.now(timezone.utc) - self._start_time).total_seconds()

    def add_check(
        self,
        name: str,
        check_func: HealthCheckFunc,
    ) -> None:
        """
        Add a health check.

        Args:
            name: Component name
            check_func: Async function that returns ComponentHealth
        """
        self._checks[name] = check_func

    def remove_check(self, name: str) -> None:
        """Remove a health check."""
        self._checks.pop(name, None)

    async def check(
        self,
        name: str,
        timeout: Optional[float] = None,
    ) -> ComponentHealth:
        """
        Run a single health check.

        Args:
            name: Component name
            timeout: Timeout in seconds (uses default if not specified)

        Returns:
            ComponentHealth result
        """
        if name not in self._checks:
            return ComponentHealth(
                name=name,
                status=ComponentStatus.UNKNOWN,
                message=f"No health check registered for '{name}'",
            )

        timeout = timeout or self._default_timeout
        start_time = asyncio.get_event_loop().time()

        try:
            result = await asyncio.wait_for(
                self._checks[name](),
                timeout=timeout,
            )
            result.latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            return result

        except asyncio.TimeoutError:
            return ComponentHealth(
                name=name,
                status=ComponentStatus.UNHEALTHY,
                message=f"Health check timed out after {timeout}s",
                latency_ms=timeout * 1000,
            )
        except Exception as e:
            return ComponentHealth(
                name=name,
                status=ComponentStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                latency_ms=(asyncio.get_event_loop().time() - start_time) * 1000,
            )

    async def check_all(
        self,
        timeout: Optional[float] = None,
    ) -> HealthStatus:
        """
        Run all health checks.

        Args:
            timeout: Timeout per check in seconds

        Returns:
            Overall HealthStatus
        """
        if not self._checks:
            return HealthStatus(
                status=ComponentStatus.HEALTHY,
                version=self._version,
                uptime_seconds=self.uptime_seconds,
                components=[],
            )

        # Run all checks concurrently
        tasks = [
            self.check(name, timeout)
            for name in self._checks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        components: list[ComponentHealth] = []
        for result in results:
            if isinstance(result, Exception):
                components.append(ComponentHealth(
                    name="unknown",
                    status=ComponentStatus.UNHEALTHY,
                    message=str(result),
                ))
            else:
                components.append(result)

        # Determine overall status
        overall_status = self._calculate_overall_status(components)

        return HealthStatus(
            status=overall_status,
            version=self._version,
            uptime_seconds=self.uptime_seconds,
            components=components,
        )

    def _calculate_overall_status(
        self,
        components: list[ComponentHealth],
    ) -> ComponentStatus:
        """Calculate overall status from component statuses."""
        if not components:
            return ComponentStatus.HEALTHY

        statuses = [c.status for c in components]

        if all(s == ComponentStatus.HEALTHY for s in statuses):
            return ComponentStatus.HEALTHY
        elif any(s == ComponentStatus.UNHEALTHY for s in statuses):
            return ComponentStatus.UNHEALTHY
        elif any(s == ComponentStatus.DEGRADED for s in statuses):
            return ComponentStatus.DEGRADED
        else:
            return ComponentStatus.UNKNOWN

    async def liveness(self) -> ComponentHealth:
        """
        Simple liveness check.

        Returns healthy if the application is running.
        Used by Kubernetes liveness probes.
        """
        return ComponentHealth(
            name="liveness",
            status=ComponentStatus.HEALTHY,
            message="Application is running",
        )

    async def readiness(self) -> HealthStatus:
        """
        Readiness check.

        Checks all components to determine if the application
        is ready to receive traffic.
        Used by Kubernetes readiness probes.
        """
        return await self.check_all()


# Pre-built health check functions

async def create_database_check(
    get_connection: Callable[[], Awaitable[Any]],
    name: str = "database",
) -> HealthCheckFunc:
    """
    Create a database health check function.

    Args:
        get_connection: Function to get database connection
        name: Component name

    Returns:
        Health check function
    """
    async def check() -> ComponentHealth:
        try:
            conn = await get_connection()
            await conn.execute("SELECT 1")
            return ComponentHealth(
                name=name,
                status=ComponentStatus.HEALTHY,
                message="Database connection OK",
            )
        except Exception as e:
            return ComponentHealth(
                name=name,
                status=ComponentStatus.UNHEALTHY,
                message=f"Database error: {e}",
            )

    return check


async def create_redis_check(
    get_client: Callable[[], Awaitable[Any]],
    name: str = "redis",
) -> HealthCheckFunc:
    """
    Create a Redis health check function.

    Args:
        get_client: Function to get Redis client
        name: Component name

    Returns:
        Health check function
    """
    async def check() -> ComponentHealth:
        try:
            client = await get_client()
            await client.ping()
            return ComponentHealth(
                name=name,
                status=ComponentStatus.HEALTHY,
                message="Redis connection OK",
            )
        except Exception as e:
            return ComponentHealth(
                name=name,
                status=ComponentStatus.UNHEALTHY,
                message=f"Redis error: {e}",
            )

    return check


def create_memory_check(
    name: str = "memory",
    threshold_percent: float = 90.0,
) -> HealthCheckFunc:
    """
    Create a memory usage health check.

    Args:
        name: Component name
        threshold_percent: Memory usage threshold for degraded status

    Returns:
        Health check function
    """
    async def check() -> ComponentHealth:
        try:
            import psutil
            memory = psutil.virtual_memory()
            usage_percent = memory.percent

            if usage_percent >= threshold_percent:
                status = ComponentStatus.DEGRADED
                message = f"High memory usage: {usage_percent:.1f}%"
            else:
                status = ComponentStatus.HEALTHY
                message = f"Memory usage: {usage_percent:.1f}%"

            return ComponentHealth(
                name=name,
                status=status,
                message=message,
                details={
                    "total_mb": memory.total / (1024 * 1024),
                    "available_mb": memory.available / (1024 * 1024),
                    "percent": usage_percent,
                },
            )
        except ImportError:
            return ComponentHealth(
                name=name,
                status=ComponentStatus.UNKNOWN,
                message="psutil not installed",
            )
        except Exception as e:
            return ComponentHealth(
                name=name,
                status=ComponentStatus.UNHEALTHY,
                message=f"Memory check error: {e}",
            )

    return check


def create_disk_check(
    path: str = "/",
    name: str = "disk",
    threshold_percent: float = 90.0,
) -> HealthCheckFunc:
    """
    Create a disk usage health check.

    Args:
        path: Filesystem path to check
        name: Component name
        threshold_percent: Disk usage threshold for degraded status

    Returns:
        Health check function
    """
    async def check() -> ComponentHealth:
        try:
            import psutil
            disk = psutil.disk_usage(path)
            usage_percent = disk.percent

            if usage_percent >= threshold_percent:
                status = ComponentStatus.DEGRADED
                message = f"High disk usage: {usage_percent:.1f}%"
            else:
                status = ComponentStatus.HEALTHY
                message = f"Disk usage: {usage_percent:.1f}%"

            return ComponentHealth(
                name=name,
                status=status,
                message=message,
                details={
                    "path": path,
                    "total_gb": disk.total / (1024 * 1024 * 1024),
                    "free_gb": disk.free / (1024 * 1024 * 1024),
                    "percent": usage_percent,
                },
            )
        except ImportError:
            return ComponentHealth(
                name=name,
                status=ComponentStatus.UNKNOWN,
                message="psutil not installed",
            )
        except Exception as e:
            return ComponentHealth(
                name=name,
                status=ComponentStatus.UNHEALTHY,
                message=f"Disk check error: {e}",
            )

    return check
