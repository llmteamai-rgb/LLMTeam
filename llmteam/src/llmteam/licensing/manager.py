"""
License manager for llmteam.

Manages license validation and enforcement of limits.
"""

from typing import TYPE_CHECKING

from llmteam.licensing.models import LicenseTier, LICENSE_LIMITS

if TYPE_CHECKING:
    from llmteam.execution.config import ExecutorConfig


class LicenseManager:
    """
    Manager for license validation and limit enforcement.

    Example:
        license_mgr = LicenseManager(LicenseTier.PROFESSIONAL)

        # Check limits
        if license_mgr.check_concurrent_limit(current=3):
            # OK to run 3 concurrent pipelines
            pass

        # Check features
        if license_mgr.check_feature("process_mining"):
            # Process mining is available
            pass

        # Enforce limits on executor config
        config = ExecutorConfig(max_concurrent=20)
        config = license_mgr.enforce(config)  # Limited to 10 for Professional
    """

    def __init__(self, tier: LicenseTier = LicenseTier.COMMUNITY):
        """
        Initialize license manager.

        Args:
            tier: License tier to use
        """
        self.tier = tier
        self.limits = LICENSE_LIMITS[tier]

    def check_concurrent_limit(self, current: int) -> bool:
        """
        Check if current concurrent pipeline count is within limits.

        Args:
            current: Current number of concurrent pipelines

        Returns:
            True if within limit, False otherwise
        """
        return current < self.limits.max_concurrent_pipelines

    def check_agents_limit(self, count: int) -> bool:
        """
        Check if agent count is within limits.

        Args:
            count: Number of agents in pipeline

        Returns:
            True if within limit, False otherwise
        """
        return count <= self.limits.max_agents_per_pipeline

    def check_parallel_limit(self, count: int) -> bool:
        """
        Check if parallel agent count is within limits.

        Args:
            count: Number of agents to execute in parallel

        Returns:
            True if within limit, False otherwise
        """
        return count <= self.limits.max_parallel_agents

    def check_feature(self, feature: str) -> bool:
        """
        Check if a feature is available for this license tier.

        Args:
            feature: Feature name to check

        Returns:
            True if feature is available, False otherwise
        """
        return "*" in self.limits.features or feature in self.limits.features

    def enforce(self, executor_config: "ExecutorConfig") -> "ExecutorConfig":
        """
        Enforce license limits on executor configuration.

        Creates a new ExecutorConfig with limits applied.

        Args:
            executor_config: Original executor configuration

        Returns:
            New ExecutorConfig with limits enforced
        """
        from llmteam.execution.config import ExecutorConfig

        return ExecutorConfig(
            mode=executor_config.mode,
            max_concurrent=min(
                executor_config.max_concurrent,
                self.limits.max_parallel_agents,
            ),
            queue_size=executor_config.queue_size,
            task_timeout=executor_config.task_timeout,
            total_timeout=executor_config.total_timeout,
            max_retries=executor_config.max_retries,
            retry_delay=executor_config.retry_delay,
            enable_backpressure=executor_config.enable_backpressure,
            backpressure_threshold=executor_config.backpressure_threshold,
        )

    def get_tier(self) -> LicenseTier:
        """
        Get current license tier.

        Returns:
            License tier
        """
        return self.tier

    def get_limits(self) -> dict:
        """
        Get current limits as dictionary.

        Returns:
            Dictionary with limit information
        """
        return {
            "tier": self.tier.value,
            "max_concurrent_pipelines": self.limits.max_concurrent_pipelines,
            "max_agents_per_pipeline": self.limits.max_agents_per_pipeline,
            "max_parallel_agents": self.limits.max_parallel_agents,
            "features": list(self.limits.features),
        }
