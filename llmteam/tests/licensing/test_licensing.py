"""
Tests for licensing module.

Tests cover:
- License tiers and limits
- Limit checking
- Feature availability
- Enforcement on ExecutorConfig
"""

import pytest

from llmteam.licensing import (
    LicenseTier,
    LicenseLimits,
    LICENSE_LIMITS,
    LicenseManager,
)

from llmteam.execution import ExecutorConfig, ExecutionMode


class TestLicenseLimits:
    """Tests for LicenseLimits dataclass."""

    def test_license_limits_structure(self):
        """Test LicenseLimits structure."""
        limits = LicenseLimits(
            max_concurrent_pipelines=5,
            max_agents_per_pipeline=20,
            max_parallel_agents=10,
            features={"parallel_execution"},
        )

        assert limits.max_concurrent_pipelines == 5
        assert limits.max_agents_per_pipeline == 20
        assert limits.max_parallel_agents == 10
        assert "parallel_execution" in limits.features


class TestLicenseTier:
    """Tests for LicenseTier enum."""

    def test_license_tier_values(self):
        """Test license tier enum values."""
        assert LicenseTier.COMMUNITY.value == "community"
        assert LicenseTier.PROFESSIONAL.value == "professional"
        assert LicenseTier.ENTERPRISE.value == "enterprise"


class TestLICENSE_LIMITS:
    """Tests for LICENSE_LIMITS configuration."""

    def test_all_tiers_defined(self):
        """Test that all tiers have limits defined."""
        assert LicenseTier.COMMUNITY in LICENSE_LIMITS
        assert LicenseTier.PROFESSIONAL in LICENSE_LIMITS
        assert LicenseTier.ENTERPRISE in LICENSE_LIMITS

    def test_community_limits(self):
        """Test Community tier limits."""
        limits = LICENSE_LIMITS[LicenseTier.COMMUNITY]

        assert limits.max_concurrent_pipelines == 1
        assert limits.max_agents_per_pipeline == 5
        assert limits.max_parallel_agents == 2
        assert "basic_agents" in limits.features
        assert "sequential_execution" in limits.features
        assert "parallel_execution" not in limits.features

    def test_professional_limits(self):
        """Test Professional tier limits."""
        limits = LICENSE_LIMITS[LicenseTier.PROFESSIONAL]

        assert limits.max_concurrent_pipelines == 5
        assert limits.max_agents_per_pipeline == 20
        assert limits.max_parallel_agents == 10
        assert "parallel_execution" in limits.features
        assert "process_mining" in limits.features

    def test_enterprise_limits(self):
        """Test Enterprise tier limits (unlimited)."""
        limits = LICENSE_LIMITS[LicenseTier.ENTERPRISE]

        assert limits.max_concurrent_pipelines == 999999
        assert limits.max_agents_per_pipeline == 999999
        assert limits.max_parallel_agents == 999999
        assert "*" in limits.features


class TestLicenseManager:
    """Tests for LicenseManager class."""

    def test_create_manager_community(self):
        """Test creating manager with Community tier."""
        manager = LicenseManager(LicenseTier.COMMUNITY)

        assert manager.tier == LicenseTier.COMMUNITY
        assert manager.limits.max_concurrent_pipelines == 1

    def test_create_manager_default(self):
        """Test creating manager with default tier."""
        manager = LicenseManager()

        assert manager.tier == LicenseTier.COMMUNITY

    def test_check_concurrent_limit(self):
        """Test checking concurrent pipeline limit."""
        manager = LicenseManager(LicenseTier.COMMUNITY)

        # Community allows 1 concurrent pipeline
        assert manager.check_concurrent_limit(0) is True
        assert manager.check_concurrent_limit(1) is False
        assert manager.check_concurrent_limit(2) is False

    def test_check_agents_limit(self):
        """Test checking agents per pipeline limit."""
        manager = LicenseManager(LicenseTier.COMMUNITY)

        # Community allows 5 agents per pipeline
        assert manager.check_agents_limit(3) is True
        assert manager.check_agents_limit(5) is True
        assert manager.check_agents_limit(6) is False

    def test_check_parallel_limit(self):
        """Test checking parallel execution limit."""
        manager = LicenseManager(LicenseTier.PROFESSIONAL)

        # Professional allows 10 parallel agents
        assert manager.check_parallel_limit(5) is True
        assert manager.check_parallel_limit(10) is True
        assert manager.check_parallel_limit(11) is False

    def test_check_feature_available(self):
        """Test checking if a feature is available."""
        manager = LicenseManager(LicenseTier.PROFESSIONAL)

        assert manager.check_feature("parallel_execution") is True
        assert manager.check_feature("process_mining") is True
        assert manager.check_feature("basic_agents") is True

    def test_check_feature_unavailable(self):
        """Test checking unavailable feature."""
        manager = LicenseManager(LicenseTier.COMMUNITY)

        assert manager.check_feature("parallel_execution") is False
        assert manager.check_feature("process_mining") is False

    def test_check_feature_enterprise_wildcard(self):
        """Test Enterprise tier with wildcard features."""
        manager = LicenseManager(LicenseTier.ENTERPRISE)

        assert manager.check_feature("parallel_execution") is True
        assert manager.check_feature("process_mining") is True
        assert manager.check_feature("any_feature") is True

    def test_enforce_community_limits(self):
        """Test enforcing Community limits on executor config."""
        manager = LicenseManager(LicenseTier.COMMUNITY)

        config = ExecutorConfig(max_concurrent=20)
        enforced = manager.enforce(config)

        # Community limits to 2 parallel agents
        assert enforced.max_concurrent == 2

    def test_enforce_professional_limits(self):
        """Test enforcing Professional limits on executor config."""
        manager = LicenseManager(LicenseTier.PROFESSIONAL)

        config = ExecutorConfig(max_concurrent=20)
        enforced = manager.enforce(config)

        # Professional limits to 10 parallel agents
        assert enforced.max_concurrent == 10

    def test_enforce_enterprise_no_limit(self):
        """Test enforcing Enterprise limits (no enforcement)."""
        manager = LicenseManager(LicenseTier.ENTERPRISE)

        config = ExecutorConfig(max_concurrent=20)
        enforced = manager.enforce(config)

        # Enterprise allows 20
        assert enforced.max_concurrent == 20

    def test_enforce_preserves_other_config(self):
        """Test that enforce preserves other config settings."""
        manager = LicenseManager(LicenseTier.PROFESSIONAL)

        config = ExecutorConfig(
            mode=ExecutionMode.PARALLEL,
            max_concurrent=20,
            queue_size=200,
            task_timeout=600.0,
        )

        enforced = manager.enforce(config)

        assert enforced.mode == ExecutionMode.PARALLEL
        assert enforced.queue_size == 200
        assert enforced.task_timeout == 600.0

    def test_get_tier(self):
        """Test getting license tier."""
        manager = LicenseManager(LicenseTier.PROFESSIONAL)

        assert manager.get_tier() == LicenseTier.PROFESSIONAL

    def test_get_limits(self):
        """Test getting limits as dictionary."""
        manager = LicenseManager(LicenseTier.PROFESSIONAL)

        limits = manager.get_limits()

        assert limits["tier"] == "professional"
        assert limits["max_concurrent_pipelines"] == 5
        assert limits["max_agents_per_pipeline"] == 20
        assert limits["max_parallel_agents"] == 10
        assert "parallel_execution" in limits["features"]
