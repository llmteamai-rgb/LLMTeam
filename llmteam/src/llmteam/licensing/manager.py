"""
License manager for LLMTeam Open Core model.

Supports:
- Offline license validation (key format)
- Environment variable configuration
- License file configuration

Usage:
    import llmteam

    # Activate via code
    llmteam.activate("LLMT-PRO-XXXX-20261231")

    # Or via environment variable
    # export LLMTEAM_LICENSE_KEY=LLMT-PRO-XXXX-20261231

    # Or via file
    # ~/.llmteam/license.key
"""

import os
import hashlib
import hmac
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from pathlib import Path

from .models import LicenseTier, LicenseLimits, License, LICENSE_LIMITS

if TYPE_CHECKING:
    from llmteam.execution.config import ExecutorConfig

logger = logging.getLogger(__name__)


# === License Key Format ===
# LLMT-{TIER}-{HASH}-{EXPIRY}
# Example: LLMT-PRO-A1B2C3D4-20261231
#
# TIER: COM (Community), PRO (Professional), ENT (Enterprise)
# HASH: Owner identifier hash
# EXPIRY: YYYYMMDD format


class LicenseValidationError(Exception):
    """License validation failed."""
    pass


class LicenseExpiredError(Exception):
    """License has expired."""
    pass


class LicenseManager:
    """
    Manager for license validation and limit enforcement.

    Singleton pattern - use LicenseManager.instance() or get_license_manager().

    Example:
        # Get singleton
        manager = LicenseManager.instance()

        # Activate license
        manager.activate("LLMT-PRO-XXXX-20261231")

        # Check tier
        if manager.has_tier(LicenseTier.PROFESSIONAL):
            # Professional features available
            pass

        # Check feature
        if manager.has_feature("process_mining"):
            # Process mining available
            pass
    """

    _instance: Optional["LicenseManager"] = None

    # Secret for offline validation
    _VALIDATION_SECRET = b"llmteam-open-core-2025"

    def __init__(self, tier: Optional[LicenseTier] = None):
        """
        Initialize license manager.

        Args:
            tier: Optional tier to set directly (for backwards compatibility)
        """
        self.license: Optional[License] = None
        self._tier = tier  # For backwards compatibility

        # If tier provided directly, use it (backwards compat)
        if tier is not None:
            self._tier = tier
        else:
            # Try to load license from environment/file
            self._load_license()

    @classmethod
    def instance(cls) -> "LicenseManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def _load_license(self) -> None:
        """Load license from environment or file."""

        # 1. Try environment variable
        key = os.environ.get("LLMTEAM_LICENSE_KEY")

        # 2. Try license file
        if not key:
            license_paths = [
                Path.home() / ".llmteam" / "license.key",
                Path(".llmteam") / "license.key",
                Path("license.key"),
            ]

            for path in license_paths:
                if path.exists():
                    try:
                        key = path.read_text().strip()
                        logger.info(f"Loaded license from {path}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to read license from {path}: {e}")

        # 3. Activate if found
        if key:
            try:
                self.activate(key)
            except Exception as e:
                logger.warning(f"Failed to activate license: {e}")

    def activate(self, key: str) -> License:
        """
        Activate a license key.

        Args:
            key: License key in format LLMT-TIER-HASH-EXPIRY

        Returns:
            Activated License object

        Raises:
            LicenseValidationError: If key is invalid
            LicenseExpiredError: If license has expired
        """
        # Clean the key
        key = key.strip().upper()

        # Validate format
        license = self._validate_key(key)

        if license is None:
            raise LicenseValidationError(
                f"Invalid license key. Please check your key or contact support@llmteam.ai"
            )

        if license.is_expired:
            raise LicenseExpiredError(
                f"License expired on {license.expires_at.date()}. "
                f"Renew at https://llmteam.ai/account"
            )

        self.license = license
        self._tier = license.tier
        logger.info(
            f"License activated: {license.tier.value} "
            f"(expires in {license.days_remaining} days)"
        )

        return license

    def deactivate(self) -> None:
        """Deactivate current license."""
        self.license = None
        self._tier = None
        logger.info("License deactivated")

    def _validate_key(self, key: str) -> Optional[License]:
        """
        Validate license key (offline validation).

        Key format: LLMT-{TIER}-{HASH}-{EXPIRY}
        """
        try:
            parts = key.split("-")

            # Check format
            if len(parts) < 4:
                return None

            if parts[0] != "LLMT":
                return None

            # Parse tier
            tier_map = {
                "COM": LicenseTier.COMMUNITY,
                "PRO": LicenseTier.PROFESSIONAL,
                "ENT": LicenseTier.ENTERPRISE,
            }

            tier = tier_map.get(parts[1])
            if tier is None:
                return None

            # Parse expiry
            try:
                expires_at = datetime.strptime(parts[3], "%Y%m%d")
            except ValueError:
                return None

            return License(
                key=key,
                tier=tier,
                owner=parts[2],  # Owner hash
                owner_email="",
                expires_at=expires_at,
                issued_at=datetime.now(),
                features=self._get_tier_features(tier),
            )

        except Exception as e:
            logger.error(f"License validation error: {e}")
            return None

    def _get_tier_features(self, tier: LicenseTier) -> List[str]:
        """Get list of features for tier."""
        features = ["basic", "memory_store"]

        if tier in (LicenseTier.PROFESSIONAL, LicenseTier.ENTERPRISE):
            features.extend([
                "process_mining",
                "postgres_store",
                "redis_store",
                "human_interaction",
                "external_actions",
            ])

        if tier == LicenseTier.ENTERPRISE:
            features.extend([
                "multi_tenant",
                "audit_trail",
                "sso",
                "priority_support",
            ])

        return features

    # === Tier checks ===

    def has_tier(self, required: LicenseTier) -> bool:
        """
        Check if current license has at least the required tier.

        Community tier is always available (no license needed).
        """
        if required == LicenseTier.COMMUNITY:
            return True

        current = self.current_tier
        if current == LicenseTier.COMMUNITY:
            return False

        # Check if license is expired
        if self.license and self.license.is_expired:
            return False

        tier_order = [
            LicenseTier.COMMUNITY,
            LicenseTier.PROFESSIONAL,
            LicenseTier.ENTERPRISE,
        ]

        try:
            current_idx = tier_order.index(current)
            required_idx = tier_order.index(required)
            return current_idx >= required_idx
        except ValueError:
            return False

    def has_feature(self, feature: str) -> bool:
        """Check if current license has specific feature."""
        # Enterprise has all features (wildcard)
        if "*" in self.limits.features:
            return True

        # Check in limits.features (always available)
        if feature in self.limits.features:
            return True

        # Check in license.features if license exists
        if self.license and feature in self.license.features:
            return True

        # Default community features
        if self.license is None and self._tier is None:
            return feature in ["basic", "memory_store"]

        return False

    @property
    def current_tier(self) -> LicenseTier:
        """Get current license tier."""
        # Check for directly set tier (backwards compat)
        if self._tier is not None:
            return self._tier

        # Check license
        if self.license and not self.license.is_expired:
            return self.license.tier

        return LicenseTier.COMMUNITY

    @property
    def tier(self) -> LicenseTier:
        """Alias for current_tier (backwards compatibility)."""
        return self.current_tier

    @property
    def limits(self) -> LicenseLimits:
        """Get limits for current tier."""
        return LICENSE_LIMITS[self.current_tier]

    # === Legacy methods for backwards compatibility ===

    def check_concurrent_limit(self, current: int) -> bool:
        """Check if current concurrent pipeline count is within limits."""
        return current < self.limits.max_concurrent_pipelines

    def check_agents_limit(self, count: int) -> bool:
        """Check if agent count is within limits."""
        return count <= self.limits.max_agents_per_pipeline

    def check_parallel_limit(self, count: int) -> bool:
        """Check if parallel agent count is within limits."""
        return count <= self.limits.max_parallel_agents

    def check_feature(self, feature: str) -> bool:
        """Check if a feature is available for this license tier."""
        return self.has_feature(feature)

    def check_limit(self, limit_name: str, current_value: int) -> bool:
        """Check if current value is within limit."""
        limit = getattr(self.limits, limit_name, None)
        if limit is None:
            return True
        return current_value < limit

    def enforce(self, executor_config: "ExecutorConfig") -> "ExecutorConfig":
        """
        Enforce license limits on executor configuration.

        Creates a new ExecutorConfig with limits applied.
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
        """Get current license tier."""
        return self.current_tier

    def get_limits(self) -> dict:
        """Get current limits as dictionary."""
        return {
            "tier": self.current_tier.value,
            "max_concurrent_pipelines": self.limits.max_concurrent_pipelines,
            "max_agents_per_pipeline": self.limits.max_agents_per_pipeline,
            "max_parallel_agents": self.limits.max_parallel_agents,
            "features": list(self.limits.features),
        }

    # === Info ===

    def get_info(self) -> dict:
        """Get current license info."""
        return {
            "tier": self.current_tier.value,
            "license": self.license.to_dict() if self.license else None,
            "limits": {
                "max_concurrent_pipelines": self.limits.max_concurrent_pipelines,
                "max_agents_per_pipeline": self.limits.max_agents_per_pipeline,
                "max_parallel_agents": self.limits.max_parallel_agents,
                "max_teams": self.limits.max_teams,
            },
            "features": {
                "process_mining": self.limits.process_mining,
                "audit_trail": self.limits.audit_trail,
                "multi_tenant": self.limits.multi_tenant,
                "postgres_store": self.limits.postgres_store,
                "human_interaction": self.limits.human_interaction,
            },
        }

    def print_status(self) -> None:
        """Print license status to console."""
        info = self.get_info()

        print("\n" + "=" * 60)
        print("  LLMTeam License Status")
        print("=" * 60)
        print(f"  Tier: {info['tier'].upper()}")

        if self.license:
            print(f"  Expires: {self.license.expires_at.date()} ({self.license.days_remaining} days)")
        else:
            print("  Status: Community (no license)")

        print("\n  Limits:")
        for k, v in info['limits'].items():
            print(f"    • {k}: {v}")

        print("\n  Features:")
        for k, v in info['features'].items():
            status = "✅" if v else "❌"
            print(f"    {status} {k}")

        print("=" * 60 + "\n")


# === Module-level functions ===

_manager: Optional[LicenseManager] = None


def get_license_manager() -> LicenseManager:
    """Get the global license manager instance."""
    global _manager
    if _manager is None:
        _manager = LicenseManager.instance()
    return _manager


def activate(key: str) -> License:
    """
    Activate a license key.

    This is the main entry point for license activation.

    Args:
        key: License key

    Returns:
        License object

    Example:
        import llmteam
        llmteam.activate("LLMT-PRO-XXXX-20261231")
    """
    return get_license_manager().activate(key)


def get_tier() -> LicenseTier:
    """Get current license tier."""
    return get_license_manager().current_tier


def has_feature(feature: str) -> bool:
    """Check if feature is available."""
    return get_license_manager().has_feature(feature)


def print_license_status() -> None:
    """Print current license status."""
    get_license_manager().print_status()
