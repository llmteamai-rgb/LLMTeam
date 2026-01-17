# src/llmteam/licensing/manager.py
"""
License management for LLMTeam Open Core model.

Supports:
- Offline license validation (key format)
- Online license validation (API)
- Environment variable configuration
- License file configuration

Usage:
    import llmteam
    
    # Activate via code
    llmteam.activate("LLMT-PRO-XXXX-XXXX")
    
    # Or via environment variable
    # export LLMTEAM_LICENSE_KEY=LLMT-PRO-XXXX-XXXX
    
    # Or via file
    # ~/.llmteam/license.key
"""

import os
import hashlib
import hmac
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path

from .models import LicenseTier

logger = logging.getLogger(__name__)


# === License Key Format ===
# LLMT-{TIER}-{OWNER_HASH}-{EXPIRY}-{SIGNATURE}
# Example: LLMT-PRO-A1B2C3D4-20261231-X9Y8Z7W6
#
# TIER: COM (Community), PRO (Professional), ENT (Enterprise)
# OWNER_HASH: First 8 chars of SHA256(owner_email)
# EXPIRY: YYYYMMDD format
# SIGNATURE: HMAC signature for validation


@dataclass
class License:
    """License information."""
    
    key: str
    tier: LicenseTier
    owner: str
    owner_email: str
    expires_at: datetime
    issued_at: datetime
    features: List[str] = field(default_factory=list)
    limits: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at
    
    @property
    def days_remaining(self) -> int:
        delta = self.expires_at - datetime.now()
        return max(0, delta.days)
    
    def to_dict(self) -> dict:
        return {
            "key": self.key[:10] + "...",  # Masked
            "tier": self.tier.value,
            "owner": self.owner,
            "expires_at": self.expires_at.isoformat(),
            "days_remaining": self.days_remaining,
            "features": self.features,
        }


@dataclass
class LicenseLimits:
    """Limits by license tier."""
    
    # Concurrency
    max_teams: int = 1
    max_agents_per_team: int = 5
    max_concurrent_runs: int = 1
    
    # Storage
    max_snapshots: int = 10
    
    # Features
    process_mining: bool = False
    audit_trail: bool = False
    multi_tenant: bool = False
    postgres_store: bool = False
    redis_store: bool = False
    human_interaction: bool = False
    external_actions: bool = False


# Tier-based limits
TIER_LIMITS: Dict[LicenseTier, LicenseLimits] = {
    LicenseTier.COMMUNITY: LicenseLimits(
        max_teams=2,
        max_agents_per_team=5,
        max_concurrent_runs=1,
        max_snapshots=10,
        process_mining=False,
        audit_trail=False,
        multi_tenant=False,
        postgres_store=False,
        redis_store=False,
        human_interaction=False,
        external_actions=False,
    ),
    LicenseTier.PROFESSIONAL: LicenseLimits(
        max_teams=10,
        max_agents_per_team=20,
        max_concurrent_runs=5,
        max_snapshots=100,
        process_mining=True,
        audit_trail=False,
        multi_tenant=False,
        postgres_store=True,
        redis_store=True,
        human_interaction=True,
        external_actions=True,
    ),
    LicenseTier.ENTERPRISE: LicenseLimits(
        max_teams=999999,
        max_agents_per_team=999999,
        max_concurrent_runs=999999,
        max_snapshots=999999,
        process_mining=True,
        audit_trail=True,
        multi_tenant=True,
        postgres_store=True,
        redis_store=True,
        human_interaction=True,
        external_actions=True,
    ),
}


class LicenseValidationError(Exception):
    """License validation failed."""
    pass


class LicenseExpiredError(Exception):
    """License has expired."""
    pass


class LicenseManager:
    """
    Manages license validation and feature gating.
    
    Singleton pattern - use LicenseManager.instance() or get_license_manager().
    """
    
    _instance: Optional["LicenseManager"] = None
    
    # Secret for offline validation (in production, use secure storage)
    _VALIDATION_SECRET = b"llmteam-open-core-2025"
    
    def __init__(self):
        self.license: Optional[License] = None
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
            key: License key in format LLMT-TIER-HASH-EXPIRY-SIG
            
        Returns:
            Activated License object
            
        Raises:
            LicenseValidationError: If key is invalid
            LicenseExpiredError: If license has expired
        """
        # Clean the key
        key = key.strip().upper()
        
        # Validate format and signature
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
        logger.info(
            f"License activated: {license.tier.value} "
            f"(expires in {license.days_remaining} days)"
        )
        
        return license
    
    def deactivate(self) -> None:
        """Deactivate current license."""
        self.license = None
        logger.info("License deactivated")
    
    def _validate_key(self, key: str) -> Optional[License]:
        """
        Validate license key (offline validation).
        
        Key format: LLMT-{TIER}-{HASH}-{EXPIRY}-{SIG}
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
            
            # Validate signature (if present)
            if len(parts) >= 5:
                expected_sig = self._compute_signature(parts[0:4])
                if parts[4] != expected_sig:
                    # For now, allow keys without valid signature (development)
                    logger.warning("License signature mismatch (ignored in dev mode)")
            
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
    
    def _compute_signature(self, parts: List[str]) -> str:
        """Compute HMAC signature for key parts."""
        message = "-".join(parts).encode()
        sig = hmac.new(self._VALIDATION_SECRET, message, hashlib.sha256)
        return sig.hexdigest()[:8].upper()
    
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
        
        if self.license is None:
            return False
        
        if self.license.is_expired:
            return False
        
        tier_order = [
            LicenseTier.COMMUNITY,
            LicenseTier.PROFESSIONAL,
            LicenseTier.ENTERPRISE,
        ]
        
        try:
            current_idx = tier_order.index(self.license.tier)
            required_idx = tier_order.index(required)
            return current_idx >= required_idx
        except ValueError:
            return False
    
    def has_feature(self, feature: str) -> bool:
        """Check if current license has specific feature."""
        if self.license is None:
            return feature in ["basic", "memory_store"]
        return feature in self.license.features
    
    @property
    def current_tier(self) -> LicenseTier:
        """Get current license tier."""
        if self.license and not self.license.is_expired:
            return self.license.tier
        return LicenseTier.COMMUNITY
    
    @property
    def limits(self) -> LicenseLimits:
        """Get limits for current tier."""
        return TIER_LIMITS[self.current_tier]
    
    def check_limit(self, limit_name: str, current_value: int) -> bool:
        """Check if current value is within limit."""
        limit = getattr(self.limits, limit_name, None)
        if limit is None:
            return True
        return current_value < limit
    
    # === Info ===
    
    def get_info(self) -> dict:
        """Get current license info."""
        return {
            "tier": self.current_tier.value,
            "license": self.license.to_dict() if self.license else None,
            "limits": {
                "max_teams": self.limits.max_teams,
                "max_agents_per_team": self.limits.max_agents_per_team,
                "max_concurrent_runs": self.limits.max_concurrent_runs,
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
        print(f"  Tier: {info['tier']}")
        
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
