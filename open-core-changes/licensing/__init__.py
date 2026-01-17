# src/llmteam/licensing/__init__.py
"""
LLMTeam Licensing Module.

Open Core licensing system with three tiers:
- COMMUNITY: Free, basic features
- PROFESSIONAL: $99/month, advanced features
- ENTERPRISE: Custom pricing, all features + support

Usage:
    import llmteam
    
    # Check current tier
    print(llmteam.get_tier())  # LicenseTier.COMMUNITY
    
    # Activate license
    llmteam.activate("LLMT-PRO-XXXX-20261231")
    
    # Check features
    if llmteam.has_feature("process_mining"):
        engine = ProcessMiningEngine()
"""

from .models import LicenseTier
from .manager import (
    LicenseManager,
    License,
    LicenseLimits,
    LicenseValidationError,
    LicenseExpiredError,
    get_license_manager,
    activate,
    get_tier,
    has_feature,
    print_license_status,
    TIER_LIMITS,
)
from .decorators import (
    FeatureNotLicensedError,
    require_tier,
    require_professional,
    require_enterprise,
    professional_only,
    enterprise_only,
)

__all__ = [
    # Models
    "LicenseTier",
    "License",
    "LicenseLimits",
    
    # Manager
    "LicenseManager",
    "get_license_manager",
    "activate",
    "get_tier",
    "has_feature",
    "print_license_status",
    "TIER_LIMITS",
    
    # Decorators
    "FeatureNotLicensedError",
    "require_tier",
    "require_professional",
    "require_enterprise",
    "professional_only",
    "enterprise_only",
    
    # Exceptions
    "LicenseValidationError",
    "LicenseExpiredError",
]
