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

    # Use decorators
    from llmteam.licensing import professional_only, enterprise_only

    @professional_only
    class MyProFeature:
        ...
"""

from llmteam.licensing.models import (
    LicenseTier,
    LicenseLimits,
    License,
    LICENSE_LIMITS,
    TIER_LIMITS,
)

from llmteam.licensing.manager import (
    LicenseManager,
    LicenseValidationError,
    LicenseExpiredError,
    get_license_manager,
    activate,
    get_tier,
    has_feature,
    print_license_status,
)

from llmteam.licensing.decorators import (
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
    "LicenseLimits",
    "License",
    "LICENSE_LIMITS",
    "TIER_LIMITS",

    # Manager
    "LicenseManager",
    "get_license_manager",
    "activate",
    "get_tier",
    "has_feature",
    "print_license_status",

    # Exceptions
    "LicenseValidationError",
    "LicenseExpiredError",
    "FeatureNotLicensedError",

    # Decorators
    "require_tier",
    "require_professional",
    "require_enterprise",
    "professional_only",
    "enterprise_only",
]
