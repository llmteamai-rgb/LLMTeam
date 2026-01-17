"""
License tier decorators for Open Core model.

Usage:
    from llmteam.licensing import professional_only, enterprise_only

    @enterprise_only
    class TenantManager:
        ...

    @professional_only
    class ProcessMiningEngine:
        ...
"""

import asyncio
from functools import wraps
from typing import Callable, TypeVar, Any

from .models import LicenseTier

F = TypeVar('F', bound=Callable[..., Any])


class FeatureNotLicensedError(Exception):
    """Feature requires higher license tier."""

    def __init__(self, feature: str, required_tier: LicenseTier):
        self.feature = feature
        self.required_tier = required_tier

        tier_info = {
            LicenseTier.PROFESSIONAL: (
                "Professional",
                "$99/month",
                "https://llmteam.ai/pricing#professional"
            ),
            LicenseTier.ENTERPRISE: (
                "Enterprise",
                "Custom pricing",
                "https://llmteam.ai/pricing#enterprise"
            ),
        }

        info = tier_info.get(required_tier, ("Unknown", "", ""))

        super().__init__(
            f"\n"
            f"╔══════════════════════════════════════════════════════════════╗\n"
            f"║  🔒 FEATURE LOCKED: {feature:<40} ║\n"
            f"╠══════════════════════════════════════════════════════════════╣\n"
            f"║  This feature requires LLMTeam {info[0]} license.           ║\n"
            f"║                                                              ║\n"
            f"║  Upgrade: {info[2]:<50} ║\n"
            f"║  Contact: sales@llmteam.ai                                   ║\n"
            f"╚══════════════════════════════════════════════════════════════╝\n"
        )


def require_tier(tier: LicenseTier) -> Callable[[F], F]:
    """
    Decorator to require specific license tier for a function.

    Args:
        tier: Minimum required license tier

    Returns:
        Decorated function that checks license before execution

    Raises:
        FeatureNotLicensedError: If license tier is insufficient

    Example:
        @require_tier(LicenseTier.PROFESSIONAL)
        def my_pro_feature():
            ...
    """
    def decorator(func: F) -> F:
        feature_name = func.__qualname__

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            from .manager import get_license_manager
            manager = get_license_manager()
            if not manager.has_tier(tier):
                raise FeatureNotLicensedError(feature_name, tier)
            return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            from .manager import get_license_manager
            manager = get_license_manager()
            if not manager.has_tier(tier):
                raise FeatureNotLicensedError(feature_name, tier)
            return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def require_professional(func: F) -> F:
    """
    Decorator for Professional tier functions.

    Example:
        @require_professional
        def export_to_postgres():
            ...
    """
    return require_tier(LicenseTier.PROFESSIONAL)(func)


def require_enterprise(func: F) -> F:
    """
    Decorator for Enterprise tier functions.

    Example:
        @require_enterprise
        def manage_tenants():
            ...
    """
    return require_tier(LicenseTier.ENTERPRISE)(func)


# === Class decorators for protecting entire classes ===


def professional_only(cls):
    """
    Class decorator to make entire class Professional-only.

    Example:
        @professional_only
        class ProcessMiningEngine:
            ...
    """
    original_init = cls.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        from .manager import get_license_manager
        manager = get_license_manager()
        if not manager.has_tier(LicenseTier.PROFESSIONAL):
            raise FeatureNotLicensedError(cls.__name__, LicenseTier.PROFESSIONAL)
        original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    cls._requires_tier = LicenseTier.PROFESSIONAL
    return cls


def enterprise_only(cls):
    """
    Class decorator to make entire class Enterprise-only.

    Example:
        @enterprise_only
        class TenantManager:
            ...
    """
    original_init = cls.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        from .manager import get_license_manager
        manager = get_license_manager()
        if not manager.has_tier(LicenseTier.ENTERPRISE):
            raise FeatureNotLicensedError(cls.__name__, LicenseTier.ENTERPRISE)
        original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    cls._requires_tier = LicenseTier.ENTERPRISE
    return cls
