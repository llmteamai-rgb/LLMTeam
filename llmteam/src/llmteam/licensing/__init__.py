"""
Licensing for llmteam.

This module provides license-based limits for:
- Concurrent pipelines
- Agents per pipeline
- Parallel agent execution
- Feature availability

Quick Start:
    from llmteam.licensing import LicenseManager, LicenseTier
    from llmteam.execution import ExecutorConfig

    # Create license manager
    license_mgr = LicenseManager(LicenseTier.PROFESSIONAL)

    # Create executor config
    config = ExecutorConfig(max_concurrent=20)

    # Enforce license limits
    config = license_mgr.enforce(config)

    # Check feature availability
    if license_mgr.check_feature("parallel_execution"):
        # Use parallel execution
        pass
"""

from llmteam.licensing.models import (
    LicenseTier,
    LicenseLimits,
    LICENSE_LIMITS,
)

from llmteam.licensing.manager import (
    LicenseManager,
)

__all__ = [
    "LicenseTier",
    "LicenseLimits",
    "LICENSE_LIMITS",
    "LicenseManager",
]
