"""
Licensing models for llmteam.

Defines license tiers and their associated limits.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Set


class LicenseTier(Enum):
    """
    License tier levels.

    Attributes:
        COMMUNITY: Free tier with basic features
        PROFESSIONAL: Paid tier with advanced features
        ENTERPRISE: Enterprise tier with unlimited resources
    """

    COMMUNITY = "community"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class LicenseLimits:
    """
    Limits associated with a license tier.

    Attributes:
        max_concurrent_pipelines: Maximum number of pipelines running concurrently
        max_agents_per_pipeline: Maximum agents in a single pipeline
        max_parallel_agents: Maximum agents executing in parallel
        features: Set of available features for this tier
    """

    max_concurrent_pipelines: int
    max_agents_per_pipeline: int
    max_parallel_agents: int
    features: Set[str]


# License tier limits configuration
LICENSE_LIMITS = {
    LicenseTier.COMMUNITY: LicenseLimits(
        max_concurrent_pipelines=1,
        max_agents_per_pipeline=5,
        max_parallel_agents=2,
        features={
            "basic_agents",
            "sequential_execution",
        },
    ),
    LicenseTier.PROFESSIONAL: LicenseLimits(
        max_concurrent_pipelines=5,
        max_agents_per_pipeline=20,
        max_parallel_agents=10,
        features={
            "basic_agents",
            "sequential_execution",
            "parallel_execution",
            "process_mining",
            "external_actions",
        },
    ),
    LicenseTier.ENTERPRISE: LicenseLimits(
        max_concurrent_pipelines=999999,
        max_agents_per_pipeline=999999,
        max_parallel_agents=999999,
        features={"*"},  # All features
    ),
}
