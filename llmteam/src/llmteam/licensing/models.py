"""
License tier models for LLMTeam Open Core.

Defines license tiers and their associated limits.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Set, List, Dict, Any


class LicenseTier(Enum):
    """
    License tiers for LLMTeam.

    COMMUNITY: Free tier with basic features
    PROFESSIONAL: Paid tier with advanced features ($99/month)
    ENTERPRISE: Enterprise tier with all features + support
    """

    COMMUNITY = "community"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"

    def __str__(self) -> str:
        return self.value

    @property
    def display_name(self) -> str:
        return self.value.title()

    @property
    def is_paid(self) -> bool:
        return self in (LicenseTier.PROFESSIONAL, LicenseTier.ENTERPRISE)


@dataclass
class LicenseLimits:
    """
    Limits associated with a license tier.

    Attributes:
        max_concurrent_pipelines: Maximum pipelines running concurrently
        max_agents_per_pipeline: Maximum agents in a single pipeline
        max_parallel_agents: Maximum agents executing in parallel
        max_teams: Maximum number of teams
        max_snapshots: Maximum stored snapshots
        features: Set of available feature names
    """

    # Concurrency limits
    max_concurrent_pipelines: int = 1
    max_agents_per_pipeline: int = 5
    max_parallel_agents: int = 2
    max_teams: int = 1
    max_snapshots: int = 10

    # Feature flags
    process_mining: bool = False
    audit_trail: bool = False
    multi_tenant: bool = False
    postgres_store: bool = False
    redis_store: bool = False
    human_interaction: bool = False
    external_actions: bool = False

    # Legacy: Set of feature names (for backwards compatibility)
    features: Set[str] = field(default_factory=set)


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


# License tier limits configuration
LICENSE_LIMITS: Dict[LicenseTier, LicenseLimits] = {
    LicenseTier.COMMUNITY: LicenseLimits(
        max_concurrent_pipelines=1,
        max_agents_per_pipeline=5,
        max_parallel_agents=2,
        max_teams=2,
        max_snapshots=10,
        process_mining=False,
        audit_trail=False,
        multi_tenant=False,
        postgres_store=False,
        redis_store=False,
        human_interaction=False,
        external_actions=False,
        features={
            "basic",
            "memory_store",
            "basic_agents",
            "sequential_execution",
        },
    ),
    LicenseTier.PROFESSIONAL: LicenseLimits(
        max_concurrent_pipelines=5,
        max_agents_per_pipeline=20,
        max_parallel_agents=10,
        max_teams=10,
        max_snapshots=100,
        process_mining=True,
        audit_trail=False,
        multi_tenant=False,
        postgres_store=True,
        redis_store=True,
        human_interaction=True,
        external_actions=True,
        features={
            "basic",
            "memory_store",
            "basic_agents",
            "sequential_execution",
            "parallel_execution",
            "process_mining",
            "postgres_store",
            "redis_store",
            "human_interaction",
            "external_actions",
        },
    ),
    LicenseTier.ENTERPRISE: LicenseLimits(
        max_concurrent_pipelines=999999,
        max_agents_per_pipeline=999999,
        max_parallel_agents=999999,
        max_teams=999999,
        max_snapshots=999999,
        process_mining=True,
        audit_trail=True,
        multi_tenant=True,
        postgres_store=True,
        redis_store=True,
        human_interaction=True,
        external_actions=True,
        features={"*"},  # All features
    ),
}

# Alias for backwards compatibility
TIER_LIMITS = LICENSE_LIMITS
