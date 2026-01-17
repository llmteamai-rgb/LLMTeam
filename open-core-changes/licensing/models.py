# src/llmteam/licensing/models.py
"""
License tier models for LLMTeam Open Core.
"""

from enum import Enum


class LicenseTier(Enum):
    """
    License tiers for LLMTeam.
    
    COMMUNITY: Free tier with basic features
    PROFESSIONAL: Paid tier with advanced features ($99/month)
    ENTERPRISE: Custom tier with all features + support
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
