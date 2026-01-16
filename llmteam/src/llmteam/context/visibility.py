"""
Visibility and sensitivity levels for context security.

This module defines:
- VisibilityLevel: Who can see context data
- SensitivityLevel: How sensitive the data is
"""

from enum import Enum


class VisibilityLevel(Enum):
    """
    Visibility levels for agent context.
    
    Determines who can access the context data.
    
    Note: There is NO "PEERS" level. Horizontal visibility between
    agents at the same level is FORBIDDEN by design.
    """
    
    SELF_ONLY = "self_only"
    """Only the agent itself can see this context."""
    
    ORCHESTRATOR = "orchestrator"
    """The agent and its direct orchestrator can see this context."""
    
    HIERARCHY = "hierarchy"
    """The agent and all orchestrators up the hierarchy can see this context."""


class SensitivityLevel(Enum):
    """
    Sensitivity levels for context data.
    
    Determines how the data should be protected.
    """
    
    PUBLIC = "public"
    """
    Visible to all in the hierarchy.
    Examples: status, progress, public metrics
    """
    
    INTERNAL = "internal"
    """
    Visible to orchestrators only.
    Examples: reasoning steps, intermediate results
    """
    
    CONFIDENTIAL = "confidential"
    """
    Visible only to the direct (immediate) orchestrator.
    Examples: business logic details, internal decisions
    """
    
    SECRET = "secret"
    """
    Visible ONLY to the agent itself (sealed).
    Examples: API keys, passwords, PII
    """
    
    TOP_SECRET = "top_secret"
    """
    Sealed + encrypted + audit logging.
    Examples: payment card data, SSN, medical records
    """


# Visibility rules by sensitivity
SENSITIVITY_VISIBILITY_MAP = {
    SensitivityLevel.PUBLIC: VisibilityLevel.HIERARCHY,
    SensitivityLevel.INTERNAL: VisibilityLevel.ORCHESTRATOR,
    SensitivityLevel.CONFIDENTIAL: VisibilityLevel.ORCHESTRATOR,
    SensitivityLevel.SECRET: VisibilityLevel.SELF_ONLY,
    SensitivityLevel.TOP_SECRET: VisibilityLevel.SELF_ONLY,
}


def get_visibility_for_sensitivity(sensitivity: SensitivityLevel) -> VisibilityLevel:
    """
    Get the maximum visibility level for a sensitivity level.
    
    Args:
        sensitivity: The sensitivity level
        
    Returns:
        The maximum allowed visibility level
    """
    return SENSITIVITY_VISIBILITY_MAP[sensitivity]


def is_visibility_allowed(
    requested: VisibilityLevel,
    sensitivity: SensitivityLevel,
) -> bool:
    """
    Check if a visibility level is allowed for a sensitivity level.
    
    Args:
        requested: The requested visibility level
        sensitivity: The data's sensitivity level
        
    Returns:
        True if the visibility is allowed
    """
    max_visibility = SENSITIVITY_VISIBILITY_MAP[sensitivity]
    
    # Define visibility ordering (most restrictive to least)
    order = [
        VisibilityLevel.SELF_ONLY,
        VisibilityLevel.ORCHESTRATOR,
        VisibilityLevel.HIERARCHY,
    ]
    
    requested_idx = order.index(requested)
    max_idx = order.index(max_visibility)
    
    # Requested visibility must be at least as restrictive
    return requested_idx <= max_idx
