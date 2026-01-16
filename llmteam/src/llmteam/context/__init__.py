"""
Context security for llmteam.

This module provides secure agent context with:
- Access control policies
- Sealed data (owner-only access)
- Visibility and sensitivity levels
- Filtered context views for orchestrators

Key Principles:
1. Agents NEVER see each other's contexts (horizontal isolation)
2. Orchestrators can see their agents' contexts (vertical visibility)
3. Sealed data is ONLY accessible by the owning agent
4. Explicit denials override default permissions

Quick Start:
    from llmteam.context import (
        SecureAgentContext,
        ContextAccessPolicy,
        SensitivityLevel,
    )
    
    # Create secure context
    context = SecureAgentContext(
        agent_id="agent_123",
        agent_name="payment_processor",
        access_policy=ContextAccessPolicy(
            sensitivity=SensitivityLevel.CONFIDENTIAL,
            sealed_fields={"card_number", "cvv"},
            audit_access=True,
        ),
    )
    
    # Store sealed data
    context.set_sealed("card_number", "4111-1111-1111-1111")
    
    # Only the agent can retrieve it
    card = context.get_sealed("card_number", requester_id="agent_123")
    
    # Orchestrators get filtered view
    visible = context.get_visible_context(
        viewer_id="pipeline_orch_1",
        viewer_role="pipeline_orch",
    )
    # sealed_fields listed but values not included
"""

from llmteam.context.visibility import (
    VisibilityLevel,
    SensitivityLevel,
    get_visibility_for_sensitivity,
    is_visibility_allowed,
    SENSITIVITY_VISIBILITY_MAP,
)

from llmteam.context.security import (
    ContextAccessPolicy,
    ContextAccessResult,
    SealedData,
    ContextAccessError,
    SealedDataAccessError,
    ROLE_AGENT,
    ROLE_PIPELINE_ORCH,
    ROLE_GROUP_ORCH,
    ROLE_SYSTEM,
)

from llmteam.context.secure_context import (
    SecureAgentContext,
    create_secure_context,
)

from llmteam.context.hierarchical import (
    ContextScope,
    HierarchicalContext,
    ContextManager,
)

from llmteam.context.propagation import (
    ContextPropagationConfig,
)

__all__ = [
    # Visibility
    "VisibilityLevel",
    "SensitivityLevel",
    "get_visibility_for_sensitivity",
    "is_visibility_allowed",
    "SENSITIVITY_VISIBILITY_MAP",

    # Security
    "ContextAccessPolicy",
    "ContextAccessResult",
    "SealedData",
    "ContextAccessError",
    "SealedDataAccessError",

    # Roles
    "ROLE_AGENT",
    "ROLE_PIPELINE_ORCH",
    "ROLE_GROUP_ORCH",
    "ROLE_SYSTEM",

    # Context
    "SecureAgentContext",
    "create_secure_context",

    # Hierarchical Context (v1.8.0)
    "ContextScope",
    "HierarchicalContext",
    "ContextManager",
    "ContextPropagationConfig",
]
