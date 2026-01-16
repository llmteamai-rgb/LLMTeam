"""
Context security components.

This module provides:
- ContextAccessPolicy: Access control rules for agent context
- SealedData: Container for sealed (owner-only) data
- ContextAccessResult: Result of access check
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Set, Tuple

from llmteam.context.visibility import (
    VisibilityLevel,
    SensitivityLevel,
    get_visibility_for_sensitivity,
)


# Role types for access checking
ROLE_AGENT = "agent"
ROLE_PIPELINE_ORCH = "pipeline_orch"
ROLE_GROUP_ORCH = "group_orch"
ROLE_SYSTEM = "system"


@dataclass
class ContextAccessResult:
    """
    Result of an access check.
    
    Attributes:
        allowed: Whether access is allowed
        reason: Human-readable reason
        fields_granted: Set of fields that can be accessed
        fields_denied: Set of fields that cannot be accessed
    """
    allowed: bool
    reason: str
    fields_granted: Set[str] = field(default_factory=set)
    fields_denied: Set[str] = field(default_factory=set)


@dataclass
class ContextAccessPolicy:
    """
    Access control policy for agent context.
    
    Defines who can see what data in an agent's context.
    
    Key principles:
    1. Agents NEVER see each other's contexts (horizontal isolation)
    2. Orchestrators can see their agents' contexts (unless denied)
    3. Sealed fields are visible ONLY to the owning agent
    4. Explicit denials override defaults
    
    Attributes:
        default_visibility: Base visibility level for the context
        denied_viewers: Set of viewer IDs explicitly denied access
        allowed_viewers: Set of viewer IDs allowed access (override deny)
        sealed_fields: Fields that only the owner can access
        sensitivity: Overall sensitivity level
        audit_access: Whether to log access attempts
        encrypt_sealed: Whether to encrypt sealed data
    
    Example:
        policy = ContextAccessPolicy(
            sensitivity=SensitivityLevel.CONFIDENTIAL,
            sealed_fields={"api_key", "password"},
            audit_access=True,
        )
        
        allowed, reason = policy.can_access(
            viewer_id="pipeline_orch_1",
            viewer_role="pipeline_orch",
            field_name="api_key",
        )
        # Returns (False, "Field 'api_key' is sealed")
    """
    
    default_visibility: VisibilityLevel = VisibilityLevel.ORCHESTRATOR
    denied_viewers: Set[str] = field(default_factory=set)
    allowed_viewers: Set[str] = field(default_factory=set)
    sealed_fields: Set[str] = field(default_factory=set)
    sensitivity: SensitivityLevel = SensitivityLevel.INTERNAL
    audit_access: bool = False
    encrypt_sealed: bool = False
    
    def can_access(
        self,
        viewer_id: str,
        viewer_role: str,
        field_name: str = None,
    ) -> Tuple[bool, str]:
        """
        Check if a viewer can access the context or a specific field.
        
        Args:
            viewer_id: Identifier of the entity requesting access
            viewer_role: Role of the viewer (agent, pipeline_orch, group_orch, system)
            field_name: Optional specific field to check
            
        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        # Rule 1: Agents NEVER see each other's contexts
        if viewer_role == ROLE_AGENT:
            return False, "Horizontal access between agents is forbidden"
        
        # Rule 2: Sealed fields are only for the owner
        if field_name and field_name in self.sealed_fields:
            return False, f"Field '{field_name}' is sealed"
        
        # Rule 3: Check explicit deny/allow lists
        if viewer_id in self.denied_viewers:
            if viewer_id not in self.allowed_viewers:
                return False, f"Viewer '{viewer_id}' is explicitly denied"
        
        # Rule 4: Check sensitivity level
        if self.sensitivity in (SensitivityLevel.SECRET, SensitivityLevel.TOP_SECRET):
            return False, "Context is sealed (SECRET level)"
        
        # Rule 5: CONFIDENTIAL only visible to direct orchestrator
        if self.sensitivity == SensitivityLevel.CONFIDENTIAL:
            if viewer_role not in (ROLE_PIPELINE_ORCH, ROLE_SYSTEM):
                return False, "CONFIDENTIAL: only direct orchestrator allowed"
        
        return True, "Access granted"
    
    def check_access(
        self,
        viewer_id: str,
        viewer_role: str,
        requested_fields: Set[str] = None,
    ) -> ContextAccessResult:
        """
        Check access and return detailed result.
        
        Args:
            viewer_id: Identifier of the entity requesting access
            viewer_role: Role of the viewer
            requested_fields: Optional set of fields to check
            
        Returns:
            ContextAccessResult with detailed information
        """
        # First check base access
        allowed, reason = self.can_access(viewer_id, viewer_role)
        
        if not allowed:
            return ContextAccessResult(
                allowed=False,
                reason=reason,
                fields_denied=requested_fields or set(),
            )
        
        # If no specific fields requested, access is granted
        if not requested_fields:
            return ContextAccessResult(allowed=True, reason=reason)
        
        # Check each field
        fields_granted = set()
        fields_denied = set()
        
        for field in requested_fields:
            field_allowed, _ = self.can_access(viewer_id, viewer_role, field)
            if field_allowed:
                fields_granted.add(field)
            else:
                fields_denied.add(field)
        
        return ContextAccessResult(
            allowed=len(fields_denied) == 0,
            reason=reason if not fields_denied else f"Denied fields: {fields_denied}",
            fields_granted=fields_granted,
            fields_denied=fields_denied,
        )
    
    def deny_access_to(self, viewer_id: str) -> None:
        """Add a viewer to the deny list."""
        self.denied_viewers.add(viewer_id)
    
    def allow_access_to(self, viewer_id: str) -> None:
        """Add a viewer to the allow list (overrides deny)."""
        self.allowed_viewers.add(viewer_id)
    
    def seal_field(self, field_name: str) -> None:
        """Mark a field as sealed (owner-only access)."""
        self.sealed_fields.add(field_name)
    
    def unseal_field(self, field_name: str) -> None:
        """Remove sealed status from a field."""
        self.sealed_fields.discard(field_name)
    
    def set_sensitivity(self, level: SensitivityLevel) -> None:
        """Set the sensitivity level."""
        self.sensitivity = level


@dataclass
class SealedData:
    """
    Container for sealed (owner-only) data.
    
    Data stored in SealedData can only be accessed by the owner.
    Even orchestrators cannot read sealed data.
    
    Attributes:
        _data: The actual data (hidden from repr/str)
        owner_id: ID of the agent that owns this data
        encryption_key_id: Key ID for encryption (TOP_SECRET)
        created_at: When the data was sealed
        
    Example:
        sealed = SealedData(
            _data="secret_api_key",
            owner_id="agent_123",
        )
        
        # Owner can access
        value = sealed.get("agent_123")  # Returns "secret_api_key"
        
        # Others cannot
        value = sealed.get("agent_456")  # Raises PermissionError
    """
    
    _data: Any = field(repr=False)  # Don't show in repr
    owner_id: str = ""
    encryption_key_id: str = ""  # For TOP_SECRET
    created_at: datetime = field(default_factory=datetime.now)
    
    def get(self, requester_id: str) -> Any:
        """
        Get the sealed data.
        
        Args:
            requester_id: ID of the entity requesting the data
            
        Returns:
            The sealed data if requester is owner
            
        Raises:
            PermissionError: If requester is not the owner
        """
        if requester_id != self.owner_id:
            raise PermissionError(
                f"Access denied: '{requester_id}' cannot access "
                f"sealed data owned by '{self.owner_id}'"
            )
        return self._data
    
    def is_owner(self, requester_id: str) -> bool:
        """Check if requester is the owner."""
        return requester_id == self.owner_id
    
    def __repr__(self) -> str:
        """Safe repr that doesn't expose data."""
        return f"SealedData(owner_id='{self.owner_id}', [REDACTED])"
    
    def __str__(self) -> str:
        """Safe str that doesn't expose data."""
        return "[SEALED]"


class ContextAccessError(Exception):
    """Raised when context access is denied."""
    
    def __init__(self, message: str, viewer_id: str = "", field_name: str = ""):
        self.viewer_id = viewer_id
        self.field_name = field_name
        super().__init__(message)


class SealedDataAccessError(ContextAccessError):
    """Raised when trying to access sealed data without permission."""
    
    def __init__(self, owner_id: str, requester_id: str, field_name: str = ""):
        self.owner_id = owner_id
        self.requester_id = requester_id
        super().__init__(
            f"Sealed data access denied: '{requester_id}' cannot access "
            f"data owned by '{owner_id}'",
            viewer_id=requester_id,
            field_name=field_name,
        )
