"""
Secure agent context implementation.

This module provides SecureAgentContext - an agent context with
built-in access control and sealed data support.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from llmteam.context.visibility import VisibilityLevel, SensitivityLevel
from llmteam.context.security import (
    ContextAccessPolicy,
    SealedData,
    ContextAccessResult,
    ContextAccessError,
    ROLE_AGENT,
    ROLE_PIPELINE_ORCH,
    ROLE_GROUP_ORCH,
    ROLE_SYSTEM,
)


@dataclass
class SecureAgentContext:
    """
    Agent context with security controls.
    
    Provides:
    - Access control via ContextAccessPolicy
    - Sealed data that only the agent can access
    - Filtered context views for orchestrators
    - Optional audit logging for access
    
    Attributes:
        agent_id: Unique identifier for the agent
        agent_name: Human-readable name
        confidence: Current confidence level (0-1)
        status: Current status (idle, running, completed, failed)
        error_count: Number of errors encountered
        last_action: Description of last action taken
        reasoning_steps: List of reasoning steps (internal)
        sources: List of sources used (internal)
        messages: Message history
        access_policy: Access control policy
        _audit_trail: Optional audit trail for logging access
        
    Example:
        context = SecureAgentContext(
            agent_id="agent_123",
            agent_name="analyzer",
            access_policy=ContextAccessPolicy(
                sensitivity=SensitivityLevel.CONFIDENTIAL,
                sealed_fields={"api_key"},
                audit_access=True,
            ),
        )
        
        # Store sealed data
        context.set_sealed("api_key", "secret_value")
        
        # Get data (only owner can)
        value = context.get_sealed("api_key", requester_id="agent_123")
        
        # Orchestrator gets filtered view
        visible = context.get_visible_context(
            viewer_id="orch_1",
            viewer_role="pipeline_orch",
        )
        # visible["sealed_fields"] = ["api_key"]
        # But actual value is not included
    """
    
    # Identity
    agent_id: str
    agent_name: str
    
    # Public data (visible to hierarchy)
    confidence: float = 0.0
    status: str = "idle"
    error_count: int = 0
    last_action: str = ""
    
    # Internal data (visible to orchestrators if allowed)
    reasoning_steps: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    
    # Custom data storage
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Sealed data (only owner can access)
    _sealed: Dict[str, SealedData] = field(default_factory=dict, repr=False)
    
    # Security
    access_policy: ContextAccessPolicy = field(default_factory=ContextAccessPolicy)
    
    # Audit (optional)
    _audit_trail: Optional[Any] = field(default=None, repr=False)  # AuditTrail
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def set_sealed(
        self,
        key: str,
        value: Any,
        encrypt: bool = False,
    ) -> None:
        """
        Store data as sealed (only this agent can access).
        
        Args:
            key: Key for the sealed data
            value: Value to seal
            encrypt: Whether to mark for encryption (TOP_SECRET)
        """
        self._sealed[key] = SealedData(
            _data=value,
            owner_id=self.agent_id,
            encryption_key_id=f"{self.agent_id}:{key}" if encrypt else "",
        )
        self.access_policy.seal_field(key)
        self.updated_at = datetime.now()
    
    def get_sealed(self, key: str, requester_id: str) -> Any:
        """
        Get sealed data (only owner can access).
        
        Args:
            key: Key of the sealed data
            requester_id: ID of entity requesting the data
            
        Returns:
            The sealed value if requester is owner
            
        Raises:
            KeyError: If key doesn't exist
            PermissionError: If requester is not owner
        """
        if key not in self._sealed:
            raise KeyError(f"No sealed data with key '{key}'")
        
        value = self._sealed[key].get(requester_id)
        
        # Audit access
        if self._audit_trail and self.access_policy.audit_access:
            self._log_sealed_access(requester_id, key, success=True)
        
        return value
    
    def has_sealed(self, key: str) -> bool:
        """Check if a sealed key exists."""
        return key in self._sealed
    
    def list_sealed_keys(self) -> List[str]:
        """List all sealed keys (not values)."""
        return list(self._sealed.keys())
    
    def delete_sealed(self, key: str, requester_id: str) -> None:
        """
        Delete sealed data (only owner can).
        
        Args:
            key: Key to delete
            requester_id: ID of entity requesting deletion
            
        Raises:
            PermissionError: If requester is not owner
        """
        if key not in self._sealed:
            return
        
        if self._sealed[key].owner_id != requester_id:
            raise PermissionError(
                f"Cannot delete sealed data: '{requester_id}' is not owner"
            )
        
        del self._sealed[key]
        self.access_policy.sealed_fields.discard(key)
        self.updated_at = datetime.now()
    
    def get_visible_context(
        self,
        viewer_id: str,
        viewer_role: str,
    ) -> Dict[str, Any]:
        """
        Get context filtered by viewer's access rights.
        
        Args:
            viewer_id: ID of entity viewing the context
            viewer_role: Role of viewer (agent, pipeline_orch, etc.)
            
        Returns:
            Dictionary with visible context data
        """
        # Check base access
        allowed, reason = self.access_policy.can_access(viewer_id, viewer_role)
        
        if not allowed:
            # Log denied access
            if self._audit_trail and self.access_policy.audit_access:
                self._log_context_access(viewer_id, viewer_role, allowed=False, reason=reason)
            
            return {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "access": "denied",
                "reason": reason,
            }
        
        # Build visible context
        result = {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": self.status,
            "confidence": self.confidence,
            "error_count": self.error_count,
            "last_action": self.last_action,
            "updated_at": self.updated_at.isoformat(),
        }
        
        # Include internal data based on sensitivity
        sensitivity = self.access_policy.sensitivity
        
        if sensitivity in (SensitivityLevel.PUBLIC, SensitivityLevel.INTERNAL):
            result["reasoning_steps"] = self.reasoning_steps.copy()
            result["sources"] = self.sources.copy()
        
        # Include custom data (excluding sealed keys)
        visible_data = {
            k: v for k, v in self.data.items()
            if k not in self.access_policy.sealed_fields
        }
        if visible_data:
            result["data"] = visible_data
        
        # List sealed fields (names only, not values)
        result["sealed_fields"] = list(self.access_policy.sealed_fields)
        
        # Log access
        if self._audit_trail and self.access_policy.audit_access:
            self._log_context_access(
                viewer_id, viewer_role,
                allowed=True,
                fields_visible=list(result.keys()),
            )
        
        return result
    
    def check_access(
        self,
        viewer_id: str,
        viewer_role: str,
        fields: Set[str] = None,
    ) -> ContextAccessResult:
        """
        Check access rights without retrieving data.
        
        Args:
            viewer_id: ID of entity checking access
            viewer_role: Role of viewer
            fields: Optional specific fields to check
            
        Returns:
            ContextAccessResult with detailed access information
        """
        return self.access_policy.check_access(viewer_id, viewer_role, fields)
    
    # Access policy shortcuts
    
    def deny_access_to(self, viewer_id: str) -> None:
        """Deny access to a specific viewer."""
        self.access_policy.deny_access_to(viewer_id)
    
    def allow_access_to(self, viewer_id: str) -> None:
        """Allow access to a specific viewer (override deny)."""
        self.access_policy.allow_access_to(viewer_id)
    
    def set_sensitivity(self, level: SensitivityLevel) -> None:
        """Set the sensitivity level for this context."""
        self.access_policy.set_sensitivity(level)
    
    # Update methods
    
    def update_status(self, status: str) -> None:
        """Update agent status."""
        self.status = status
        self.updated_at = datetime.now()
    
    def update_confidence(self, confidence: float) -> None:
        """Update confidence level."""
        self.confidence = max(0.0, min(1.0, confidence))
        self.updated_at = datetime.now()
    
    def add_reasoning_step(self, step: str) -> None:
        """Add a reasoning step."""
        self.reasoning_steps.append(step)
        self.updated_at = datetime.now()
    
    def add_source(self, source: str) -> None:
        """Add a source reference."""
        self.sources.append(source)
        self.updated_at = datetime.now()
    
    def add_message(self, message: Dict[str, Any]) -> None:
        """Add a message to history."""
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def increment_error_count(self) -> None:
        """Increment error counter."""
        self.error_count += 1
        self.updated_at = datetime.now()
    
    def set_data(self, key: str, value: Any) -> None:
        """Set custom data (not sealed)."""
        self.data[key] = value
        self.updated_at = datetime.now()
    
    def get_data(self, key: str, default: Any = None) -> Any:
        """Get custom data."""
        return self.data.get(key, default)
    
    # Serialization
    
    def to_dict(self, include_sealed: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Args:
            include_sealed: Whether to include sealed data
                           (only use for persistence by owner)
        """
        result = {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "confidence": self.confidence,
            "status": self.status,
            "error_count": self.error_count,
            "last_action": self.last_action,
            "reasoning_steps": self.reasoning_steps,
            "sources": self.sources,
            "messages": self.messages,
            "data": self.data,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "access_policy": {
                "default_visibility": self.access_policy.default_visibility.value,
                "sensitivity": self.access_policy.sensitivity.value,
                "sealed_fields": list(self.access_policy.sealed_fields),
                "audit_access": self.access_policy.audit_access,
            },
        }
        
        if include_sealed:
            result["_sealed"] = {
                k: v._data for k, v in self._sealed.items()
            }
        
        return result
    
    # Private methods
    
    def _log_context_access(
        self,
        viewer_id: str,
        viewer_role: str,
        allowed: bool,
        reason: str = "",
        fields_visible: List[str] = None,
    ) -> None:
        """Log context access to audit trail."""
        if not self._audit_trail:
            return
        
        from llmteam.audit import AuditEventType
        
        event_type = (
            AuditEventType.CONTEXT_ACCESSED if allowed
            else AuditEventType.ACCESS_DENIED
        )
        
        # Create task to not block
        asyncio.create_task(self._audit_trail.log(
            event_type,
            actor_id=viewer_id,
            actor_type=viewer_role,
            resource_type="agent_context",
            resource_id=self.agent_id,
            success=allowed,
            error_message=reason if not allowed else "",
            metadata={
                "fields_visible": fields_visible or [],
            },
        ))
    
    def _log_sealed_access(
        self,
        requester_id: str,
        key: str,
        success: bool,
    ) -> None:
        """Log sealed data access to audit trail."""
        if not self._audit_trail:
            return
        
        from llmteam.audit import AuditEventType
        
        event_type = (
            AuditEventType.SEALED_DATA_ACCESSED if success
            else AuditEventType.ACCESS_DENIED
        )
        
        asyncio.create_task(self._audit_trail.log(
            event_type,
            actor_id=requester_id,
            resource_type="sealed_data",
            resource_id=f"{self.agent_id}:{key}",
            success=success,
        ))


def create_secure_context(
    agent_id: str,
    agent_name: str,
    sensitivity: SensitivityLevel = SensitivityLevel.INTERNAL,
    sealed_fields: Set[str] = None,
    audit_access: bool = False,
    audit_trail: Any = None,
) -> SecureAgentContext:
    """
    Factory function to create a SecureAgentContext.
    
    Args:
        agent_id: Unique identifier for the agent
        agent_name: Human-readable name
        sensitivity: Sensitivity level for the context
        sealed_fields: Fields that should be sealed
        audit_access: Whether to log access attempts
        audit_trail: AuditTrail instance for logging
        
    Returns:
        Configured SecureAgentContext
    """
    policy = ContextAccessPolicy(
        sensitivity=sensitivity,
        sealed_fields=sealed_fields or set(),
        audit_access=audit_access,
    )
    
    return SecureAgentContext(
        agent_id=agent_id,
        agent_name=agent_name,
        access_policy=policy,
        _audit_trail=audit_trail,
    )
