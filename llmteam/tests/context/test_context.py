"""
Tests for context security module.
"""

import pytest

from llmteam.context import (
    SecureAgentContext,
    ContextAccessPolicy,
    SealedData,
    VisibilityLevel,
    SensitivityLevel,
    ROLE_AGENT,
    ROLE_PIPELINE_ORCH,
    ROLE_GROUP_ORCH,
    create_secure_context,
)


class TestContextAccessPolicy:
    """Tests for ContextAccessPolicy."""
    
    def test_agents_cannot_access_each_other(self):
        """Test horizontal isolation between agents."""
        policy = ContextAccessPolicy()
        
        allowed, reason = policy.can_access(
            viewer_id="agent_2",
            viewer_role=ROLE_AGENT,
        )
        
        assert allowed is False
        assert "forbidden" in reason.lower()
    
    def test_orchestrator_can_access_by_default(self):
        """Test orchestrator has default access."""
        policy = ContextAccessPolicy()
        
        allowed, _ = policy.can_access(
            viewer_id="orch_1",
            viewer_role=ROLE_PIPELINE_ORCH,
        )
        
        assert allowed is True
    
    def test_sealed_fields_blocked(self):
        """Test sealed fields are blocked."""
        policy = ContextAccessPolicy(
            sealed_fields={"secret_key"},
        )
        
        allowed, reason = policy.can_access(
            viewer_id="orch_1",
            viewer_role=ROLE_PIPELINE_ORCH,
            field_name="secret_key",
        )
        
        assert allowed is False
        assert "sealed" in reason.lower()
    
    def test_explicit_deny(self):
        """Test explicit deny list."""
        policy = ContextAccessPolicy(
            denied_viewers={"bad_orch"},
        )
        
        allowed, _ = policy.can_access(
            viewer_id="bad_orch",
            viewer_role=ROLE_PIPELINE_ORCH,
        )
        
        assert allowed is False
    
    def test_allow_overrides_deny(self):
        """Test allow list overrides deny list."""
        policy = ContextAccessPolicy(
            denied_viewers={"special_orch"},
            allowed_viewers={"special_orch"},
        )
        
        allowed, _ = policy.can_access(
            viewer_id="special_orch",
            viewer_role=ROLE_PIPELINE_ORCH,
        )
        
        assert allowed is True
    
    def test_secret_sensitivity(self):
        """Test SECRET sensitivity blocks all."""
        policy = ContextAccessPolicy(
            sensitivity=SensitivityLevel.SECRET,
        )
        
        allowed, _ = policy.can_access(
            viewer_id="orch_1",
            viewer_role=ROLE_PIPELINE_ORCH,
        )
        
        assert allowed is False
    
    def test_confidential_only_direct_orchestrator(self):
        """Test CONFIDENTIAL only allows direct orchestrator."""
        policy = ContextAccessPolicy(
            sensitivity=SensitivityLevel.CONFIDENTIAL,
        )
        
        # Pipeline orchestrator allowed
        allowed, _ = policy.can_access(
            viewer_id="pipeline_orch",
            viewer_role=ROLE_PIPELINE_ORCH,
        )
        assert allowed is True
        
        # Group orchestrator blocked
        allowed, _ = policy.can_access(
            viewer_id="group_orch",
            viewer_role=ROLE_GROUP_ORCH,
        )
        assert allowed is False


class TestSealedData:
    """Tests for SealedData."""
    
    def test_owner_can_access(self):
        """Test owner can access sealed data."""
        sealed = SealedData(
            _data="secret_value",
            owner_id="agent_1",
        )
        
        value = sealed.get("agent_1")
        
        assert value == "secret_value"
    
    def test_non_owner_blocked(self):
        """Test non-owner cannot access."""
        sealed = SealedData(
            _data="secret_value",
            owner_id="agent_1",
        )
        
        with pytest.raises(PermissionError):
            sealed.get("agent_2")
    
    def test_repr_hides_data(self):
        """Test repr doesn't expose data."""
        sealed = SealedData(
            _data="super_secret",
            owner_id="agent_1",
        )
        
        repr_str = repr(sealed)
        
        assert "super_secret" not in repr_str
        assert "REDACTED" in repr_str
    
    def test_str_hides_data(self):
        """Test str doesn't expose data."""
        sealed = SealedData(
            _data="super_secret",
            owner_id="agent_1",
        )
        
        str_val = str(sealed)
        
        assert "super_secret" not in str_val
        assert "SEALED" in str_val


class TestSecureAgentContext:
    """Tests for SecureAgentContext."""
    
    def test_basic_creation(self):
        """Test basic context creation."""
        context = SecureAgentContext(
            agent_id="agent_1",
            agent_name="test_agent",
        )
        
        assert context.agent_id == "agent_1"
        assert context.status == "idle"
        assert context.confidence == 0.0
    
    def test_set_and_get_sealed(self):
        """Test setting and getting sealed data."""
        context = SecureAgentContext(
            agent_id="agent_1",
            agent_name="test_agent",
        )
        
        context.set_sealed("api_key", "secret_123")
        
        # Owner can get
        value = context.get_sealed("api_key", "agent_1")
        assert value == "secret_123"
        
        # Non-owner cannot
        with pytest.raises(PermissionError):
            context.get_sealed("api_key", "other_agent")
    
    def test_sealed_field_added_to_policy(self):
        """Test sealing a field updates the policy."""
        context = SecureAgentContext(
            agent_id="agent_1",
            agent_name="test_agent",
        )
        
        context.set_sealed("secret", "value")
        
        assert "secret" in context.access_policy.sealed_fields
    
    def test_get_visible_context_orchestrator(self):
        """Test orchestrator gets filtered view."""
        context = SecureAgentContext(
            agent_id="agent_1",
            agent_name="test_agent",
            confidence=0.95,
            status="running",
        )
        context.set_sealed("api_key", "secret")
        
        visible = context.get_visible_context(
            viewer_id="orch_1",
            viewer_role=ROLE_PIPELINE_ORCH,
        )
        
        assert visible["agent_id"] == "agent_1"
        assert visible["confidence"] == 0.95
        assert "api_key" not in visible
        assert "sealed_fields" in visible
        assert "api_key" in visible["sealed_fields"]
    
    def test_get_visible_context_denied(self):
        """Test denied viewer gets minimal info."""
        context = SecureAgentContext(
            agent_id="agent_1",
            agent_name="test_agent",
        )
        context.deny_access_to("bad_orch")
        
        visible = context.get_visible_context(
            viewer_id="bad_orch",
            viewer_role=ROLE_PIPELINE_ORCH,
        )
        
        assert visible["access"] == "denied"
        assert "confidence" not in visible
    
    def test_get_visible_context_agent_blocked(self):
        """Test agent cannot view other agent's context."""
        context = SecureAgentContext(
            agent_id="agent_1",
            agent_name="test_agent",
        )
        
        visible = context.get_visible_context(
            viewer_id="agent_2",
            viewer_role=ROLE_AGENT,
        )
        
        assert visible["access"] == "denied"
    
    def test_update_methods(self):
        """Test update methods."""
        context = SecureAgentContext(
            agent_id="agent_1",
            agent_name="test_agent",
        )
        
        context.update_status("running")
        context.update_confidence(0.85)
        context.add_reasoning_step("Step 1")
        context.increment_error_count()
        
        assert context.status == "running"
        assert context.confidence == 0.85
        assert "Step 1" in context.reasoning_steps
        assert context.error_count == 1
    
    def test_confidence_clamped(self):
        """Test confidence is clamped to 0-1."""
        context = SecureAgentContext(
            agent_id="agent_1",
            agent_name="test_agent",
        )
        
        context.update_confidence(1.5)
        assert context.confidence == 1.0
        
        context.update_confidence(-0.5)
        assert context.confidence == 0.0
    
    def test_custom_data(self):
        """Test custom data storage."""
        context = SecureAgentContext(
            agent_id="agent_1",
            agent_name="test_agent",
        )
        
        context.set_data("custom_field", {"key": "value"})
        
        assert context.get_data("custom_field") == {"key": "value"}
        assert context.get_data("missing", "default") == "default"
    
    def test_to_dict_excludes_sealed(self):
        """Test to_dict excludes sealed by default."""
        context = SecureAgentContext(
            agent_id="agent_1",
            agent_name="test_agent",
        )
        context.set_sealed("secret", "value")
        
        data = context.to_dict(include_sealed=False)
        
        assert "_sealed" not in data
    
    def test_to_dict_includes_sealed_when_requested(self):
        """Test to_dict can include sealed data."""
        context = SecureAgentContext(
            agent_id="agent_1",
            agent_name="test_agent",
        )
        context.set_sealed("secret", "value")
        
        data = context.to_dict(include_sealed=True)
        
        assert "_sealed" in data
        assert data["_sealed"]["secret"] == "value"


class TestCreateSecureContext:
    """Tests for create_secure_context factory."""
    
    def test_basic_creation(self):
        """Test factory creates context correctly."""
        context = create_secure_context(
            agent_id="agent_1",
            agent_name="test",
        )
        
        assert context.agent_id == "agent_1"
        assert context.access_policy.sensitivity == SensitivityLevel.INTERNAL
    
    def test_with_sensitivity(self):
        """Test factory with custom sensitivity."""
        context = create_secure_context(
            agent_id="agent_1",
            agent_name="test",
            sensitivity=SensitivityLevel.CONFIDENTIAL,
        )
        
        assert context.access_policy.sensitivity == SensitivityLevel.CONFIDENTIAL
    
    def test_with_sealed_fields(self):
        """Test factory with pre-sealed fields."""
        context = create_secure_context(
            agent_id="agent_1",
            agent_name="test",
            sealed_fields={"field1", "field2"},
        )
        
        assert "field1" in context.access_policy.sealed_fields
        assert "field2" in context.access_policy.sealed_fields
