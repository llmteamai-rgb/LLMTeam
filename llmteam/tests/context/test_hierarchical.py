"""
Tests for hierarchical context.

Tests cover:
- Context creation and hierarchy
- Parent-child relationships
- Context propagation (up and down)
- Aggregation rules
- Visibility with access control
- Integration with SecureAgentContext
"""

import pytest
from datetime import datetime

from llmteam.context import (
    ContextScope,
    HierarchicalContext,
    ContextManager,
    ContextPropagationConfig,
    SecureAgentContext,
    ContextAccessPolicy,
    SensitivityLevel,
    ROLE_AGENT,
    ROLE_PIPELINE_ORCH,
    ROLE_GROUP_ORCH,
)


class TestHierarchicalContext:
    """Tests for HierarchicalContext class."""

    def test_create_context(self):
        """Test creating a basic context."""
        ctx = HierarchicalContext(
            scope=ContextScope.AGENT,
            owner_id="agent_1",
        )

        assert ctx.scope == ContextScope.AGENT
        assert ctx.owner_id == "agent_1"
        assert ctx.parent_id is None
        assert len(ctx.children) == 0
        assert len(ctx.data) == 0

    def test_parent_child_relationship(self):
        """Test parent-child relationships."""
        parent = HierarchicalContext(
            scope=ContextScope.PIPELINE,
            owner_id="pipeline_a",
        )

        child = HierarchicalContext(
            scope=ContextScope.AGENT,
            owner_id="agent_1",
        )

        parent.set_child_context("agent_1", child)

        assert child.parent_id == "pipeline_a"
        assert len(parent.children) == 1
        assert parent.get_child_context("agent_1") == child

    def test_remove_child_context(self):
        """Test removing a child context."""
        parent = HierarchicalContext(
            scope=ContextScope.PIPELINE,
            owner_id="pipeline_a",
        )

        child = HierarchicalContext(
            scope=ContextScope.AGENT,
            owner_id="agent_1",
        )

        parent.set_child_context("agent_1", child)
        assert len(parent.children) == 1

        parent.remove_child_context("agent_1")
        assert len(parent.children) == 0
        assert parent.get_child_context("agent_1") is None

    def test_get_summary(self):
        """Test getting context summary."""
        ctx = HierarchicalContext(
            scope=ContextScope.AGENT,
            owner_id="agent_1",
        )

        summary = ctx.get_summary()

        assert summary["scope"] == "agent"
        assert summary["owner_id"] == "agent_1"
        assert summary["children_count"] == 0
        assert "updated_at" in summary
        assert summary["access_count"] == 0

    def test_to_dict(self):
        """Test converting context to dictionary."""
        ctx = HierarchicalContext(
            scope=ContextScope.AGENT,
            owner_id="agent_1",
            parent_id="pipeline_a",
        )
        ctx.data["test"] = "value"

        result = ctx.to_dict()

        assert result["scope"] == "agent"
        assert result["owner_id"] == "agent_1"
        assert result["parent_id"] == "pipeline_a"
        assert result["data"]["test"] == "value"

    def test_get_visible_children_without_security(self):
        """Test getting visible children without SecureAgentContext."""
        parent = HierarchicalContext(
            scope=ContextScope.PIPELINE,
            owner_id="pipeline_a",
        )

        child1 = HierarchicalContext(
            scope=ContextScope.AGENT,
            owner_id="agent_1",
        )

        child2 = HierarchicalContext(
            scope=ContextScope.AGENT,
            owner_id="agent_2",
        )

        parent.set_child_context("agent_1", child1)
        parent.set_child_context("agent_2", child2)

        visible = parent.get_visible_children(
            viewer_id="pipeline_a",
            viewer_role=ROLE_PIPELINE_ORCH,
        )

        assert len(visible) == 2
        assert "agent_1" in visible
        assert "agent_2" in visible
        assert visible["agent_1"]["owner_id"] == "agent_1"

    def test_get_visible_children_with_security(self):
        """Test getting visible children with SecureAgentContext."""
        parent = HierarchicalContext(
            scope=ContextScope.PIPELINE,
            owner_id="pipeline_a",
        )

        # Create child with secure context
        secure_ctx = SecureAgentContext(
            agent_id="agent_1",
            agent_name="test_agent",
            access_policy=ContextAccessPolicy(
                sensitivity=SensitivityLevel.INTERNAL,
            ),
        )
        secure_ctx.status = "running"
        secure_ctx.confidence = 0.95

        child = HierarchicalContext(
            scope=ContextScope.AGENT,
            owner_id="agent_1",
            secure_context=secure_ctx,
        )

        parent.set_child_context("agent_1", child)

        visible = parent.get_visible_children(
            viewer_id="pipeline_a",
            viewer_role=ROLE_PIPELINE_ORCH,
        )

        assert len(visible) == 1
        assert "agent_1" in visible
        assert visible["agent_1"]["status"] == "running"
        assert visible["agent_1"]["confidence"] == 0.95

    def test_get_visible_children_with_denied_access(self):
        """Test getting visible children with denied access."""
        parent = HierarchicalContext(
            scope=ContextScope.PIPELINE,
            owner_id="pipeline_a",
        )

        # Create child with access denied for viewer
        secure_ctx = SecureAgentContext(
            agent_id="agent_1",
            agent_name="test_agent",
        )
        secure_ctx.deny_access_to("pipeline_a")

        child = HierarchicalContext(
            scope=ContextScope.AGENT,
            owner_id="agent_1",
            secure_context=secure_ctx,
        )

        parent.set_child_context("agent_1", child)

        visible = parent.get_visible_children(
            viewer_id="pipeline_a",
            viewer_role=ROLE_PIPELINE_ORCH,
        )

        assert len(visible) == 1
        assert "agent_1" in visible
        assert visible["agent_1"]["access"] == "denied"


class TestContextManager:
    """Tests for ContextManager class."""

    def test_create_context(self):
        """Test creating a context via manager."""
        manager = ContextManager()

        ctx = manager.create_context(
            ContextScope.AGENT,
            "agent_1",
        )

        assert ctx.scope == ContextScope.AGENT
        assert ctx.owner_id == "agent_1"
        assert manager.get_context("agent_1") == ctx

    def test_create_hierarchy(self):
        """Test creating a complete hierarchy."""
        manager = ContextManager()

        # Create group
        group = manager.create_context(ContextScope.GROUP, "group_1")

        # Create pipeline under group
        pipeline = manager.create_context(
            ContextScope.PIPELINE,
            "pipeline_a",
            parent_id="group_1",
        )

        # Create agent under pipeline
        agent = manager.create_context(
            ContextScope.AGENT,
            "agent_1",
            parent_id="pipeline_a",
        )

        assert agent.parent_id == "pipeline_a"
        assert pipeline.parent_id == "group_1"
        assert group.parent_id is None

        assert len(pipeline.children) == 1
        assert len(group.children) == 1

        assert group.get_child_context("pipeline_a") == pipeline
        assert pipeline.get_child_context("agent_1") == agent

    def test_remove_context(self):
        """Test removing a context."""
        manager = ContextManager()

        group = manager.create_context(ContextScope.GROUP, "group_1")
        pipeline = manager.create_context(
            ContextScope.PIPELINE,
            "pipeline_a",
            parent_id="group_1",
        )
        agent = manager.create_context(
            ContextScope.AGENT,
            "agent_1",
            parent_id="pipeline_a",
        )

        # Remove pipeline (should also remove agent)
        manager.remove_context("pipeline_a")

        assert manager.get_context("pipeline_a") is None
        assert manager.get_context("agent_1") is None
        assert len(group.children) == 0

    def test_propagate_up_simple(self):
        """Test simple upward propagation."""
        manager = ContextManager()

        pipeline = manager.create_context(ContextScope.PIPELINE, "pipeline_a")
        agent = manager.create_context(
            ContextScope.AGENT,
            "agent_1",
            parent_id="pipeline_a",
        )

        # Update agent data
        agent.data["status"] = "completed"
        agent.data["confidence"] = 0.95

        # Propagate up
        manager.propagate_up("agent_1")

        # Check pipeline data
        assert pipeline.data["status"] == "completed"
        assert pipeline.data["confidence"] == 0.95

    def test_propagate_up_sum_aggregation(self):
        """Test upward propagation with sum aggregation."""
        config = ContextPropagationConfig(
            propagate_up=["error_count"],
            aggregation_rules={"error_count": "sum"},
        )
        manager = ContextManager(config)

        pipeline = manager.create_context(ContextScope.PIPELINE, "pipeline_a")
        agent1 = manager.create_context(
            ContextScope.AGENT,
            "agent_1",
            parent_id="pipeline_a",
        )
        agent2 = manager.create_context(
            ContextScope.AGENT,
            "agent_2",
            parent_id="pipeline_a",
        )

        # Update agents
        agent1.data["error_count"] = 3
        agent2.data["error_count"] = 5

        # Propagate up
        manager.propagate_up("agent_1")
        manager.propagate_up("agent_2")

        # Check sum
        assert pipeline.data["error_count"] == 8

    def test_propagate_up_avg_aggregation(self):
        """Test upward propagation with average aggregation."""
        config = ContextPropagationConfig(
            propagate_up=["confidence"],
            aggregation_rules={"confidence": "avg"},
        )
        manager = ContextManager(config)

        pipeline = manager.create_context(ContextScope.PIPELINE, "pipeline_a")
        agent1 = manager.create_context(
            ContextScope.AGENT,
            "agent_1",
            parent_id="pipeline_a",
        )
        agent2 = manager.create_context(
            ContextScope.AGENT,
            "agent_2",
            parent_id="pipeline_a",
        )

        # Update agents
        agent1.data["confidence"] = 0.8
        agent2.data["confidence"] = 0.6

        # Propagate up
        manager.propagate_up("agent_1")
        manager.propagate_up("agent_2")

        # Check average
        assert pipeline.data["confidence"] == pytest.approx(0.7)

    def test_propagate_up_worst_aggregation(self):
        """Test upward propagation with worst status aggregation."""
        config = ContextPropagationConfig(
            propagate_up=["status"],
            aggregation_rules={"status": "worst"},
        )
        manager = ContextManager(config)

        pipeline = manager.create_context(ContextScope.PIPELINE, "pipeline_a")
        agent1 = manager.create_context(
            ContextScope.AGENT,
            "agent_1",
            parent_id="pipeline_a",
        )
        agent2 = manager.create_context(
            ContextScope.AGENT,
            "agent_2",
            parent_id="pipeline_a",
        )

        # Update agents
        agent1.data["status"] = "running"
        agent2.data["status"] = "failed"

        # Propagate up
        manager.propagate_up("agent_1")
        manager.propagate_up("agent_2")

        # Check worst status
        assert pipeline.data["status"] == "failed"

    def test_propagate_down(self):
        """Test downward propagation."""
        config = ContextPropagationConfig(
            propagate_down=["global_config", "timeout"],
        )
        manager = ContextManager(config)

        pipeline = manager.create_context(ContextScope.PIPELINE, "pipeline_a")
        agent1 = manager.create_context(
            ContextScope.AGENT,
            "agent_1",
            parent_id="pipeline_a",
        )
        agent2 = manager.create_context(
            ContextScope.AGENT,
            "agent_2",
            parent_id="pipeline_a",
        )

        # Update pipeline data
        pipeline.data["global_config"] = {"mode": "production"}
        pipeline.data["timeout"] = 300

        # Propagate down
        manager.propagate_down("pipeline_a")

        # Check agents received data
        assert agent1.data["global_config"] == {"mode": "production"}
        assert agent1.data["timeout"] == 300
        assert agent2.data["global_config"] == {"mode": "production"}
        assert agent2.data["timeout"] == 300

    def test_get_root_contexts(self):
        """Test getting root contexts."""
        manager = ContextManager()

        group1 = manager.create_context(ContextScope.GROUP, "group_1")
        group2 = manager.create_context(ContextScope.GROUP, "group_2")
        pipeline = manager.create_context(
            ContextScope.PIPELINE,
            "pipeline_a",
            parent_id="group_1",
        )

        roots = manager.get_root_contexts()

        assert len(roots) == 2
        assert "group_1" in roots
        assert "group_2" in roots
        assert "pipeline_a" not in roots


class TestContextPropagationConfig:
    """Tests for ContextPropagationConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = ContextPropagationConfig()

        assert "status" in config.propagate_up
        assert "confidence" in config.propagate_up
        assert "error_count" in config.propagate_up

        assert "global_config" in config.propagate_down
        assert "shared_state" in config.propagate_down

        assert config.aggregation_rules["confidence"] == "avg"
        assert config.aggregation_rules["error_count"] == "sum"
        assert config.aggregation_rules["status"] == "worst"

    def test_add_propagate_up(self):
        """Test adding upward propagation field."""
        config = ContextPropagationConfig()

        config.add_propagate_up("custom_metric", "max")

        assert "custom_metric" in config.propagate_up
        assert config.aggregation_rules["custom_metric"] == "max"

    def test_add_propagate_down(self):
        """Test adding downward propagation field."""
        config = ContextPropagationConfig()

        config.add_propagate_down("custom_config")

        assert "custom_config" in config.propagate_down

    def test_remove_propagate_up(self):
        """Test removing upward propagation field."""
        config = ContextPropagationConfig()

        config.remove_propagate_up("confidence")

        assert "confidence" not in config.propagate_up
        assert "confidence" not in config.aggregation_rules

    def test_remove_propagate_down(self):
        """Test removing downward propagation field."""
        config = ContextPropagationConfig()

        config.remove_propagate_down("global_config")

        assert "global_config" not in config.propagate_down

    def test_set_aggregation_rule(self):
        """Test setting aggregation rule."""
        config = ContextPropagationConfig()

        config.set_aggregation_rule("status", "max")

        assert config.aggregation_rules["status"] == "max"

    def test_get_aggregation_rule(self):
        """Test getting aggregation rule."""
        config = ContextPropagationConfig()

        assert config.get_aggregation_rule("confidence") == "avg"
        assert config.get_aggregation_rule("unknown_field") == "last"
