"""
Hierarchical context for llmteam.

This module provides hierarchical context management for multi-level
agent orchestration:
- Agent-level context (individual agents)
- Pipeline-level context (orchestrator managing agents)
- Group-level context (group orchestrator managing pipelines)

Key Features:
1. Parent-child relationships (agent → pipeline → group)
2. Vertical visibility (parent can see children, children cannot see siblings)
3. Context propagation up and down the hierarchy
4. Integration with SecureAgentContext for access control

Quick Start:
    from llmteam.context import ContextManager, ContextScope

    # Create context manager
    ctx_manager = ContextManager()

    # Create hierarchy
    group_ctx = ctx_manager.create_context(ContextScope.GROUP, "group_1")
    pipeline_ctx = ctx_manager.create_context(
        ContextScope.PIPELINE, "pipeline_a", parent_id="group_1"
    )
    agent_ctx = ctx_manager.create_context(
        ContextScope.AGENT, "agent_1", parent_id="pipeline_a"
    )

    # Agent updates its context
    agent_ctx.data["confidence"] = 0.95
    agent_ctx.data["status"] = "completed"

    # Propagate data up the hierarchy
    ctx_manager.propagate_up("agent_1")

    # Pipeline sees aggregated data
    print(pipeline_ctx.data)  # {"confidence": 0.95, "status": "completed"}
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from llmteam.context.propagation import ContextPropagationConfig


class ContextScope(Enum):
    """Scope of context in the hierarchy."""

    AGENT = "agent"
    PIPELINE = "pipeline"
    GROUP = "group"


@dataclass
class HierarchicalContext:
    """
    Context with hierarchical parent-child relationships.

    Integrates with SecureAgentContext from v1.7.0 for access control.

    Attributes:
        scope: Level in the hierarchy (AGENT, PIPELINE, GROUP)
        owner_id: Unique identifier for the owner
        parent_id: ID of the parent context (None for root)
        data: Context data storage
        children: Child contexts (for pipeline/group)
        secure_context: Optional SecureAgentContext for security
        created_at: When the context was created
        updated_at: When the context was last updated
        access_count: Number of times the context was accessed

    Example:
        # Create agent context
        agent_ctx = HierarchicalContext(
            scope=ContextScope.AGENT,
            owner_id="agent_1",
            parent_id="pipeline_a",
        )

        # Create pipeline context
        pipeline_ctx = HierarchicalContext(
            scope=ContextScope.PIPELINE,
            owner_id="pipeline_a",
        )

        # Link agent to pipeline
        pipeline_ctx.set_child_context("agent_1", agent_ctx)

        # Get visible children (with access control)
        visible = pipeline_ctx.get_visible_children(
            viewer_id="pipeline_a",
            viewer_role="pipeline_orch",
        )
    """

    scope: ContextScope
    owner_id: str
    parent_id: Optional[str] = None

    # Data storage
    data: Dict[str, Any] = field(default_factory=dict)

    # Child contexts (for pipeline/group scopes)
    children: Dict[str, "HierarchicalContext"] = field(default_factory=dict)

    # Optional integration with SecureAgentContext
    secure_context: Optional[Any] = field(default=None, repr=False)  # SecureAgentContext

    # Metadata for Process Mining
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    def get_child_context(self, child_id: str) -> Optional["HierarchicalContext"]:
        """
        Get child context by ID.

        Args:
            child_id: ID of the child context

        Returns:
            Child context if found, None otherwise
        """
        return self.children.get(child_id)

    def set_child_context(
        self,
        child_id: str,
        context: "HierarchicalContext",
    ) -> None:
        """
        Set child context and establish parent-child relationship.

        Args:
            child_id: ID of the child context
            context: Child context instance
        """
        context.parent_id = self.owner_id
        self.children[child_id] = context
        self.updated_at = datetime.now()

    def remove_child_context(self, child_id: str) -> None:
        """
        Remove child context.

        Args:
            child_id: ID of the child to remove
        """
        if child_id in self.children:
            del self.children[child_id]
            self.updated_at = datetime.now()

    def get_visible_children(
        self,
        viewer_id: str,
        viewer_role: str,
    ) -> Dict[str, dict]:
        """
        Get visible child contexts with access control.

        Uses ContextAccessPolicy from v1.7.0 if available.

        Args:
            viewer_id: ID of the viewer
            viewer_role: Role of the viewer

        Returns:
            Dictionary mapping child_id to visible context data
        """
        result = {}

        for child_id, child_ctx in self.children.items():
            # Check access using SecureAgentContext if available
            if child_ctx.secure_context:
                allowed, reason = child_ctx.secure_context.access_policy.can_access(
                    viewer_id, viewer_role
                )
                if not allowed:
                    result[child_id] = {
                        "access": "denied",
                        "reason": reason,
                    }
                    continue

                # Get filtered view
                result[child_id] = child_ctx.secure_context.get_visible_context(
                    viewer_id, viewer_role
                )
            else:
                # No security context, return summary
                result[child_id] = child_ctx.get_summary()

        self.access_count += 1
        return result

    def get_summary(self) -> dict:
        """
        Get summary of this context (without detailed data).

        Returns:
            Dictionary with context summary
        """
        return {
            "scope": self.scope.value,
            "owner_id": self.owner_id,
            "children_count": len(self.children),
            "updated_at": self.updated_at.isoformat(),
            "access_count": self.access_count,
        }

    def to_dict(self) -> dict:
        """
        Convert context to dictionary.

        Returns:
            Dictionary representation of the context
        """
        return {
            "scope": self.scope.value,
            "owner_id": self.owner_id,
            "parent_id": self.parent_id,
            "data": self.data.copy(),
            "children_count": len(self.children),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "access_count": self.access_count,
        }


class ContextManager:
    """
    Manager for hierarchical contexts.

    Handles:
    - Context creation and lifecycle
    - Parent-child relationships
    - Context propagation (up/down the hierarchy)
    - Aggregation of child data to parent

    Example:
        manager = ContextManager()

        # Create hierarchy
        group = manager.create_context(ContextScope.GROUP, "group_1")
        pipeline = manager.create_context(
            ContextScope.PIPELINE, "pipeline_a", parent_id="group_1"
        )
        agent = manager.create_context(
            ContextScope.AGENT, "agent_1", parent_id="pipeline_a"
        )

        # Update agent data
        agent.data["confidence"] = 0.95
        agent.data["status"] = "completed"

        # Propagate up
        manager.propagate_up("agent_1")

        # Check pipeline data
        assert pipeline.data["confidence"] == 0.95
    """

    def __init__(self, propagation_config: Optional[ContextPropagationConfig] = None):
        """
        Initialize context manager.

        Args:
            propagation_config: Configuration for context propagation
        """
        self.config = propagation_config or ContextPropagationConfig()
        self._contexts: Dict[str, HierarchicalContext] = {}

    def create_context(
        self,
        scope: ContextScope,
        owner_id: str,
        parent_id: Optional[str] = None,
        secure_context: Optional[Any] = None,
    ) -> HierarchicalContext:
        """
        Create a new hierarchical context.

        Args:
            scope: Level in the hierarchy
            owner_id: Unique identifier for the owner
            parent_id: ID of the parent context (None for root)
            secure_context: Optional SecureAgentContext for access control

        Returns:
            Newly created HierarchicalContext
        """
        ctx = HierarchicalContext(
            scope=scope,
            owner_id=owner_id,
            parent_id=parent_id,
            secure_context=secure_context,
        )

        self._contexts[owner_id] = ctx

        # Link to parent if specified
        if parent_id and parent_id in self._contexts:
            self._contexts[parent_id].set_child_context(owner_id, ctx)

        return ctx

    def get_context(self, owner_id: str) -> Optional[HierarchicalContext]:
        """
        Get context by owner ID.

        Args:
            owner_id: ID of the context owner

        Returns:
            Context if found, None otherwise
        """
        return self._contexts.get(owner_id)

    def remove_context(self, owner_id: str) -> None:
        """
        Remove context and its children.

        Args:
            owner_id: ID of the context to remove
        """
        ctx = self._contexts.get(owner_id)
        if not ctx:
            return

        # Remove from parent
        if ctx.parent_id and ctx.parent_id in self._contexts:
            self._contexts[ctx.parent_id].remove_child_context(owner_id)

        # Remove children
        for child_id in list(ctx.children.keys()):
            self.remove_context(child_id)

        # Remove self
        del self._contexts[owner_id]

    def propagate_up(self, child_id: str) -> None:
        """
        Propagate data from child to parent.

        Applies aggregation rules defined in ContextPropagationConfig.

        Args:
            child_id: ID of the child context
        """
        child = self._contexts.get(child_id)
        if not child or not child.parent_id:
            return

        parent = self._contexts.get(child.parent_id)
        if not parent:
            return

        # Propagate configured fields
        for field in self.config.propagate_up:
            if field in child.data:
                self._aggregate_to_parent(parent, field, child.data[field])

        parent.updated_at = datetime.now()

    def propagate_down(self, parent_id: str) -> None:
        """
        Propagate data from parent to children.

        Args:
            parent_id: ID of the parent context
        """
        parent = self._contexts.get(parent_id)
        if not parent:
            return

        # Propagate configured fields to all children
        for child in parent.children.values():
            for field in self.config.propagate_down:
                if field in parent.data:
                    child.data[field] = parent.data[field]
                    child.updated_at = datetime.now()

    def _aggregate_to_parent(
        self,
        parent: HierarchicalContext,
        field: str,
        value: Any,
    ) -> None:
        """
        Aggregate value into parent context based on aggregation rule.

        Args:
            parent: Parent context
            field: Field name
            value: Value to aggregate
        """
        rule = self.config.aggregation_rules.get(field, "last")

        if rule == "sum":
            # Sum numeric values
            parent.data[field] = parent.data.get(field, 0) + value

        elif rule == "avg":
            # Average (simplified: store values and compute)
            if f"_{field}_values" not in parent.data:
                parent.data[f"_{field}_values"] = []
            parent.data[f"_{field}_values"].append(value)

            values = parent.data[f"_{field}_values"]
            parent.data[field] = sum(values) / len(values)

        elif rule == "worst":
            # Status ordering: idle < running < completed < error < failed
            status_order = {
                "idle": 0,
                "running": 1,
                "completed": 2,
                "error": 3,
                "failed": 4,
            }

            current = parent.data.get(field, "idle")
            current_level = status_order.get(current, 0)
            value_level = status_order.get(value, 0)

            if value_level > current_level:
                parent.data[field] = value

        elif rule == "max":
            # Maximum value
            current = parent.data.get(field, float('-inf'))
            parent.data[field] = max(current, value)

        elif rule == "min":
            # Minimum value
            current = parent.data.get(field, float('inf'))
            parent.data[field] = min(current, value)

        else:
            # Default: last value wins
            parent.data[field] = value

    def get_all_contexts(self) -> Dict[str, HierarchicalContext]:
        """
        Get all registered contexts.

        Returns:
            Dictionary mapping owner_id to context
        """
        return self._contexts.copy()

    def get_root_contexts(self) -> Dict[str, HierarchicalContext]:
        """
        Get all root contexts (those without parents).

        Returns:
            Dictionary of root contexts
        """
        return {
            owner_id: ctx
            for owner_id, ctx in self._contexts.items()
            if ctx.parent_id is None
        }
