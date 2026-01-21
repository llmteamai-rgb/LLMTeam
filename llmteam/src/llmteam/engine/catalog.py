"""
Step Catalog API.

This module provides the Step Catalog for registering and managing
step types that can be used in segments.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Dict, List

from llmteam.observability import get_logger


logger = get_logger(__name__)


class StepCategory(Enum):
    """Step categories for UI grouping."""

    AI = "ai"
    DATA = "data"
    INTEGRATION = "integration"
    CONTROL = "control"
    HUMAN = "human"
    UTILITY = "utility"


@dataclass
class PortSpec:
    """Port specification for step type."""

    name: str
    type: str = "any"  # JSON Schema type
    description: str = ""
    required: bool = True
    default: Any = None
    schema: Optional[dict] = None  # Full JSON Schema for complex types

    def to_dict(self) -> dict:
        """Serialize for UI."""
        result = {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "required": self.required,
        }
        if self.default is not None:
            result["default"] = self.default
        if self.schema:
            result["schema"] = self.schema
        return result


@dataclass
class StepTypeMetadata:
    """
    Step type metadata for Step Catalog.

    Used by canvas for:
    - Building the block palette
    - Generating configuration forms
    - Validating connections between steps
    """

    # === Identity ===
    type_id: str  # "llm_agent", "http_action"
    version: str  # "1.0"

    # === Display ===
    display_name: str  # "LLM Agent"
    description: str  # "Executes LLM prompt"
    category: StepCategory  # StepCategory.AI
    icon: str = "box"  # Icon name for UI
    color: str = "#4A90D9"  # Default color

    # === Configuration Schema ===
    config_schema: dict[str, Any] = field(default_factory=dict)  # JSON Schema

    # === Ports ===
    input_ports: list[PortSpec] = field(default_factory=list)
    output_ports: list[PortSpec] = field(default_factory=list)

    # === Behavior ===
    supports_retry: bool = True
    supports_timeout: bool = True
    supports_parallel: bool = False
    is_async: bool = True

    # === Documentation ===
    docs_url: Optional[str] = None
    examples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Export for UI consumption."""
        return {
            "type_id": self.type_id,
            "version": self.version,
            "display_name": self.display_name,
            "description": self.description,
            "category": self.category.value,
            "icon": self.icon,
            "color": self.color,
            "config_schema": self.config_schema,
            "input_ports": [p.to_dict() for p in self.input_ports],
            "output_ports": [p.to_dict() for p in self.output_ports],
            "supports_retry": self.supports_retry,
            "supports_timeout": self.supports_timeout,
            "supports_parallel": self.supports_parallel,
            "is_async": self.is_async,
            "docs_url": self.docs_url,
            "examples": self.examples,
        }


class StepCatalog:
    """
    Step type catalog.

    Singleton, accessible via StepCatalog.instance().

    The catalog maintains a registry of step types that can be used
    in segment definitions. Each step type has metadata for UI display
    and an optional handler for execution.
    """

    _instance: Optional["StepCatalog"] = None

    def __init__(self) -> None:
        self._types: dict[str, StepTypeMetadata] = {}
        self._handlers: dict[str, Callable] = {}
        self._version = "1.0"
        logger.debug("StepCatalog initialized")

    @classmethod
    def instance(cls) -> "StepCatalog":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_builtin_types()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None
        logger.debug("StepCatalog instance reset")

    def register(
        self,
        metadata: StepTypeMetadata,
        handler: Optional[Callable] = None,
    ) -> None:
        """
        Register step type.

        Args:
            metadata: Step type metadata
            handler: Optional handler function/class for execution
        """
        self._types[metadata.type_id] = metadata
        if handler:
            self._handlers[metadata.type_id] = handler
        
        logger.debug(f"Registered step type: {metadata.type_id} (handler: {bool(handler)})")

    def unregister(self, type_id: str) -> None:
        """Unregister step type."""
        if type_id in self._types:
            self._types.pop(type_id, None)
            self._handlers.pop(type_id, None)
            logger.info(f"Unregistered step type: {type_id}")
        else:
            logger.warning(f"Attempted to unregister unknown step type: {type_id}")

    def get(self, type_id: str) -> Optional[StepTypeMetadata]:
        """Get step type metadata."""
        return self._types.get(type_id)

    def get_handler(self, type_id: str) -> Optional[Callable]:
        """Get step handler."""
        handler = self._handlers.get(type_id)
        if not handler:
            logger.warning(f"No handler found for step type: {type_id}")
        return handler

    def has(self, type_id: str) -> bool:
        """Check if step type exists."""
        return type_id in self._types

    def list_all(self) -> list[StepTypeMetadata]:
        """List all registered step types."""
        return list(self._types.values())

    def list_by_category(self, category: StepCategory) -> list[StepTypeMetadata]:
        """List step types by category."""
        return [t for t in self._types.values() if t.category == category]

    def list_type_ids(self) -> list[str]:
        """List all registered type IDs."""
        return list(self._types.keys())

    def export_for_ui(self) -> dict:
        """
        Export catalog for canvas UI.

        Returns dict suitable for JSON serialization.
        """
        return {
            "version": self._version,
            "categories": [c.value for c in StepCategory],
            "types": {type_id: meta.to_dict() for type_id, meta in self._types.items()},
        }

    def validate_step_config(self, type_id: str, config: dict) -> list[str]:
        """
        Validate step config against schema.

        Returns list of error messages (empty if valid).
        """
        metadata = self.get(type_id)
        if not metadata:
            msg = f"Unknown step type: {type_id}"
            logger.warning(msg)
            return [msg]

        errors: list[str] = []

        # Basic validation against required fields in schema
        schema = metadata.config_schema
        if schema.get("type") == "object":
            required = schema.get("required", [])
            for field_name in required:
                if field_name not in config:
                    errors.append(f"Missing required field: {field_name}")

        if errors:
            logger.debug(f"Config validation failed for {type_id}: {errors}")
            
        return errors

    def _register_builtin_types(self) -> None:
        """Register built-in step types with handlers."""
        logger.debug("Registering built-in step types")

        # Import handlers
        from llmteam.engine.handlers import (
            LLMAgentHandler,
            HTTPActionHandler,
            TransformHandler,
            ConditionHandler,
            ParallelSplitHandler,
            ParallelJoinHandler,
            HumanTaskHandler,
            SubworkflowHandler,
            SwitchHandler,
        )
        from llmteam.engine.handlers.team_handler import TeamHandler

        # Create handler instances
        llm_handler = LLMAgentHandler()
        http_handler = HTTPActionHandler()
        transform_handler = TransformHandler()
        condition_handler = ConditionHandler()
        parallel_split_handler = ParallelSplitHandler()
        parallel_join_handler = ParallelJoinHandler()
        subworkflow_handler = SubworkflowHandler()
        switch_handler = SwitchHandler()
        team_handler = TeamHandler()
        try:
            human_handler = HumanTaskHandler()
            
            # Human Task
            self.register(
                StepTypeMetadata(
                    type_id="human_task",
                    version="1.0",
                    display_name="Human Task",
                    description="Request human input or approval",
                    category=StepCategory.HUMAN,
                    icon="user",
                    color="#FF6B6B",
                    config_schema={
                        "type": "object",
                        "properties": {
                            "task_type": {
                                "type": "string",
                                "enum": ["approval", "input", "review", "choice"],
                                "default": "approval",
                            },
                            "title": {
                                "type": "string",
                                "description": "Task title",
                            },
                            "description": {
                                "type": "string",
                                "description": "Task description",
                            },
                            "assignee_ref": {
                                "type": "string",
                                "description": "Reference to assignee/group",
                            },
                            "timeout_hours": {
                                "type": "number",
                                "default": 24,
                            },
                            "escalation_chain": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["task_type"],
                    },
                    input_ports=[
                        PortSpec("data", "object", "Data for human review"),
                    ],
                    output_ports=[
                        PortSpec("approved", "object", "Output if approved"),
                        PortSpec("rejected", "object", "Output if rejected"),
                        PortSpec("modified", "object", "Output if modified"),
                    ],
                    supports_parallel=False,
                ),
                handler=human_handler,
            )
        except Exception as e:
            # Skip if license does not support human tasks
            logger.info(f"Skipping HumanTaskHandler registration: {e}")


        # LLM Agent
        self.register(
            StepTypeMetadata(
                type_id="llm_agent",
                version="1.0",
                display_name="LLM Agent",
                description="Execute LLM prompt with optional tools",
                category=StepCategory.AI,
                icon="robot",
                color="#4A90D9",
                config_schema={
                    "type": "object",
                    "properties": {
                        "llm_ref": {
                            "type": "string",
                            "description": "Reference to LLM provider",
                        },
                        "prompt_template_id": {
                            "type": "string",
                            "description": "Prompt template ID",
                        },
                        "temperature": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 2,
                            "default": 0.7,
                        },
                        "max_tokens": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 1000,
                        },
                    },
                    "required": ["llm_ref"],
                },
                input_ports=[
                    PortSpec("input", "object", "Input data"),
                ],
                output_ports=[
                    PortSpec("output", "string", "LLM response"),
                    PortSpec("error", "object", "Error if failed", required=False),
                ],
            ),
            handler=llm_handler,
        )

        # HTTP Action
        self.register(
            StepTypeMetadata(
                type_id="http_action",
                version="1.0",
                display_name="HTTP Request",
                description="Make HTTP request to external API",
                category=StepCategory.INTEGRATION,
                icon="globe",
                color="#50C878",
                config_schema={
                    "type": "object",
                    "properties": {
                        "client_ref": {
                            "type": "string",
                            "description": "Reference to HTTP client",
                        },
                        "method": {
                            "type": "string",
                            "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                            "default": "POST",
                        },
                        "path": {
                            "type": "string",
                            "description": "Request path",
                        },
                        "headers": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                        "retry_count": {
                            "type": "integer",
                            "minimum": 0,
                            "default": 3,
                        },
                    },
                    "required": ["client_ref", "path"],
                },
                input_ports=[
                    PortSpec("body", "object", "Request body"),
                ],
                output_ports=[
                    PortSpec("response", "object", "Response data"),
                    PortSpec("status", "integer", "HTTP status code"),
                ],
            ),
            handler=http_handler,
        )

        # Human Task registration moved to try-except block above

        # Condition (branching)
        self.register(
            StepTypeMetadata(
                type_id="condition",
                version="1.0",
                display_name="Condition",
                description="Branch based on condition",
                category=StepCategory.CONTROL,
                icon="git-branch",
                color="#9B59B6",
                config_schema={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Condition expression (Python-like)",
                        },
                    },
                    "required": ["expression"],
                },
                input_ports=[
                    PortSpec("input", "any", "Data to evaluate"),
                ],
                output_ports=[
                    PortSpec("true", "any", "Output if true"),
                    PortSpec("false", "any", "Output if false"),
                ],
            ),
            handler=condition_handler,
        )

        # Parallel Split
        self.register(
            StepTypeMetadata(
                type_id="parallel_split",
                version="1.0",
                display_name="Parallel Split",
                description="Execute multiple branches in parallel",
                category=StepCategory.CONTROL,
                icon="git-fork",
                color="#9B59B6",
                config_schema={
                    "type": "object",
                    "properties": {
                        "branches": {
                            "type": "integer",
                            "minimum": 2,
                            "default": 2,
                        },
                    },
                },
                input_ports=[
                    PortSpec("input", "any", "Data to distribute"),
                ],
                output_ports=[
                    PortSpec("branch_1", "any", "Branch 1 output"),
                    PortSpec("branch_2", "any", "Branch 2 output"),
                ],
                supports_parallel=True,
            ),
            handler=parallel_split_handler,
        )

        # Parallel Join
        self.register(
            StepTypeMetadata(
                type_id="parallel_join",
                version="1.0",
                display_name="Parallel Join",
                description="Wait for all parallel branches",
                category=StepCategory.CONTROL,
                icon="git-merge",
                color="#9B59B6",
                config_schema={
                    "type": "object",
                    "properties": {
                        "merge_strategy": {
                            "type": "string",
                            "enum": ["all", "any", "first"],
                            "default": "all",
                        },
                    },
                },
                input_ports=[
                    PortSpec("branch_1", "any", "Branch 1 input"),
                    PortSpec("branch_2", "any", "Branch 2 input"),
                ],
                output_ports=[
                    PortSpec("output", "array", "Merged results"),
                ],
            ),
            handler=parallel_join_handler,
        )

        # Data Transform
        self.register(
            StepTypeMetadata(
                type_id="transform",
                version="1.0",
                display_name="Transform",
                description="Transform data using expression",
                category=StepCategory.DATA,
                icon="shuffle",
                color="#F39C12",
                config_schema={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Transform expression (JSONPath-like)",
                        },
                        "mapping": {
                            "type": "object",
                            "description": "Field mapping",
                        },
                    },
                },
                input_ports=[
                    PortSpec("input", "any", "Input data"),
                ],
                output_ports=[
                    PortSpec("output", "any", "Transformed data"),
                ],
            ),
            handler=transform_handler,
        )

        # Subworkflow
        self.register(
            StepTypeMetadata(
                type_id="subworkflow",
                version="1.0",
                display_name="Subworkflow",
                description="Execute a nested workflow",
                category=StepCategory.CONTROL,
                icon="layers",
                color="#8E44AD",
                config_schema={
                    "type": "object",
                    "properties": {
                        "segment_id": {
                            "type": "string",
                            "description": "ID of the segment to run",
                        },
                        "segment_ref": {
                            "type": "string",
                            "description": "Reference/Alias to segment",
                        },
                        "isolated": {
                            "type": "boolean",
                            "default": False,
                            "description": "Run in isolated trace context",
                        },
                        "input_mapping": {
                            "type": "object",
                            "description": "Map parent input to subworkflow input",
                        },
                        "output_mapping": {
                            "type": "object",
                            "description": "Map subworkflow output to parent output",
                        },
                    },
                },
                input_ports=[
                    PortSpec("input", "any", "Parent Input"),
                ],
                output_ports=[
                    PortSpec("output", "any", "Subworkflow Output"),
                ],
            ),
            handler=subworkflow_handler,
        )

        # Switch
        self.register(
            StepTypeMetadata(
                type_id="switch",
                version="1.0",
                display_name="Switch",
                description="Multi-way branching logic",
                category=StepCategory.CONTROL,
                icon="git-pull-request",
                color="#9B59B6",
                config_schema={
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Value to switch on",
                        },
                        "cases": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "value": {"type": "string"},
                                    "expression": {"type": "string"},
                                    "port": {"type": "string"},
                                },
                            },
                        },
                        "default_port": {
                            "type": "string",
                            "default": "default",
                        },
                    },
                },
                input_ports=[
                    PortSpec("input", "any", "Input Data"),
                ],
                output_ports=[
                    PortSpec("default", "any", "Default Path"),
                    # Note: Dynamic ports are added at runtime or UI configuration
                ],
            ),
            handler=switch_handler,
        )

        # Team (v2.3.0)
        self.register(
            StepTypeMetadata(
                type_id="team",
                version="1.0",
                display_name="Team",
                description="Invoke an agent team as a workflow step",
                category=StepCategory.AI,
                icon="users",
                color="#27AE60",
                config_schema={
                    "type": "object",
                    "properties": {
                        "team_ref": {
                            "type": "string",
                            "description": "Reference to registered team",
                        },
                        "input_mapping": {
                            "type": "object",
                            "description": "Map step input to team input",
                            "additionalProperties": {"type": "string"},
                        },
                        "output_mapping": {
                            "type": "object",
                            "description": "Map team output to step output",
                            "additionalProperties": {"type": "string"},
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Execution timeout in seconds",
                        },
                    },
                    "required": ["team_ref"],
                },
                input_ports=[
                    PortSpec("input", "any", "Input Data"),
                ],
                output_ports=[
                    PortSpec("output", "any", "Team Output"),
                ],
            ),
            handler=team_handler,
        )
