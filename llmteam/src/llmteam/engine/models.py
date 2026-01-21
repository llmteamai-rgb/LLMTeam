"""
Engine Workflow Models.

This module defines the JSON contract for Workflows:
- WorkflowDefinition: Main workflow container (formerly SegmentDefinition)
- StepDefinition: Individual step in a workflow
- EdgeDefinition: Connection between steps
"""

from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict, List, ClassVar
import json


@dataclass
class PortDefinition:
    """Port definition for a step."""

    name: str
    type: str = "any"  # "any", "string", "object", "array"
    required: bool = True
    description: str = ""

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return JSON Schema for PortDefinition."""
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Port name"},
                "type": {
                    "type": "string",
                    "enum": ["any", "string", "object", "array", "number", "boolean"],
                    "default": "any",
                    "description": "Data type accepted by the port",
                },
                "required": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether this port must be connected",
                },
                "description": {
                    "type": "string",
                    "default": "",
                    "description": "Human-readable description of the port",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        }


class StepPositionDict(TypedDict):
    """Dictionary representation of StepPosition."""
    x: float
    y: float


@dataclass
class StepPosition:
    """Position in workflow editor."""

    x: float
    y: float

    def to_dict(self) -> StepPositionDict:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: dict) -> "StepPosition":
        return cls(x=data["x"], y=data["y"])

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return JSON Schema for StepPosition."""
        return {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "X coordinate on canvas"},
                "y": {"type": "number", "description": "Y coordinate on canvas"},
            },
            "required": ["x", "y"],
            "additionalProperties": False,
        }


class StepUIMetadataDict(TypedDict, total=False):
    """Dictionary representation of StepUIMetadata."""
    color: Optional[str]
    icon: Optional[str]
    collapsed: bool


@dataclass
class StepUIMetadata:
    """UI metadata for a step."""

    color: Optional[str] = None
    icon: Optional[str] = None
    collapsed: bool = False

    def to_dict(self) -> StepUIMetadataDict:
        result: StepUIMetadataDict = {"collapsed": self.collapsed}
        if self.color:
            result["color"] = self.color
        if self.icon:
            result["icon"] = self.icon
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "StepUIMetadata":
        return cls(
            color=data.get("color"),
            icon=data.get("icon"),
            collapsed=data.get("collapsed", False),
        )

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return JSON Schema for StepUIMetadata."""
        return {
            "type": "object",
            "properties": {
                "color": {
                    "type": "string",
                    "description": "CSS color for the step node",
                },
                "icon": {
                    "type": "string",
                    "description": "Icon identifier for the step",
                },
                "collapsed": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether the step is collapsed in UI",
                },
            },
            "additionalProperties": False,
        }


class StepPortsDict(TypedDict):
    """Dictionary representation of step ports."""
    input: List[str]
    output: List[str]


class StepDefinitionDict(TypedDict, total=False):
    """Dictionary representation of StepDefinition."""
    step_id: str
    type: str
    config: dict[str, Any]
    ports: StepPortsDict
    name: str
    position: StepPositionDict
    ui: StepUIMetadataDict


@dataclass
class StepDefinition:
    """
    Step definition in a segment.

    Represents a single node in the workflow graph.
    """

    step_id: str
    type: str  # Reference to Step Catalog type_id
    name: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    # Ports
    input_ports: list[str] = field(default_factory=lambda: ["input"])
    output_ports: list[str] = field(default_factory=lambda: ["output"])

    # UI
    position: Optional[StepPosition] = None
    ui: Optional[StepUIMetadata] = None

    def to_dict(self) -> StepDefinitionDict:
        """Serialize to dict."""
        ports: StepPortsDict = {
            "input": self.input_ports,
            "output": self.output_ports,
        }
        
        result: StepDefinitionDict = {
            "step_id": self.step_id,
            "type": self.type,
            "config": self.config,
            "ports": ports,
        }
        if self.name:
            result["name"] = self.name
        if self.position:
            result["position"] = self.position.to_dict()
        if self.ui:
            result["ui"] = self.ui.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "StepDefinition":
        """Deserialize from dict."""
        ports = data.get("ports", {})
        position = None
        if data.get("position"):
            position = StepPosition.from_dict(data["position"])
        ui = None
        if data.get("ui"):
            ui = StepUIMetadata.from_dict(data["ui"])

        return cls(
            step_id=data["step_id"],
            type=data.get("type") or data.get("step_type") or "", # Support alias
            name=data.get("name", ""),
            config=data.get("config", {}),
            input_ports=ports.get("input", ["input"]),
            output_ports=ports.get("output", ["output"]),
            position=position,
            ui=ui,
        )

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return JSON Schema for StepDefinition."""
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "step_id": {
                    "type": "string",
                    "pattern": "^[a-z][a-z0-9_]*$",
                    "description": "Unique identifier for the step (lowercase, starts with letter)",
                },
                "type": {
                    "type": "string",
                    "description": "Step type from the Step Catalog",
                },
                "name": {
                    "type": "string",
                    "description": "Human-readable name for the step",
                },
                "config": {
                    "type": "object",
                    "description": "Step-specific configuration",
                    "additionalProperties": True,
                },
                "ports": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": ["input"],
                            "description": "Input port names",
                        },
                        "output": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": ["output"],
                            "description": "Output port names",
                        },
                    },
                },
                "position": StepPosition.json_schema(),
                "ui": StepUIMetadata.json_schema(),
            },
            "required": ["step_id", "type"],
            "additionalProperties": False,
        }


class EdgeDefinitionDict(TypedDict, total=False):
    """Dictionary representation of EdgeDefinition."""
    # reserved keyword 'from' cannot be used as key in class definition syntax for TypedDict
    # so we use alternative syntax or accept keys matching to_dict() logic
    # Here we stick to flexible TypedDict or just Dict[str, Any] for edge since 'from' is reserved.
    # But wait, TypedDict keys can be string literals.
    pass

# Using functional syntax for EdgeDefinitionDict because 'from' is a keyword
EdgeDefinitionDict = TypedDict("EdgeDefinitionDict", {
    "from": str,
    "from_port": str,
    "to": str,
    "to_port": str,
    "condition": Optional[str],
}, total=False)


@dataclass
class EdgeDefinition:
    """
    Edge definition connecting two steps.

    Represents a directed connection in the workflow graph.
    """

    from_step: str
    to_step: str
    from_port: str = "output"
    to_port: str = "input"
    condition: Optional[str] = None  # Expression for conditional transitions

    def to_dict(self) -> EdgeDefinitionDict:
        """Serialize to dict."""
        result: EdgeDefinitionDict = {
            "from": self.from_step,
            "from_port": self.from_port,
            "to": self.to_step,
            "to_port": self.to_port,
        }
        if self.condition:
            result["condition"] = self.condition
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "EdgeDefinition":
        """Deserialize from dict."""
        return cls(
            from_step=data["from"],
            to_step=data["to"],
            from_port=data.get("from_port", "output"),
            to_port=data.get("to_port", "input"),
            condition=data.get("condition"),
        )

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return JSON Schema for EdgeDefinition."""
        return {
            "type": "object",
            "properties": {
                "from": {
                    "type": "string",
                    "description": "Source step ID",
                },
                "from_port": {
                    "type": "string",
                    "default": "output",
                    "description": "Source port name",
                },
                "to": {
                    "type": "string",
                    "description": "Target step ID",
                },
                "to_port": {
                    "type": "string",
                    "default": "input",
                    "description": "Target port name",
                },
                "condition": {
                    "type": "string",
                    "description": "Conditional expression for the edge",
                },
            },
            "required": ["from", "to"],
            "additionalProperties": False,
        }


class WorkflowParamsDict(TypedDict):
    """Dictionary representation of WorkflowParams."""
    max_retries: int
    timeout_seconds: float
    parallel_execution: bool


# Backward compatibility alias
SegmentParamsDict = WorkflowParamsDict


@dataclass
class WorkflowParams:
    """Workflow-level parameters."""

    max_retries: int = 3
    timeout_seconds: float = 300
    parallel_execution: bool = False

    def to_dict(self) -> WorkflowParamsDict:
        return {
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "parallel_execution": self.parallel_execution,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowParams":
        return cls(
            max_retries=data.get("max_retries", 3),
            timeout_seconds=data.get("timeout_seconds", 300),
            parallel_execution=data.get("parallel_execution", False),
        )

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return JSON Schema for WorkflowParams."""
        return {
            "type": "object",
            "properties": {
                "max_retries": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 3,
                    "description": "Maximum retry attempts for failed steps",
                },
                "timeout_seconds": {
                    "type": "number",
                    "minimum": 0,
                    "default": 300,
                    "description": "Segment execution timeout in seconds",
                },
                "parallel_execution": {
                    "type": "boolean",
                    "default": False,
                    "description": "Enable parallel execution of independent steps",
                },
            },
            "additionalProperties": False,
        }


class WorkflowDefinitionDict(TypedDict, total=False):
    """Dictionary representation of WorkflowDefinition."""
    version: str
    workflow_id: str
    name: str
    description: str
    entrypoint: str
    params: WorkflowParamsDict
    steps: List[StepDefinitionDict]
    edges: List[EdgeDefinitionDict]
    metadata: dict[str, Any]


# Backward compatibility alias
SegmentDefinitionDict = WorkflowDefinitionDict


@dataclass
class WorkflowDefinition:
    """
    Workflow Definition.

    This is the main JSON contract for engine integration.
    A workflow represents a complete execution graph that can be run.

    Formerly known as SegmentDefinition.
    """

    workflow_id: str
    name: str
    entrypoint: str
    steps: list[StepDefinition]

    # Optional fields
    description: str = ""
    version: str = "1.0"
    params: WorkflowParams = field(default_factory=WorkflowParams)
    edges: list[EdgeDefinition] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Backward compatibility property
    @property
    def segment_id(self) -> str:
        """Deprecated: Use workflow_id instead."""
        return self.workflow_id

    def to_dict(self) -> WorkflowDefinitionDict:
        """Serialize to dict (JSON-compatible)."""
        return {
            "version": self.version,
            "workflow_id": self.workflow_id,
            "segment_id": self.workflow_id,  # Backward compatibility alias
            "name": self.name,
            "description": self.description,
            "entrypoint": self.entrypoint,
            "params": self.params.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "edges": [e.to_dict() for e in self.edges],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowDefinition":
        """Deserialize from dict."""
        steps = [StepDefinition.from_dict(s) for s in data["steps"]]

        edges = [EdgeDefinition.from_dict(e) for e in data.get("edges", [])]

        params = WorkflowParams.from_dict(data.get("params", {}))

        # Support both workflow_id and segment_id for backward compatibility
        workflow_id = data.get("workflow_id") or data.get("segment_id")

        return cls(
            workflow_id=workflow_id,
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            entrypoint=data["entrypoint"],
            params=params,
            steps=steps,
            edges=edges,
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "WorkflowDefinition":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def validate(self) -> list[str]:
        """
        Validate workflow definition.

        Returns list of error messages (empty if valid).
        """
        errors: list[str] = []

        step_ids = {s.step_id for s in self.steps}

        # Check entrypoint exists
        if self.entrypoint not in step_ids:
            errors.append(f"Entrypoint '{self.entrypoint}' not found in steps")

        # Check edges reference valid steps
        for edge in self.edges:
            if edge.from_step not in step_ids:
                errors.append(f"Edge from '{edge.from_step}' references unknown step")
            if edge.to_step not in step_ids:
                errors.append(f"Edge to '{edge.to_step}' references unknown step")

        # Check for duplicate step IDs
        if len(step_ids) != len(self.steps):
            errors.append("Duplicate step IDs found")

        # Check step_id format (lowercase, starts with letter)
        import re

        id_pattern = re.compile(r"^[a-z][a-z0-9_]*$")
        for step in self.steps:
            if not id_pattern.match(step.step_id):
                errors.append(
                    f"Invalid step_id format: '{step.step_id}' "
                    "(must be lowercase, start with letter)"
                )

        if not id_pattern.match(self.workflow_id):
            errors.append(
                f"Invalid segment_id/workflow_id format: '{self.workflow_id}' "
                "(must be lowercase, start with letter)"
            )

        return errors

    def get_step(self, step_id: str) -> Optional[StepDefinition]:
        """Get step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_outgoing_edges(self, step_id: str) -> list[EdgeDefinition]:
        """Get all edges starting from a step."""
        return [e for e in self.edges if e.from_step == step_id]

    def get_incoming_edges(self, step_id: str) -> list[EdgeDefinition]:
        """Get all edges ending at a step."""
        return [e for e in self.edges if e.to_step == step_id]

    def get_next_steps(self, step_id: str) -> list[str]:
        """Get IDs of steps that follow a given step."""
        return [e.to_step for e in self.get_outgoing_edges(step_id)]

    def get_previous_steps(self, step_id: str) -> list[str]:
        """Get IDs of steps that precede a given step."""
        return [e.from_step for e in self.get_incoming_edges(step_id)]

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """
        Return JSON Schema for WorkflowDefinition.

        This schema can be used for:
        - Validating workflow JSON files
        - Generating documentation
        - IDE autocompletion support
        - Workflow UI integration

        Returns:
            dict: Complete JSON Schema for WorkflowDefinition
        """
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://llmteam.ai/schemas/workflow/v1.0",
            "title": "LLMTeam Workflow Definition",
            "description": "Workflow definition for execution engine",
            "type": "object",
            "properties": {
                "version": {
                    "type": "string",
                    "default": "1.0",
                    "description": "Schema version for the workflow definition",
                },
                "workflow_id": {
                    "type": "string",
                    "pattern": "^[a-z][a-z0-9_]*$",
                    "description": "Unique identifier for the workflow (lowercase, starts with letter)",
                },
                "name": {
                    "type": "string",
                    "description": "Human-readable name for the workflow",
                },
                "description": {
                    "type": "string",
                    "default": "",
                    "description": "Detailed description of the workflow's purpose",
                },
                "entrypoint": {
                    "type": "string",
                    "description": "Step ID where execution begins",
                },
                "params": WorkflowParams.json_schema(),
                "steps": {
                    "type": "array",
                    "items": StepDefinition.json_schema(),
                    "minItems": 1,
                    "description": "List of step definitions in the workflow",
                },
                "edges": {
                    "type": "array",
                    "items": EdgeDefinition.json_schema(),
                    "default": [],
                    "description": "List of edges connecting steps",
                },
                "metadata": {
                    "type": "object",
                    "additionalProperties": True,
                    "default": {},
                    "description": "Additional metadata for the workflow",
                },
            },
            "required": ["workflow_id", "name", "entrypoint", "steps"],
            "additionalProperties": False,
        }

    @classmethod
    def json_schema_string(cls, indent: int = 2) -> str:
        """Return JSON Schema as a formatted JSON string."""
        return json.dumps(cls.json_schema(), indent=indent)


# =============================================================================
# Backward compatibility aliases (deprecated)
# =============================================================================

# These aliases allow existing code to continue working while migrating to new names
SegmentParams = WorkflowParams


class SegmentDefinition:
    """
    Backward compatibility wrapper for WorkflowDefinition.

    DEPRECATED: Use WorkflowDefinition instead.

    This wrapper class accepts both segment_id (deprecated) and workflow_id,
    and proxies all class methods to WorkflowDefinition.
    """

    def __new__(
        cls,
        segment_id: str = None,
        workflow_id: str = None,
        name: str = None,
        entrypoint: str = None,
        steps: list = None,
        **kwargs,
    ) -> WorkflowDefinition:
        """
        Create a WorkflowDefinition instance.

        Args:
            segment_id: Deprecated, use workflow_id instead.
            workflow_id: Workflow identifier.
            name: Workflow name.
            entrypoint: Entry step ID.
            steps: List of steps.
            **kwargs: Other WorkflowDefinition parameters.

        Returns:
            WorkflowDefinition instance.
        """
        # Map segment_id to workflow_id for backward compatibility
        if segment_id is not None and workflow_id is None:
            workflow_id = segment_id
        elif segment_id is None and workflow_id is None:
            raise TypeError("SegmentDefinition() requires 'workflow_id' or 'segment_id' argument")

        return WorkflowDefinition(
            workflow_id=workflow_id,
            name=name or "",
            entrypoint=entrypoint or "",
            steps=steps or [],
            **kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict) -> WorkflowDefinition:
        """
        Create from dictionary (backward compatibility).

        Accepts both segment_id and workflow_id in the data.
        """
        # Map segment_id to workflow_id if present
        if "segment_id" in data and "workflow_id" not in data:
            data = data.copy()
            data["workflow_id"] = data.pop("segment_id")

        return WorkflowDefinition.from_dict(data)

    @classmethod
    def from_json(cls, json_str: str) -> WorkflowDefinition:
        """Create from JSON string (backward compatibility)."""
        data = json.loads(json_str)
        return cls.from_dict(data)

