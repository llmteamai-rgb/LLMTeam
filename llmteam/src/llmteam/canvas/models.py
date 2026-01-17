"""
Canvas Segment Models.

This module defines the JSON contract for Worktrail Segments:
- SegmentDefinition: Main segment container
- StepDefinition: Individual step in a segment
- EdgeDefinition: Connection between steps
"""

from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict, List
import json


@dataclass
class PortDefinition:
    """Port definition for a step."""

    name: str
    type: str = "any"  # "any", "string", "object", "array"
    required: bool = True
    description: str = ""


class StepPositionDict(TypedDict):
    """Dictionary representation of StepPosition."""
    x: float
    y: float


@dataclass
class StepPosition:
    """Position on canvas."""

    x: float
    y: float

    def to_dict(self) -> StepPositionDict:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: dict) -> "StepPosition":
        return cls(x=data["x"], y=data["y"])


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
            type=data["type"],
            name=data.get("name", ""),
            config=data.get("config", {}),
            input_ports=ports.get("input", ["input"]),
            output_ports=ports.get("output", ["output"]),
            position=position,
            ui=ui,
        )


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


class SegmentParamsDict(TypedDict):
    """Dictionary representation of SegmentParams."""
    max_retries: int
    timeout_seconds: float
    parallel_execution: bool


@dataclass
class SegmentParams:
    """Segment-level parameters."""

    max_retries: int = 3
    timeout_seconds: float = 300
    parallel_execution: bool = False

    def to_dict(self) -> SegmentParamsDict:
        return {
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "parallel_execution": self.parallel_execution,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SegmentParams":
        return cls(
            max_retries=data.get("max_retries", 3),
            timeout_seconds=data.get("timeout_seconds", 300),
            parallel_execution=data.get("parallel_execution", False),
        )


class SegmentDefinitionDict(TypedDict, total=False):
    """Dictionary representation of SegmentDefinition."""
    version: str
    segment_id: str
    name: str
    description: str
    entrypoint: str
    params: SegmentParamsDict
    steps: List[StepDefinitionDict]
    edges: List[EdgeDefinitionDict]
    metadata: dict[str, Any]


@dataclass
class SegmentDefinition:
    """
    Worktrail Segment Definition.

    This is the main JSON contract for canvas integration.
    A segment represents a complete workflow that can be executed.
    """

    segment_id: str
    name: str
    entrypoint: str
    steps: list[StepDefinition]

    # Optional fields
    description: str = ""
    version: str = "1.0"
    params: SegmentParams = field(default_factory=SegmentParams)
    edges: list[EdgeDefinition] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> SegmentDefinitionDict:
        """Serialize to dict (JSON-compatible)."""
        return {
            "version": self.version,
            "segment_id": self.segment_id,
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
    def from_dict(cls, data: dict) -> "SegmentDefinition":
        """Deserialize from dict."""
        steps = [StepDefinition.from_dict(s) for s in data["steps"]]

        edges = [EdgeDefinition.from_dict(e) for e in data.get("edges", [])]

        params = SegmentParams.from_dict(data.get("params", {}))

        return cls(
            segment_id=data["segment_id"],
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
    def from_json(cls, json_str: str) -> "SegmentDefinition":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def validate(self) -> list[str]:
        """
        Validate segment definition.

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

        if not id_pattern.match(self.segment_id):
            errors.append(
                f"Invalid segment_id format: '{self.segment_id}' "
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

