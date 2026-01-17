"""
Three-Level Ports Model.

This module provides the port hierarchy for workflow steps:
- WORKFLOW level: External communication (KorpOS, webhooks)
- AGENT level: Internal agent-to-agent communication
- HUMAN level: Human interaction ports
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class PortLevel(Enum):
    """
    Port communication level.

    Defines the scope and visibility of data flowing through ports.
    """

    WORKFLOW = "workflow"  # External: KorpOS, webhooks, API calls
    AGENT = "agent"  # Internal: Between agents in the pipeline
    HUMAN = "human"  # Human interaction: Approvals, chat, escalation


class PortDirection(Enum):
    """Port direction."""

    INPUT = "input"
    OUTPUT = "output"


class PortDataType(Enum):
    """Common port data types."""

    ANY = "any"
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"
    FILE = "file"
    IMAGE = "image"
    MESSAGE = "message"  # Chat message
    APPROVAL = "approval"  # Approval request/response


@dataclass
class TypedPort:
    """
    A typed port definition with level and metadata.

    Attributes:
        name: Port identifier
        level: Communication level (workflow, agent, human)
        direction: Input or output
        data_type: Data type flowing through this port
        required: Whether this port must be connected
        description: Human-readable description
        schema: Optional JSON schema for validation
    """

    name: str
    level: PortLevel
    direction: PortDirection
    data_type: str = "any"
    required: bool = True
    description: str = ""
    schema: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "level": self.level.value,
            "direction": self.direction.value,
            "data_type": self.data_type,
            "required": self.required,
            "description": self.description,
            "schema": self.schema,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TypedPort":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            level=PortLevel(data["level"]),
            direction=PortDirection(data["direction"]),
            data_type=data.get("data_type", "any"),
            required=data.get("required", True),
            description=data.get("description", ""),
            schema=data.get("schema"),
        )


@dataclass
class StepPorts:
    """
    Complete port specification for a step.

    Organizes ports by level and direction for easy access.
    """

    # Workflow level
    workflow_in: list[TypedPort] = field(default_factory=list)
    workflow_out: list[TypedPort] = field(default_factory=list)

    # Agent level
    agent_in: list[TypedPort] = field(default_factory=list)
    agent_out: list[TypedPort] = field(default_factory=list)

    # Human level
    human_in: list[TypedPort] = field(default_factory=list)
    human_out: list[TypedPort] = field(default_factory=list)

    def get_inputs(self, level: Optional[PortLevel] = None) -> list[TypedPort]:
        """Get all input ports, optionally filtered by level."""
        if level == PortLevel.WORKFLOW:
            return self.workflow_in
        elif level == PortLevel.AGENT:
            return self.agent_in
        elif level == PortLevel.HUMAN:
            return self.human_in
        else:
            return self.workflow_in + self.agent_in + self.human_in

    def get_outputs(self, level: Optional[PortLevel] = None) -> list[TypedPort]:
        """Get all output ports, optionally filtered by level."""
        if level == PortLevel.WORKFLOW:
            return self.workflow_out
        elif level == PortLevel.AGENT:
            return self.agent_out
        elif level == PortLevel.HUMAN:
            return self.human_out
        else:
            return self.workflow_out + self.agent_out + self.human_out

    def get_port(self, name: str) -> Optional[TypedPort]:
        """Get a port by name."""
        for port in self.get_inputs() + self.get_outputs():
            if port.name == name:
                return port
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "workflow_in": [p.to_dict() for p in self.workflow_in],
            "workflow_out": [p.to_dict() for p in self.workflow_out],
            "agent_in": [p.to_dict() for p in self.agent_in],
            "agent_out": [p.to_dict() for p in self.agent_out],
            "human_in": [p.to_dict() for p in self.human_in],
            "human_out": [p.to_dict() for p in self.human_out],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StepPorts":
        """Create from dictionary."""
        return cls(
            workflow_in=[TypedPort.from_dict(p) for p in data.get("workflow_in", [])],
            workflow_out=[TypedPort.from_dict(p) for p in data.get("workflow_out", [])],
            agent_in=[TypedPort.from_dict(p) for p in data.get("agent_in", [])],
            agent_out=[TypedPort.from_dict(p) for p in data.get("agent_out", [])],
            human_in=[TypedPort.from_dict(p) for p in data.get("human_in", [])],
            human_out=[TypedPort.from_dict(p) for p in data.get("human_out", [])],
        )


# Port factories for common patterns


def workflow_input(
    name: str,
    data_type: str = "any",
    required: bool = True,
    description: str = "",
) -> TypedPort:
    """Create a workflow-level input port."""
    return TypedPort(
        name=name,
        level=PortLevel.WORKFLOW,
        direction=PortDirection.INPUT,
        data_type=data_type,
        required=required,
        description=description,
    )


def workflow_output(
    name: str,
    data_type: str = "any",
    description: str = "",
) -> TypedPort:
    """Create a workflow-level output port."""
    return TypedPort(
        name=name,
        level=PortLevel.WORKFLOW,
        direction=PortDirection.OUTPUT,
        data_type=data_type,
        required=False,
        description=description,
    )


def agent_input(
    name: str,
    data_type: str = "any",
    required: bool = True,
    description: str = "",
) -> TypedPort:
    """Create an agent-level input port."""
    return TypedPort(
        name=name,
        level=PortLevel.AGENT,
        direction=PortDirection.INPUT,
        data_type=data_type,
        required=required,
        description=description,
    )


def agent_output(
    name: str,
    data_type: str = "any",
    description: str = "",
) -> TypedPort:
    """Create an agent-level output port."""
    return TypedPort(
        name=name,
        level=PortLevel.AGENT,
        direction=PortDirection.OUTPUT,
        data_type=data_type,
        required=False,
        description=description,
    )


def human_input(
    name: str,
    data_type: str = "message",
    required: bool = True,
    description: str = "",
) -> TypedPort:
    """Create a human-level input port."""
    return TypedPort(
        name=name,
        level=PortLevel.HUMAN,
        direction=PortDirection.INPUT,
        data_type=data_type,
        required=required,
        description=description,
    )


def human_output(
    name: str,
    data_type: str = "message",
    description: str = "",
) -> TypedPort:
    """Create a human-level output port."""
    return TypedPort(
        name=name,
        level=PortLevel.HUMAN,
        direction=PortDirection.OUTPUT,
        data_type=data_type,
        required=False,
        description=description,
    )


# Pre-defined port sets for common step types


def llm_agent_ports() -> StepPorts:
    """Standard ports for an LLM agent step."""
    return StepPorts(
        workflow_in=[
            workflow_input("config", "object", False, "LLM configuration overrides"),
        ],
        workflow_out=[
            workflow_output("result", "object", "Structured result"),
        ],
        agent_in=[
            agent_input("prompt", "string", True, "Input prompt or message"),
            agent_input("context", "object", False, "Additional context"),
        ],
        agent_out=[
            agent_output("response", "string", "LLM response"),
            agent_output("metadata", "object", "Token usage, latency, etc."),
        ],
    )


def human_task_ports() -> StepPorts:
    """Standard ports for a human task step."""
    return StepPorts(
        workflow_in=[
            workflow_input("task_config", "object", False, "Task configuration"),
        ],
        workflow_out=[
            workflow_output("result", "object", "Task result"),
        ],
        agent_in=[
            agent_input("data", "object", True, "Data for human review"),
        ],
        agent_out=[
            agent_output("response", "object", "Human response"),
        ],
        human_in=[
            human_input("request", "object", True, "Request for human"),
        ],
        human_out=[
            human_output("decision", "object", "Human decision"),
            human_output("feedback", "message", "Human feedback"),
        ],
    )


def transform_ports() -> StepPorts:
    """Standard ports for a transform/mapping step."""
    return StepPorts(
        agent_in=[
            agent_input("input", "any", True, "Data to transform"),
        ],
        agent_out=[
            agent_output("output", "any", "Transformed data"),
        ],
    )


def http_action_ports() -> StepPorts:
    """Standard ports for an HTTP action step."""
    return StepPorts(
        workflow_in=[
            workflow_input("config", "object", True, "HTTP request configuration"),
        ],
        workflow_out=[
            workflow_output("response", "object", "HTTP response"),
        ],
        agent_in=[
            agent_input("body", "object", False, "Request body data"),
            agent_input("params", "object", False, "Query parameters"),
        ],
        agent_out=[
            agent_output("data", "any", "Response data"),
            agent_output("status", "number", "HTTP status code"),
        ],
    )
