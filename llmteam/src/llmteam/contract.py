"""
Team Contract Definition.

Provides formal input/output contracts for LLMTeams.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from llmteam.ports.models import TypedPort, PortDirection, PortLevel


@dataclass
class ContractValidationResult:
    """Result of contract validation."""
    valid: bool
    errors: List[str]


class ContractValidationError(Exception):
    """Raised when contract validation fails."""
    def __init__(self, message: str, errors: List[str]):
        super().__init__(f"{message}: {', '.join(errors)}")
        self.errors = errors


@dataclass
class TeamContract:
    """
    Formal contract for an LLMTeam.

    Defines input and output ports for validation and documentation.
    """

    inputs: List[TypedPort] = field(default_factory=list)
    outputs: List[TypedPort] = field(default_factory=list)
    name: str = "TeamContract"
    version: str = "1.0"
    description: str = ""

    def validate_input(self, data: dict[str, Any]) -> ContractValidationResult:
        """
        Validate input data against the contract.

        Args:
            data: Input dictionary

        Returns:
            ContractValidationResult
        """
        errors = []
        for port in self.inputs:
            # Check required fields
            if port.required and port.name not in data:
                errors.append(f"Missing required input: {port.name}")

            # TODO: Add JSON Schema validation if port.schema is defined

        return ContractValidationResult(valid=len(errors) == 0, errors=errors)

    def validate_output(self, data: dict[str, Any]) -> ContractValidationResult:
        """
        Validate output data against the contract.

        Args:
            data: Output dictionary

        Returns:
            ContractValidationResult
        """
        errors = []
        for port in self.outputs:
            # Check required fields
            if port.required and port.name not in data:
                errors.append(f"Missing required output: {port.name}")

        return ContractValidationResult(valid=len(errors) == 0, errors=errors)

    @classmethod
    def default(cls) -> "TeamContract":
        """
        Get default loose contract for backward compatibility.

        Returns a contract with optional 'input' and 'output' ports.
        """
        return cls(
            inputs=[TypedPort(
                name="input",
                level=PortLevel.WORKFLOW,
                direction=PortDirection.INPUT,
                required=False,
                description="Default input"
            )],
            outputs=[TypedPort(
                name="output",
                level=PortLevel.WORKFLOW,
                direction=PortDirection.OUTPUT,
                required=False,
                description="Default output"
            )],
            name="DefaultContract",
            description="Default backward-compatible contract"
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "inputs": [p.to_dict() for p in self.inputs],
            "outputs": [p.to_dict() for p in self.outputs],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TeamContract":
        """Deserialize from dictionary."""
        return cls(
            name=data.get("name", "TeamContract"),
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            inputs=[TypedPort.from_dict(p) for p in data.get("inputs", [])],
            outputs=[TypedPort.from_dict(p) for p in data.get("outputs", [])],
        )
