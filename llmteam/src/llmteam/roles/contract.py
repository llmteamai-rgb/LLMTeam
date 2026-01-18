"""
Team Contract Module.

Provides formal interface definitions for team inputs/outputs.
TeamContract enforces type-safe data exchange between workflow steps and agent teams.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llmteam.ports import TypedPort, PortDirection, PortLevel
from llmteam.observability import get_logger


logger = get_logger(__name__)


class ContractValidationError(Exception):
    """Raised when contract validation fails."""

    def __init__(self, message: str, errors: List[str]):
        super().__init__(message)
        self.errors = errors


@dataclass
class ValidationResult:
    """Result of contract validation."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid


@dataclass
class TeamContract:
    """
    Formal contract defining team's input/output interface.

    TeamContract ensures type-safe data exchange between workflow steps
    and agent teams, enabling Canvas to treat teams as regular steps.

    Attributes:
        name: Contract name for identification
        inputs: List of input port definitions
        outputs: List of output port definitions
        strict: If True, extra fields are rejected
        version: Contract version for compatibility checks

    Example:
        contract = TeamContract(
            name="translation_team",
            inputs=[
                TypedPort(
                    name="text",
                    level=PortLevel.WORKFLOW,
                    direction=PortDirection.INPUT,
                    data_type="string",
                    required=True,
                ),
                TypedPort(
                    name="target_language",
                    level=PortLevel.WORKFLOW,
                    direction=PortDirection.INPUT,
                    data_type="string",
                    required=True,
                ),
            ],
            outputs=[
                TypedPort(
                    name="translated_text",
                    level=PortLevel.WORKFLOW,
                    direction=PortDirection.OUTPUT,
                    data_type="string",
                    required=True,
                ),
            ],
        )

        # Validate input data
        result = contract.validate_input({"text": "Hello", "target_language": "ru"})
        if not result.valid:
            raise ContractValidationError("Invalid input", result.errors)
    """

    name: str
    inputs: List[TypedPort] = field(default_factory=list)
    outputs: List[TypedPort] = field(default_factory=list)
    strict: bool = False
    version: str = "1.0.0"
    description: str = ""

    def validate_input(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate input data against contract inputs.

        Args:
            data: Input data dictionary

        Returns:
            ValidationResult with validation status and errors
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Check required fields
        for port in self.inputs:
            if port.required and port.name not in data:
                errors.append(f"Missing required input: '{port.name}'")

        # Check for unknown fields in strict mode
        if self.strict:
            known_fields = {p.name for p in self.inputs}
            for key in data.keys():
                if key not in known_fields:
                    errors.append(f"Unknown input field: '{key}'")

        # Type validation (basic)
        for port in self.inputs:
            if port.name in data:
                value = data[port.name]
                if not self._validate_type(value, port.data_type):
                    errors.append(
                        f"Type mismatch for '{port.name}': expected {port.data_type}, "
                        f"got {type(value).__name__}"
                    )

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def validate_output(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate output data against contract outputs.

        Args:
            data: Output data dictionary

        Returns:
            ValidationResult with validation status and errors
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Check required fields
        for port in self.outputs:
            if port.required and port.name not in data:
                errors.append(f"Missing required output: '{port.name}'")

        # Check for unknown fields in strict mode
        if self.strict:
            known_fields = {p.name for p in self.outputs}
            for key in data.keys():
                if key not in known_fields:
                    warnings.append(f"Extra output field: '{key}'")

        # Type validation (basic)
        for port in self.outputs:
            if port.name in data:
                value = data[port.name]
                if not self._validate_type(value, port.data_type):
                    errors.append(
                        f"Type mismatch for '{port.name}': expected {port.data_type}, "
                        f"got {type(value).__name__}"
                    )

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """
        Validate value against expected type.

        Args:
            value: Value to validate
            expected_type: Expected type string

        Returns:
            True if valid, False otherwise
        """
        if expected_type == "any":
            return True

        type_map = {
            "string": str,
            "number": (int, float),
            "boolean": bool,
            "object": dict,
            "array": list,
        }

        expected = type_map.get(expected_type)
        if expected is None:
            return True  # Unknown type, allow

        return isinstance(value, expected)

    @classmethod
    def default(cls) -> "TeamContract":
        """
        Create a default contract for backward compatibility.

        Returns:
            TeamContract with flexible input/output
        """
        return cls(
            name="default",
            inputs=[
                TypedPort(
                    name="input",
                    level=PortLevel.AGENT,
                    direction=PortDirection.INPUT,
                    data_type="any",
                    required=False,
                    description="Default input port",
                ),
            ],
            outputs=[
                TypedPort(
                    name="output",
                    level=PortLevel.AGENT,
                    direction=PortDirection.OUTPUT,
                    data_type="any",
                    required=False,
                    description="Default output port",
                ),
            ],
            strict=False,
            description="Default contract for backward compatibility",
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert contract to dictionary."""
        return {
            "name": self.name,
            "inputs": [p.to_dict() for p in self.inputs],
            "outputs": [p.to_dict() for p in self.outputs],
            "strict": self.strict,
            "version": self.version,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeamContract":
        """Create contract from dictionary."""
        return cls(
            name=data["name"],
            inputs=[TypedPort.from_dict(p) for p in data.get("inputs", [])],
            outputs=[TypedPort.from_dict(p) for p in data.get("outputs", [])],
            strict=data.get("strict", False),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
        )
