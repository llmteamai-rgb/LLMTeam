"""
Workflow Validation.

This module provides validation for workflow definitions:
- Structure validation (steps, edges, entrypoint)
- Config validation against JSON Schema
- Connection validation (port compatibility)

Formerly known as Segment Validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from llmteam.engine.models import WorkflowDefinition, StepDefinition
    from llmteam.engine.catalog import StepCatalog
    # Backward compatibility aliases
    SegmentDefinition = WorkflowDefinition


class ValidationSeverity(Enum):
    """Validation message severity."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationMessage:
    """Single validation message."""

    severity: ValidationSeverity
    code: str  # Machine-readable code like "MISSING_ENTRYPOINT"
    message: str  # Human-readable message
    path: str = ""  # JSON path to the issue, e.g., "steps[0].config.llm_ref"
    suggestion: str = ""  # Optional fix suggestion

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        result = {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
        }
        if self.path:
            result["path"] = self.path
        if self.suggestion:
            result["suggestion"] = self.suggestion
        return result


@dataclass
class ValidationResult:
    """Validation result containing all messages."""

    messages: list[ValidationMessage] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if no errors."""
        return not any(m.severity == ValidationSeverity.ERROR for m in self.messages)

    @property
    def errors(self) -> list[ValidationMessage]:
        """Get only error messages."""
        return [m for m in self.messages if m.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationMessage]:
        """Get only warning messages."""
        return [m for m in self.messages if m.severity == ValidationSeverity.WARNING]

    def add_error(
        self,
        code: str,
        message: str,
        path: str = "",
        suggestion: str = "",
    ) -> None:
        """Add error message."""
        self.messages.append(
            ValidationMessage(
                severity=ValidationSeverity.ERROR,
                code=code,
                message=message,
                path=path,
                suggestion=suggestion,
            )
        )

    def add_warning(
        self,
        code: str,
        message: str,
        path: str = "",
        suggestion: str = "",
    ) -> None:
        """Add warning message."""
        self.messages.append(
            ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code=code,
                message=message,
                path=path,
                suggestion=suggestion,
            )
        )

    def add_info(
        self,
        code: str,
        message: str,
        path: str = "",
    ) -> None:
        """Add info message."""
        self.messages.append(
            ValidationMessage(
                severity=ValidationSeverity.INFO,
                code=code,
                message=message,
                path=path,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "messages": [m.to_dict() for m in self.messages],
        }

    def raise_if_invalid(self) -> None:
        """Raise WorkflowValidationError if not valid."""
        if not self.is_valid:
            from llmteam.engine.exceptions import WorkflowValidationError

            error_messages = [m.message for m in self.errors]
            raise WorkflowValidationError(
                errors=error_messages,
                result=self,
            )




class SegmentValidator:
    """
    Segment validator.

    Validates segment definitions against:
    - Structure rules (required fields, valid references)
    - Step catalog (valid types, config schemas)
    - Connection rules (port compatibility)
    """

    def __init__(self, catalog: Optional["StepCatalog"] = None):
        """
        Initialize validator.

        Args:
            catalog: Optional step catalog for type validation.
                    If not provided, type validation is skipped.
        """
        self._catalog = catalog

    def validate(
        self,
        segment: "SegmentDefinition",
        strict: bool = True,
    ) -> ValidationResult:
        """
        Validate segment definition.

        Args:
            segment: Segment to validate
            strict: If True, treats warnings as errors

        Returns:
            ValidationResult with all messages
        """
        result = ValidationResult()

        # Structure validation
        self._validate_structure(segment, result)

        # Step validation
        self._validate_steps(segment, result)

        # Edge validation
        self._validate_edges(segment, result)

        # Config validation (if catalog available)
        if self._catalog:
            self._validate_configs(segment, result)

        return result

    def validate_dict(
        self,
        data: dict[str, Any],
        strict: bool = True,
    ) -> ValidationResult:
        """
        Validate segment from dict before parsing.

        Useful for validating JSON before creating SegmentDefinition.
        """
        result = ValidationResult()

        # Check required top-level fields
        required_fields = ["segment_id", "name", "steps"]
        for field_name in required_fields:
            if field_name not in data:
                result.add_error(
                    code="MISSING_FIELD",
                    message=f"Missing required field: {field_name}",
                    path=field_name,
                )

        # Check entrypoint
        if "entrypoint" not in data:
            steps = data.get("steps", [])
            if steps:
                result.add_warning(
                    code="MISSING_ENTRYPOINT",
                    message="No entrypoint defined, first step will be used",
                    suggestion=f"Add 'entrypoint': '{steps[0].get('step_id', 'first_step')}'",
                )

        # Validate steps array
        steps = data.get("steps", [])
        if not isinstance(steps, list):
            result.add_error(
                code="INVALID_TYPE",
                message="'steps' must be an array",
                path="steps",
            )
            return result

        step_ids: set[str] = set()
        for i, step in enumerate(steps):
            path = f"steps[{i}]"

            if not isinstance(step, dict):
                result.add_error(
                    code="INVALID_TYPE",
                    message=f"Step must be an object",
                    path=path,
                )
                continue

            # Check step_id
            step_id = step.get("step_id")
            if not step_id:
                result.add_error(
                    code="MISSING_FIELD",
                    message="Step missing step_id",
                    path=f"{path}.step_id",
                )
                continue

            if step_id in step_ids:
                result.add_error(
                    code="DUPLICATE_ID",
                    message=f"Duplicate step_id: {step_id}",
                    path=f"{path}.step_id",
                )
            step_ids.add(step_id)

            # Check type
            step_type = step.get("type") or step.get("step_type")
            if not step_type:
                result.add_error(
                    code="MISSING_FIELD",
                    message=f"Step '{step_id}' missing type",
                    path=f"{path}.type",
                )

        # Validate edges
        edges = data.get("edges", [])
        if not isinstance(edges, list):
            result.add_error(
                code="INVALID_TYPE",
                message="'edges' must be an array",
                path="edges",
            )
        else:
            for i, edge in enumerate(edges):
                path = f"edges[{i}]"

                source = edge.get("from") or edge.get("source_step") or edge.get("from_step")
                target = edge.get("to") or edge.get("target_step") or edge.get("to_step")

                if not source:
                    result.add_error(
                        code="MISSING_FIELD",
                        message="Edge missing source step",
                        path=f"{path}.from",
                    )
                elif source not in step_ids:
                    result.add_error(
                        code="INVALID_REFERENCE",
                        message=f"Edge references unknown source step: {source}",
                        path=f"{path}.from",
                    )

                if not target:
                    result.add_error(
                        code="MISSING_FIELD",
                        message="Edge missing target step",
                        path=f"{path}.to",
                    )
                elif target not in step_ids:
                    result.add_error(
                        code="INVALID_REFERENCE",
                        message=f"Edge references unknown target step: {target}",
                        path=f"{path}.to",
                    )

        # Validate entrypoint reference
        entrypoint = data.get("entrypoint")
        if entrypoint and step_ids and entrypoint not in step_ids:
            result.add_error(
                code="INVALID_REFERENCE",
                message=f"Entrypoint references unknown step: {entrypoint}",
                path="entrypoint",
            )

        return result

    def _validate_structure(
        self,
        segment: "SegmentDefinition",
        result: ValidationResult,
    ) -> None:
        """Validate basic structure."""
        import re

        # ID format validation
        id_pattern = re.compile(r"^[a-z][a-z0-9_]*$")

        if not id_pattern.match(segment.segment_id):
            result.add_error(
                code="INVALID_ID_FORMAT",
                message=f"Invalid segment_id format: '{segment.segment_id}'",
                path="segment_id",
                suggestion="Use lowercase letters, numbers, underscores; start with letter",
            )

        # Must have at least one step
        if not segment.steps:
            result.add_error(
                code="EMPTY_SEGMENT",
                message="Segment must have at least one step",
                path="steps",
            )

    def _validate_steps(
        self,
        segment: "SegmentDefinition",
        result: ValidationResult,
    ) -> None:
        """Validate steps."""
        import re

        id_pattern = re.compile(r"^[a-z][a-z0-9_]*$")
        step_ids: set[str] = set()

        for i, step in enumerate(segment.steps):
            path = f"steps[{i}]"

            # Check for duplicates
            if step.step_id in step_ids:
                result.add_error(
                    code="DUPLICATE_ID",
                    message=f"Duplicate step_id: {step.step_id}",
                    path=f"{path}.step_id",
                )
            step_ids.add(step.step_id)

            # Validate ID format
            if not id_pattern.match(step.step_id):
                result.add_error(
                    code="INVALID_ID_FORMAT",
                    message=f"Invalid step_id format: '{step.step_id}'",
                    path=f"{path}.step_id",
                    suggestion="Use lowercase letters, numbers, underscores; start with letter",
                )

            # Check type exists in catalog
            if self._catalog and not self._catalog.has(step.type):
                result.add_error(
                    code="UNKNOWN_TYPE",
                    message=f"Unknown step type: {step.type}",
                    path=f"{path}.type",
                    suggestion=f"Available types: {', '.join(self._catalog.list_type_ids())}",
                )

        # Validate entrypoint exists
        if segment.entrypoint not in step_ids:
            result.add_error(
                code="INVALID_ENTRYPOINT",
                message=f"Entrypoint '{segment.entrypoint}' not found in steps",
                path="entrypoint",
            )

    def _validate_edges(
        self,
        segment: "SegmentDefinition",
        result: ValidationResult,
    ) -> None:
        """Validate edges."""
        step_ids = {s.step_id for s in segment.steps}
        step_map = {s.step_id: s for s in segment.steps}

        for i, edge in enumerate(segment.edges):
            path = f"edges[{i}]"

            # Check source step exists
            if edge.from_step not in step_ids:
                result.add_error(
                    code="INVALID_REFERENCE",
                    message=f"Edge source '{edge.from_step}' not found",
                    path=f"{path}.from",
                )
            else:
                # Check source port exists
                source_step = step_map[edge.from_step]
                if edge.from_port not in source_step.output_ports:
                    result.add_warning(
                        code="UNKNOWN_PORT",
                        message=f"Port '{edge.from_port}' not in step '{edge.from_step}' outputs",
                        path=f"{path}.from_port",
                    )

            # Check target step exists
            if edge.to_step not in step_ids:
                result.add_error(
                    code="INVALID_REFERENCE",
                    message=f"Edge target '{edge.to_step}' not found",
                    path=f"{path}.to",
                )
            else:
                # Check target port exists
                target_step = step_map[edge.to_step]
                if edge.to_port not in target_step.input_ports:
                    result.add_warning(
                        code="UNKNOWN_PORT",
                        message=f"Port '{edge.to_port}' not in step '{edge.to_step}' inputs",
                        path=f"{path}.to_port",
                    )

            # Check for self-loops
            if edge.from_step == edge.to_step:
                result.add_warning(
                    code="SELF_LOOP",
                    message=f"Edge creates self-loop on step '{edge.from_step}'",
                    path=path,
                )

        # Check for unreachable steps (no incoming edges except entrypoint)
        steps_with_incoming = {e.to_step for e in segment.edges}
        steps_with_incoming.add(segment.entrypoint)

        for step in segment.steps:
            if step.step_id not in steps_with_incoming:
                result.add_warning(
                    code="UNREACHABLE_STEP",
                    message=f"Step '{step.step_id}' has no incoming edges and is not entrypoint",
                    path=f"steps",
                )

    def _validate_configs(
        self,
        segment: "SegmentDefinition",
        result: ValidationResult,
    ) -> None:
        """Validate step configs against schemas."""
        if not self._catalog:
            return

        for i, step in enumerate(segment.steps):
            path = f"steps[{i}].config"

            metadata = self._catalog.get(step.type)
            if not metadata:
                continue  # Unknown type already reported

            schema = metadata.config_schema
            if not schema:
                continue

            # Validate config against schema
            errors = self._validate_against_schema(step.config, schema, path)
            for error in errors:
                result.add_error(**error)

    def _validate_against_schema(
        self,
        config: dict[str, Any],
        schema: dict[str, Any],
        path: str,
    ) -> list[dict[str, Any]]:
        """
        Validate config against JSON Schema.

        Uses jsonschema if available, falls back to basic validation.
        """
        errors: list[dict[str, Any]] = []

        try:
            from jsonschema import Draft7Validator, ValidationError as JsonSchemaError

            validator = Draft7Validator(schema)
            for error in validator.iter_errors(config):
                error_path = ".".join(str(p) for p in error.absolute_path)
                full_path = f"{path}.{error_path}" if error_path else path

                errors.append({
                    "code": "SCHEMA_VIOLATION",
                    "message": error.message,
                    "path": full_path,
                })

        except ImportError:
            # Fallback to basic validation
            errors.extend(self._basic_schema_validation(config, schema, path))

        return errors

    def _basic_schema_validation(
        self,
        config: dict[str, Any],
        schema: dict[str, Any],
        path: str,
    ) -> list[dict[str, Any]]:
        """Basic schema validation without jsonschema library."""
        errors: list[dict[str, Any]] = []

        # Check required fields
        required = schema.get("required", [])
        for field_name in required:
            if field_name not in config:
                errors.append({
                    "code": "MISSING_REQUIRED",
                    "message": f"Missing required field: {field_name}",
                    "path": f"{path}.{field_name}",
                })

        # Check property types
        properties = schema.get("properties", {})
        for prop_name, prop_schema in properties.items():
            if prop_name not in config:
                continue

            value = config[prop_name]
            expected_type = prop_schema.get("type")

            if expected_type and not self._check_type(value, expected_type):
                errors.append({
                    "code": "INVALID_TYPE",
                    "message": f"Expected {expected_type}, got {type(value).__name__}",
                    "path": f"{path}.{prop_name}",
                })

            # Check enum
            enum_values = prop_schema.get("enum")
            if enum_values and value not in enum_values:
                errors.append({
                    "code": "INVALID_VALUE",
                    "message": f"Value must be one of: {enum_values}",
                    "path": f"{path}.{prop_name}",
                })

            # Check minimum/maximum for numbers
            if expected_type in ("number", "integer"):
                if "minimum" in prop_schema and value < prop_schema["minimum"]:
                    errors.append({
                        "code": "VALUE_TOO_SMALL",
                        "message": f"Value must be >= {prop_schema['minimum']}",
                        "path": f"{path}.{prop_name}",
                    })
                if "maximum" in prop_schema and value > prop_schema["maximum"]:
                    errors.append({
                        "code": "VALUE_TOO_LARGE",
                        "message": f"Value must be <= {prop_schema['maximum']}",
                        "path": f"{path}.{prop_name}",
                    })

        return errors

    def _check_type(self, value: Any, expected: str) -> bool:
        """Check if value matches expected JSON Schema type."""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }
        expected_types = type_map.get(expected)
        if expected_types is None:
            return True  # Unknown type, skip check
        return isinstance(value, expected_types)


def validate_segment(
    segment: "SegmentDefinition",
    catalog: Optional["StepCatalog"] = None,
) -> ValidationResult:
    """
    Convenience function to validate a segment.

    Args:
        segment: Segment to validate
        catalog: Optional step catalog

    Returns:
        ValidationResult
    """
    validator = SegmentValidator(catalog)
    return validator.validate(segment)


def validate_segment_dict(
    data: dict[str, Any],
    catalog: Optional["StepCatalog"] = None,
) -> ValidationResult:
    """
    Convenience function to validate segment from dict.

    Args:
        data: Segment data as dict
        catalog: Optional step catalog

    Returns:
        ValidationResult
    """
    validator = SegmentValidator(catalog)
    return validator.validate_dict(data)
