"""Tests for TeamContract."""

import pytest

from llmteam.roles.contract import (
    TeamContract,
    ValidationResult,
    ContractValidationError,
)
from llmteam.ports import TypedPort, PortLevel, PortDirection


class TestTeamContract:
    """Tests for TeamContract."""

    def test_create_contract(self):
        """Test creating a basic contract."""
        contract = TeamContract(
            name="test_contract",
            inputs=[
                TypedPort(
                    name="query",
                    level=PortLevel.WORKFLOW,
                    direction=PortDirection.INPUT,
                    data_type="string",
                    required=True,
                ),
            ],
            outputs=[
                TypedPort(
                    name="result",
                    level=PortLevel.WORKFLOW,
                    direction=PortDirection.OUTPUT,
                    data_type="object",
                    required=True,
                ),
            ],
        )

        assert contract.name == "test_contract"
        assert len(contract.inputs) == 1
        assert len(contract.outputs) == 1
        assert contract.inputs[0].name == "query"
        assert contract.outputs[0].name == "result"

    def test_default_contract(self):
        """Test default contract creation."""
        contract = TeamContract.default()

        assert contract.name == "default"
        assert len(contract.inputs) == 1
        assert len(contract.outputs) == 1
        assert contract.strict is False

    def test_validate_input_success(self):
        """Test successful input validation."""
        contract = TeamContract(
            name="test",
            inputs=[
                TypedPort(
                    name="text",
                    level=PortLevel.WORKFLOW,
                    direction=PortDirection.INPUT,
                    data_type="string",
                    required=True,
                ),
                TypedPort(
                    name="count",
                    level=PortLevel.WORKFLOW,
                    direction=PortDirection.INPUT,
                    data_type="number",
                    required=False,
                ),
            ],
        )

        result = contract.validate_input({"text": "hello", "count": 5})

        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_input_missing_required(self):
        """Test validation fails for missing required field."""
        contract = TeamContract(
            name="test",
            inputs=[
                TypedPort(
                    name="text",
                    level=PortLevel.WORKFLOW,
                    direction=PortDirection.INPUT,
                    data_type="string",
                    required=True,
                ),
            ],
        )

        result = contract.validate_input({})

        assert result.valid is False
        assert any("Missing required input: 'text'" in e for e in result.errors)

    def test_validate_input_type_mismatch(self):
        """Test validation fails for type mismatch."""
        contract = TeamContract(
            name="test",
            inputs=[
                TypedPort(
                    name="count",
                    level=PortLevel.WORKFLOW,
                    direction=PortDirection.INPUT,
                    data_type="number",
                    required=True,
                ),
            ],
        )

        result = contract.validate_input({"count": "not_a_number"})

        assert result.valid is False
        assert any("Type mismatch for 'count'" in e for e in result.errors)

    def test_validate_input_strict_mode(self):
        """Test strict mode rejects unknown fields."""
        contract = TeamContract(
            name="test",
            inputs=[
                TypedPort(
                    name="text",
                    level=PortLevel.WORKFLOW,
                    direction=PortDirection.INPUT,
                    data_type="string",
                    required=True,
                ),
            ],
            strict=True,
        )

        result = contract.validate_input({"text": "hello", "extra_field": "value"})

        assert result.valid is False
        assert any("Unknown input field: 'extra_field'" in e for e in result.errors)

    def test_validate_output_success(self):
        """Test successful output validation."""
        contract = TeamContract(
            name="test",
            outputs=[
                TypedPort(
                    name="result",
                    level=PortLevel.WORKFLOW,
                    direction=PortDirection.OUTPUT,
                    data_type="object",
                    required=True,
                ),
            ],
        )

        result = contract.validate_output({"result": {"key": "value"}})

        assert result.valid is True

    def test_validate_output_missing_required(self):
        """Test output validation fails for missing required field."""
        contract = TeamContract(
            name="test",
            outputs=[
                TypedPort(
                    name="result",
                    level=PortLevel.WORKFLOW,
                    direction=PortDirection.OUTPUT,
                    data_type="string",
                    required=True,
                ),
            ],
        )

        result = contract.validate_output({})

        assert result.valid is False
        assert any("Missing required output: 'result'" in e for e in result.errors)

    def test_to_dict(self):
        """Test contract serialization."""
        contract = TeamContract(
            name="test",
            inputs=[
                TypedPort(
                    name="query",
                    level=PortLevel.WORKFLOW,
                    direction=PortDirection.INPUT,
                    data_type="string",
                    required=True,
                ),
            ],
            outputs=[
                TypedPort(
                    name="result",
                    level=PortLevel.WORKFLOW,
                    direction=PortDirection.OUTPUT,
                    data_type="object",
                    required=True,
                ),
            ],
            version="2.0",
        )

        data = contract.to_dict()

        assert data["name"] == "test"
        assert data["version"] == "2.0"
        assert len(data["inputs"]) == 1
        assert len(data["outputs"]) == 1

    def test_from_dict(self):
        """Test contract deserialization."""
        data = {
            "name": "loaded_contract",
            "inputs": [
                {
                    "name": "input",
                    "level": "workflow",
                    "direction": "input",
                    "data_type": "string",
                    "required": True,
                    "description": "",
                    "schema": None,
                }
            ],
            "outputs": [],
            "strict": True,
            "version": "1.5",
            "description": "Test contract",
        }

        contract = TeamContract.from_dict(data)

        assert contract.name == "loaded_contract"
        assert contract.strict is True
        assert contract.version == "1.5"
        assert len(contract.inputs) == 1


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_bool_valid(self):
        """Test ValidationResult bool for valid."""
        result = ValidationResult(valid=True)
        assert bool(result) is True

    def test_bool_invalid(self):
        """Test ValidationResult bool for invalid."""
        result = ValidationResult(valid=False, errors=["Error 1"])
        assert bool(result) is False
