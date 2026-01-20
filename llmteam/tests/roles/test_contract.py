import pytest
from llmteam.roles.contract import TeamContract, ContractValidationResult
from llmteam.ports.models import TypedPort, PortDirection, PortLevel

class TestTeamContract:

    def test_default_contract(self):
        """Test default contract creation."""
        contract = TeamContract.default()
        assert len(contract.inputs) == 1
        assert len(contract.outputs) == 1
        assert contract.inputs[0].name == "input"

    def test_validation_success(self):
        """Test successful validation."""
        contract = TeamContract(
            inputs=[
                TypedPort(name="ticket", level=PortLevel.WORKFLOW, direction=PortDirection.INPUT, required=True)
            ],
            outputs=[
                TypedPort(name="result", level=PortLevel.WORKFLOW, direction=PortDirection.OUTPUT, required=True)
            ]
        )

        # Valid input
        res = contract.validate_input({"ticket": "foo", "extra": "bar"})
        assert res.valid
        assert not res.errors

        # Valid output
        res = contract.validate_output({"result": "done"})
        assert res.valid

    def test_validation_failure(self):
        """Test validation failures."""
        contract = TeamContract(
            inputs=[
                TypedPort(name="ticket", level=PortLevel.WORKFLOW, direction=PortDirection.INPUT, required=True)
            ]
        )

        # Missing required input
        res = contract.validate_input({"other": "foo"})
        assert not res.valid
        assert "Missing required input: ticket" in res.errors[0]

    def test_serialization(self):
        """Test to_dict and from_dict."""
        contract = TeamContract(
            inputs=[
                TypedPort(name="in1", level=PortLevel.WORKFLOW, direction=PortDirection.INPUT)
            ],
            description="Test Contract"
        )
        
        data = contract.to_dict()
        assert data["description"] == "Test Contract"
        assert len(data["inputs"]) == 1
        
        restored = TeamContract.from_dict(data)
        assert restored.description == "Test Contract"
        assert restored.inputs[0].name == "in1"
