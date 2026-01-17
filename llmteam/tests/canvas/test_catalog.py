"""
Tests for Step Catalog API.
"""

import pytest

from llmteam.canvas import (
    StepCategory,
    PortSpec,
    StepTypeMetadata,
    StepCatalog,
)


# === Tests for PortSpec ===


class TestPortSpec:
    """Tests for PortSpec model."""

    def test_create_minimal(self) -> None:
        port = PortSpec("input")

        assert port.name == "input"
        assert port.type == "any"
        assert port.description == ""
        assert port.required is True
        assert port.default is None

    def test_create_full(self) -> None:
        port = PortSpec(
            name="data",
            type="object",
            description="Input data",
            required=False,
            default={"key": "value"},
            schema={"type": "object", "properties": {}},
        )

        assert port.name == "data"
        assert port.type == "object"
        assert port.description == "Input data"
        assert port.required is False
        assert port.default == {"key": "value"}

    def test_to_dict(self) -> None:
        port = PortSpec(
            name="output",
            type="string",
            description="Output text",
            default="default_value",
        )

        data = port.to_dict()

        assert data["name"] == "output"
        assert data["type"] == "string"
        assert data["description"] == "Output text"
        assert data["default"] == "default_value"


# === Tests for StepTypeMetadata ===


class TestStepTypeMetadata:
    """Tests for StepTypeMetadata model."""

    def _create_metadata(self) -> StepTypeMetadata:
        return StepTypeMetadata(
            type_id="test_type",
            version="1.0",
            display_name="Test Type",
            description="A test step type",
            category=StepCategory.UTILITY,
            icon="test-icon",
            color="#123456",
            config_schema={
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
            input_ports=[PortSpec("input", "object", "Input data")],
            output_ports=[PortSpec("output", "object", "Output data")],
        )

    def test_create(self) -> None:
        metadata = self._create_metadata()

        assert metadata.type_id == "test_type"
        assert metadata.version == "1.0"
        assert metadata.display_name == "Test Type"
        assert metadata.category == StepCategory.UTILITY

    def test_defaults(self) -> None:
        metadata = StepTypeMetadata(
            type_id="minimal",
            version="1.0",
            display_name="Minimal",
            description="Minimal type",
            category=StepCategory.DATA,
        )

        assert metadata.icon == "box"
        assert metadata.color == "#4A90D9"
        assert metadata.supports_retry is True
        assert metadata.supports_timeout is True
        assert metadata.supports_parallel is False
        assert metadata.is_async is True

    def test_to_dict(self) -> None:
        metadata = self._create_metadata()
        data = metadata.to_dict()

        assert data["type_id"] == "test_type"
        assert data["version"] == "1.0"
        assert data["display_name"] == "Test Type"
        assert data["category"] == "utility"
        assert data["icon"] == "test-icon"
        assert data["color"] == "#123456"
        assert len(data["input_ports"]) == 1
        assert len(data["output_ports"]) == 1
        assert data["supports_retry"] is True


# === Tests for StepCatalog ===


class TestStepCatalog:
    """Tests for StepCatalog."""

    @pytest.fixture(autouse=True)
    def reset_catalog(self) -> None:
        """Reset catalog singleton before each test."""
        StepCatalog.reset_instance()

    def test_singleton(self) -> None:
        catalog1 = StepCatalog.instance()
        catalog2 = StepCatalog.instance()

        assert catalog1 is catalog2

    def test_builtin_types_registered(self) -> None:
        catalog = StepCatalog.instance()

        # Check built-in types are registered
        assert catalog.has("llm_agent")
        assert catalog.has("http_action")
        assert catalog.has("human_task")
        assert catalog.has("condition")
        assert catalog.has("parallel_split")
        assert catalog.has("parallel_join")
        assert catalog.has("transform")

    def test_get_builtin_type(self) -> None:
        catalog = StepCatalog.instance()

        llm_agent = catalog.get("llm_agent")

        assert llm_agent is not None
        assert llm_agent.type_id == "llm_agent"
        assert llm_agent.display_name == "LLM Agent"
        assert llm_agent.category == StepCategory.AI

    def test_get_nonexistent(self) -> None:
        catalog = StepCatalog.instance()

        result = catalog.get("nonexistent")

        assert result is None

    def test_register_custom_type(self) -> None:
        catalog = StepCatalog.instance()

        custom_type = StepTypeMetadata(
            type_id="custom_step",
            version="1.0",
            display_name="Custom Step",
            description="A custom step type",
            category=StepCategory.UTILITY,
        )

        catalog.register(custom_type)

        assert catalog.has("custom_step")
        assert catalog.get("custom_step") is custom_type

    def test_register_with_handler(self) -> None:
        catalog = StepCatalog.instance()

        def my_handler(input_data):
            return {"result": "processed"}

        custom_type = StepTypeMetadata(
            type_id="handler_step",
            version="1.0",
            display_name="Handler Step",
            description="Step with handler",
            category=StepCategory.UTILITY,
        )

        catalog.register(custom_type, handler=my_handler)

        assert catalog.get_handler("handler_step") is my_handler

    def test_unregister(self) -> None:
        catalog = StepCatalog.instance()

        custom_type = StepTypeMetadata(
            type_id="to_remove",
            version="1.0",
            display_name="To Remove",
            description="Will be removed",
            category=StepCategory.UTILITY,
        )

        catalog.register(custom_type)
        assert catalog.has("to_remove")

        catalog.unregister("to_remove")
        assert not catalog.has("to_remove")

    def test_list_all(self) -> None:
        catalog = StepCatalog.instance()

        all_types = catalog.list_all()

        # Should have at least the built-in types
        assert len(all_types) >= 7
        type_ids = [t.type_id for t in all_types]
        assert "llm_agent" in type_ids
        assert "http_action" in type_ids

    def test_list_by_category(self) -> None:
        catalog = StepCatalog.instance()

        ai_types = catalog.list_by_category(StepCategory.AI)
        control_types = catalog.list_by_category(StepCategory.CONTROL)
        human_types = catalog.list_by_category(StepCategory.HUMAN)

        assert len(ai_types) >= 1
        assert any(t.type_id == "llm_agent" for t in ai_types)

        assert len(control_types) >= 3
        assert any(t.type_id == "condition" for t in control_types)

        assert len(human_types) >= 1
        assert any(t.type_id == "human_task" for t in human_types)

    def test_list_type_ids(self) -> None:
        catalog = StepCatalog.instance()

        type_ids = catalog.list_type_ids()

        assert "llm_agent" in type_ids
        assert "http_action" in type_ids
        assert "human_task" in type_ids

    def test_export_for_ui(self) -> None:
        catalog = StepCatalog.instance()

        export = catalog.export_for_ui()

        assert "version" in export
        assert "categories" in export
        assert "types" in export

        # Check categories
        assert "ai" in export["categories"]
        assert "human" in export["categories"]

        # Check types dict structure
        assert "llm_agent" in export["types"]
        assert export["types"]["llm_agent"]["display_name"] == "LLM Agent"

    def test_validate_step_config_valid(self) -> None:
        catalog = StepCatalog.instance()

        config = {"llm_ref": "gpt4", "temperature": 0.7}
        errors = catalog.validate_step_config("llm_agent", config)

        assert errors == []

    def test_validate_step_config_missing_required(self) -> None:
        catalog = StepCatalog.instance()

        config = {"temperature": 0.7}  # Missing required 'llm_ref'
        errors = catalog.validate_step_config("llm_agent", config)

        assert len(errors) == 1
        assert "llm_ref" in errors[0]

    def test_validate_step_config_unknown_type(self) -> None:
        catalog = StepCatalog.instance()

        errors = catalog.validate_step_config("unknown_type", {})

        assert len(errors) == 1
        assert "Unknown step type" in errors[0]


# === Tests for Built-in Step Types ===


class TestBuiltinStepTypes:
    """Tests for built-in step type configurations."""

    @pytest.fixture(autouse=True)
    def reset_catalog(self) -> None:
        StepCatalog.reset_instance()

    def test_llm_agent_config_schema(self) -> None:
        catalog = StepCatalog.instance()
        llm_agent = catalog.get("llm_agent")

        schema = llm_agent.config_schema
        assert schema["type"] == "object"
        assert "llm_ref" in schema["properties"]
        assert "temperature" in schema["properties"]
        assert "llm_ref" in schema["required"]

    def test_llm_agent_ports(self) -> None:
        catalog = StepCatalog.instance()
        llm_agent = catalog.get("llm_agent")

        assert len(llm_agent.input_ports) == 1
        assert llm_agent.input_ports[0].name == "input"

        assert len(llm_agent.output_ports) == 2
        output_names = [p.name for p in llm_agent.output_ports]
        assert "output" in output_names
        assert "error" in output_names

    def test_http_action_config_schema(self) -> None:
        catalog = StepCatalog.instance()
        http_action = catalog.get("http_action")

        schema = http_action.config_schema
        assert "client_ref" in schema["properties"]
        assert "method" in schema["properties"]
        assert "path" in schema["properties"]
        assert "client_ref" in schema["required"]
        assert "path" in schema["required"]

    def test_human_task_config_schema(self) -> None:
        catalog = StepCatalog.instance()
        human_task = catalog.get("human_task")

        schema = human_task.config_schema
        assert "task_type" in schema["properties"]
        assert "assignee_ref" in schema["properties"]
        assert "timeout_hours" in schema["properties"]

    def test_human_task_no_parallel(self) -> None:
        catalog = StepCatalog.instance()
        human_task = catalog.get("human_task")

        assert human_task.supports_parallel is False

    def test_condition_ports(self) -> None:
        catalog = StepCatalog.instance()
        condition = catalog.get("condition")

        assert len(condition.output_ports) == 2
        output_names = [p.name for p in condition.output_ports]
        assert "true" in output_names
        assert "false" in output_names

    def test_parallel_split_parallel_support(self) -> None:
        catalog = StepCatalog.instance()
        parallel_split = catalog.get("parallel_split")

        assert parallel_split.supports_parallel is True

    def test_step_categories(self) -> None:
        catalog = StepCatalog.instance()

        assert catalog.get("llm_agent").category == StepCategory.AI
        assert catalog.get("http_action").category == StepCategory.INTEGRATION
        assert catalog.get("human_task").category == StepCategory.HUMAN
        assert catalog.get("condition").category == StepCategory.CONTROL
        assert catalog.get("transform").category == StepCategory.DATA
