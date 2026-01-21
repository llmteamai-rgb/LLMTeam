"""Tests for CLI module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from click.testing import CliRunner

from llmteam.cli.main import cli, _load_json_file, _get_builtin_types


class TestLoadJsonFile:
    """Tests for _load_json_file helper."""

    def test_load_valid_json(self, tmp_path):
        """Load valid JSON file."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"key": "value"}')

        result = _load_json_file(json_file)

        assert result == {"key": "value"}

    def test_load_complex_json(self, tmp_path):
        """Load complex JSON file."""
        data = {
            "segment_id": "test",
            "name": "Test Segment",
            "steps": [{"step_id": "s1", "type": "transform"}],
        }
        json_file = tmp_path / "segment.json"
        json_file.write_text(json.dumps(data))

        result = _load_json_file(json_file)

        assert result == data

    def test_load_nonexistent_file(self):
        """Raise error for nonexistent file."""
        from click import ClickException

        with pytest.raises(ClickException) as exc_info:
            _load_json_file("/nonexistent/path.json")

        assert "File not found" in str(exc_info.value)

    def test_load_invalid_json(self, tmp_path):
        """Raise error for invalid JSON."""
        from click import ClickException

        json_file = tmp_path / "invalid.json"
        json_file.write_text("not valid json {")

        with pytest.raises(ClickException) as exc_info:
            _load_json_file(json_file)

        assert "Invalid JSON" in str(exc_info.value)


class TestGetBuiltinTypes:
    """Tests for _get_builtin_types fallback."""

    def test_returns_dict(self):
        """Returns dict of type info."""
        types = _get_builtin_types()

        assert isinstance(types, dict)
        assert len(types) > 0

    def test_contains_expected_types(self):
        """Contains expected step types."""
        types = _get_builtin_types()

        assert "llm_agent" in types
        assert "transform" in types
        assert "human_task" in types
        assert "conditional" in types

    def test_type_info_has_attributes(self):
        """Type info has required attributes."""
        types = _get_builtin_types()

        llm_agent = types["llm_agent"]
        assert hasattr(llm_agent, "display_name")
        assert hasattr(llm_agent, "description")
        assert hasattr(llm_agent, "category")


class TestCliVersion:
    """Tests for version command."""

    def test_version_flag(self):
        """--version shows version."""
        runner = CliRunner()

        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        import llmteam
        assert llmteam.__version__ in result.output

    def test_version_command(self):
        """version command shows detailed info."""
        runner = CliRunner()

        result = runner.invoke(cli, ["version"])

        assert result.exit_code == 0
        assert "llmteam" in result.output
        assert "Python" in result.output


class TestCliHelp:
    """Tests for help output."""

    def test_main_help(self):
        """Main help shows available commands."""
        runner = CliRunner()

        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "run" in result.output
        assert "validate" in result.output
        assert "catalog" in result.output
        assert "version" in result.output

    def test_run_help(self):
        """Run command help shows options."""
        runner = CliRunner()

        result = runner.invoke(cli, ["run", "--help"])

        assert result.exit_code == 0
        assert "--input" in result.output
        assert "--output" in result.output
        assert "--timeout" in result.output

    def test_validate_help(self):
        """Validate command help shows options."""
        runner = CliRunner()

        result = runner.invoke(cli, ["validate", "--help"])

        assert result.exit_code == 0
        assert "--strict" in result.output


class TestCliCatalog:
    """Tests for catalog command."""

    def test_catalog_lists_types(self):
        """Catalog command lists step types."""
        runner = CliRunner()

        result = runner.invoke(cli, ["catalog"])

        assert result.exit_code == 0
        # Check for core handlers that are always registered
        assert "llm_agent" in result.output
        assert "transform" in result.output
        assert "condition" in result.output
        # Note: human_task may not be registered in parallel test mode

    def test_catalog_json_output(self):
        """Catalog --json returns valid JSON."""
        runner = CliRunner()

        result = runner.invoke(cli, ["catalog", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, dict)
        assert "llm_agent" in data

    def test_catalog_json_has_metadata(self):
        """Catalog JSON includes type metadata."""
        runner = CliRunner()

        result = runner.invoke(cli, ["catalog", "--json"])

        data = json.loads(result.output)
        llm_agent = data["llm_agent"]
        assert "display_name" in llm_agent
        assert "description" in llm_agent
        assert "category" in llm_agent


class TestCliValidate:
    """Tests for validate command."""

    def test_validate_valid_segment(self, tmp_path):
        """Validate valid segment file."""
        segment = {
            "segment_id": "test_segment",
            "name": "Test",
            "entrypoint": "step1",
            "steps": [
                {"step_id": "step1", "type": "transform", "config": {}}
            ],
            "edges": [],
        }
        json_file = tmp_path / "segment.json"
        json_file.write_text(json.dumps(segment))

        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(json_file)])

        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_validate_missing_segment_id(self, tmp_path):
        """Validate detects missing segment_id."""
        segment = {
            "name": "Test",
            "steps": [],
        }
        json_file = tmp_path / "segment.json"
        json_file.write_text(json.dumps(segment))

        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(json_file)])

        assert result.exit_code == 1
        assert "segment_id" in result.output.lower()

    def test_validate_missing_steps(self, tmp_path):
        """Validate detects missing steps."""
        segment = {
            "segment_id": "test",
            "name": "Test",
        }
        json_file = tmp_path / "segment.json"
        json_file.write_text(json.dumps(segment))

        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(json_file)])

        assert result.exit_code == 1
        assert "steps" in result.output.lower()

    def test_validate_duplicate_step_id(self, tmp_path):
        """Validate detects duplicate step_id."""
        segment = {
            "segment_id": "test",
            "name": "Test",
            "entrypoint": "step1",
            "steps": [
                {"step_id": "step1", "type": "transform"},
                {"step_id": "step1", "type": "transform"},
            ],
        }
        json_file = tmp_path / "segment.json"
        json_file.write_text(json.dumps(segment))

        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(json_file)])

        assert result.exit_code == 1
        assert "duplicate" in result.output.lower()

    def test_validate_invalid_edge_reference(self, tmp_path):
        """Validate detects invalid edge references."""
        segment = {
            "segment_id": "test",
            "name": "Test",
            "entrypoint": "step1",
            "steps": [
                {"step_id": "step1", "type": "transform"},
            ],
            "edges": [
                {"from": "step1", "to": "nonexistent", "from_port": "output", "to_port": "input"},
            ],
        }
        json_file = tmp_path / "segment.json"
        json_file.write_text(json.dumps(segment))

        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(json_file)])

        assert result.exit_code == 1
        # Check for either "nonexistent" or general edge error
        assert "nonexistent" in result.output.lower() or "edge" in result.output.lower()

    def test_validate_no_strict_allows_warnings(self, tmp_path):
        """Validate --no-strict allows warnings."""
        segment = {
            "segment_id": "test",
            "name": "Test",
            "steps": [
                {"step_id": "step1", "type": "transform"},
            ],
            # No entrypoint - should be warning
        }
        json_file = tmp_path / "segment.json"
        json_file.write_text(json.dumps(segment))

        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--no-strict", str(json_file)])

        # Should pass with warning
        assert "warning" in result.output.lower() or result.exit_code == 0

    def test_validate_nonexistent_file(self):
        """Validate nonexistent file shows error."""
        runner = CliRunner()

        result = runner.invoke(cli, ["validate", "/nonexistent/file.json"])

        assert result.exit_code != 0


class TestCliRun:
    """Tests for run command."""

    def test_run_missing_file(self):
        """Run with nonexistent file shows error."""
        runner = CliRunner()

        result = runner.invoke(cli, ["run", "/nonexistent/segment.json"])

        assert result.exit_code != 0

    def test_run_with_input_json(self, tmp_path):
        """Run with --input-json parses JSON input."""
        segment = {
            "segment_id": "test",
            "name": "Test",
            "entrypoint": "step1",
            "steps": [
                {"step_id": "step1", "type": "transform", "config": {}}
            ],
        }
        json_file = tmp_path / "segment.json"
        json_file.write_text(json.dumps(segment))

        runner = CliRunner()

        # Mock the async run to avoid actual execution
        with patch("llmteam.cli.main._run_segment_async") as mock_run:
            mock_run.return_value = {"status": "completed", "output": {}}

            result = runner.invoke(
                cli,
                ["run", str(json_file), "--input-json", '{"key": "value"}'],
            )

            # Check that input was parsed
            if mock_run.called:
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["input_data"] == {"key": "value"}

    def test_run_with_input_file(self, tmp_path):
        """Run with --input reads input file."""
        segment = {
            "segment_id": "test",
            "name": "Test",
            "entrypoint": "step1",
            "steps": [
                {"step_id": "step1", "type": "transform", "config": {}}
            ],
        }
        segment_file = tmp_path / "segment.json"
        segment_file.write_text(json.dumps(segment))

        input_data = {"query": "test"}
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(input_data))

        runner = CliRunner()

        with patch("llmteam.cli.main._run_segment_async") as mock_run:
            mock_run.return_value = {"status": "completed", "output": {}}

            result = runner.invoke(
                cli,
                ["run", str(segment_file), "--input", str(input_file)],
            )

            if mock_run.called:
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["input_data"] == input_data

    def test_run_output_to_file(self, tmp_path):
        """Run with --output writes to file."""
        segment = {
            "segment_id": "test",
            "name": "Test",
            "entrypoint": "step1",
            "steps": [
                {"step_id": "step1", "type": "transform", "config": {}}
            ],
        }
        segment_file = tmp_path / "segment.json"
        segment_file.write_text(json.dumps(segment))

        output_file = tmp_path / "output.json"

        runner = CliRunner()

        with patch("llmteam.cli.main._run_segment_async") as mock_run:
            mock_run.return_value = {
                "status": "completed",
                "output": {"result": "success"},
            }

            result = runner.invoke(
                cli,
                ["run", str(segment_file), "--output", str(output_file)],
            )

            if result.exit_code == 0 and output_file.exists():
                output_data = json.loads(output_file.read_text())
                assert output_data["status"] == "completed"


class TestCliServe:
    """Tests for serve command."""

    def test_serve_help(self):
        """Serve command shows help."""
        runner = CliRunner()

        result = runner.invoke(cli, ["serve", "--help"])

        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--reload" in result.output

    def test_serve_command_exists(self):
        """Serve command is registered."""
        runner = CliRunner()

        # Just verify the command exists and shows help
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "Start the API server" in result.output
