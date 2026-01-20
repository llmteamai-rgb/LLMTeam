"""Tests for TeamHandler (v3.0.0 API)."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from llmteam.canvas.handlers.team_handler import TeamHandler, TeamNotFoundError
from llmteam.team import TeamResult


class TestTeamHandler:
    """Tests for TeamHandler with v3.0.0 LLMTeam API."""

    @pytest.fixture
    def handler(self):
        """Create a TeamHandler instance."""
        return TeamHandler()

    @pytest.fixture
    def mock_team(self):
        """Create a mock team with v3.0.0 API."""
        team = MagicMock()
        team.run = AsyncMock(return_value=TeamResult(
            output={"result": "success", "score": 0.95},
            success=True,
            iterations=1,
            agents_called=["agent1"],
        ))
        return team

    @pytest.fixture
    def mock_runtime(self, mock_team):
        """Create a mock runtime with team registry."""
        runtime = MagicMock()
        runtime.get_team = MagicMock(return_value=mock_team)
        return runtime

    @pytest.fixture
    def mock_ctx(self, mock_runtime, mock_team):
        """Create a mock step context with v3.0.0 API."""
        ctx = MagicMock()
        ctx.step_id = "team_step_1"
        ctx.instance_id = "run_123"
        ctx.runtime = mock_runtime
        ctx.get_team = MagicMock(return_value=mock_team)  # v3.0.0 uses ctx.get_team()
        return ctx

    async def test_execute_team_basic(self, handler, mock_ctx, mock_team):
        """Test basic team execution."""
        config = {"team_ref": "analysis_team"}
        input_data = {"query": "analyze this"}

        result = await handler(mock_ctx, config, input_data)

        assert "output" in result
        assert result["output"]["result"] == "success"
        mock_team.run.assert_called_once()
        # v3.0.0 returns team_metadata
        assert "team_metadata" in result

    async def test_execute_team_with_input_mapping(self, handler, mock_ctx, mock_team):
        """Test team execution with input mapping."""
        config = {
            "team_ref": "analysis_team",
            "input_mapping": {
                "text": "input.query",
            },
        }
        input_data = {"input": {"query": "test query"}}

        result = await handler(mock_ctx, config, input_data)

        # Verify the mapped input was passed to team
        call_args = mock_team.run.call_args
        mapped_input = call_args[0][0]  # First positional arg in v3.0.0
        assert "text" in mapped_input
        assert mapped_input["text"] == "test query"

    async def test_execute_team_with_output_mapping(self, handler, mock_ctx, mock_team):
        """Test team execution with output mapping."""
        config = {
            "team_ref": "analysis_team",
            "output_mapping": {
                "analysis_result": "result",
                "confidence": "score",
            },
        }
        input_data = {"query": "test"}

        result = await handler(mock_ctx, config, input_data)

        assert "output" in result
        output = result["output"]
        assert "analysis_result" in output
        assert output["analysis_result"] == "success"
        assert "confidence" in output
        assert output["confidence"] == 0.95

    async def test_team_not_found(self, handler):
        """Test error when team is not found."""
        # Create a ctx without the team
        runtime = MagicMock()
        runtime.get_team = MagicMock(return_value=None)

        ctx = MagicMock()
        ctx.step_id = "team_step_1"
        ctx.instance_id = "run_123"
        ctx.runtime = runtime
        ctx.get_team = MagicMock(return_value=None)  # v3.0.0 uses ctx.get_team()

        config = {"team_ref": "nonexistent_team"}
        input_data = {"query": "test"}

        with pytest.raises(TeamNotFoundError) as exc_info:
            await handler(ctx, config, input_data)

        assert "nonexistent_team" in str(exc_info.value)

    async def test_missing_team_ref(self, handler, mock_ctx):
        """Test error when team_ref is missing."""
        config = {}
        input_data = {"query": "test"}

        with pytest.raises(ValueError) as exc_info:
            await handler(mock_ctx, config, input_data)

        assert "team_ref is required" in str(exc_info.value)

    async def test_team_execution_error(self, handler):
        """Test handling of team execution error."""
        # Create a mock team that raises an error
        failing_team = MagicMock()
        failing_team.run = AsyncMock(side_effect=RuntimeError("Team failed"))

        runtime = MagicMock()
        runtime.get_team = MagicMock(return_value=failing_team)

        ctx = MagicMock()
        ctx.step_id = "team_step_1"
        ctx.instance_id = "run_123"
        ctx.runtime = runtime
        ctx.get_team = MagicMock(return_value=failing_team)  # v3.0.0

        config = {"team_ref": "analysis_team"}
        input_data = {"query": "test"}

        with pytest.raises(RuntimeError) as exc_info:
            await handler(ctx, config, input_data)

        assert "Team failed" in str(exc_info.value)

    async def test_team_execution_failure_result(self, handler):
        """Test handling when team returns a failure result."""
        # Create a mock team that returns a failure
        failing_team = MagicMock()
        failing_team.run = AsyncMock(return_value=TeamResult(
            output={},
            success=False,
            iterations=1,
            agents_called=[],
            error="Validation failed",
        ))

        runtime = MagicMock()
        runtime.get_team = MagicMock(return_value=failing_team)

        ctx = MagicMock()
        ctx.step_id = "team_step_1"
        ctx.instance_id = "run_123"
        ctx.runtime = runtime
        ctx.get_team = MagicMock(return_value=failing_team)

        config = {"team_ref": "analysis_team"}
        input_data = {"query": "test"}

        with pytest.raises(RuntimeError) as exc_info:
            await handler(ctx, config, input_data)

        assert "Validation failed" in str(exc_info.value)


class TestTeamHandlerInputMapping:
    """Tests for input mapping functionality."""

    @pytest.fixture
    def handler(self):
        return TeamHandler()

    def test_apply_input_mapping_empty(self, handler):
        """Test input mapping with empty mapping."""
        input_data = {"key": "value"}
        result = handler._apply_input_mapping(input_data, {})
        assert result == input_data

    def test_apply_input_mapping_simple(self, handler):
        """Test simple input mapping."""
        input_data = {"original_key": "value"}
        mapping = {"new_key": "original_key"}

        result = handler._apply_input_mapping(input_data, mapping)

        assert "new_key" in result
        # Note: the mapping resolves through _get_nested_value

    def test_apply_input_mapping_nested(self, handler):
        """Test nested input mapping."""
        input_data = {"outer": {"inner": "nested_value"}}
        mapping = {"flat_key": "outer.inner"}

        result = handler._apply_input_mapping(input_data, mapping)

        assert result.get("flat_key") == "nested_value"


class TestTeamHandlerOutputMapping:
    """Tests for output mapping functionality."""

    @pytest.fixture
    def handler(self):
        return TeamHandler()

    def test_apply_output_mapping_empty(self, handler):
        """Test output mapping with empty mapping."""
        output_data = {"result": "success"}
        result = handler._apply_output_mapping(output_data, {})
        assert result == output_data

    def test_apply_output_mapping_rename(self, handler):
        """Test output mapping that renames fields."""
        output_data = {"result": "success", "score": 0.9}
        mapping = {"outcome": "result", "confidence": "score"}

        result = handler._apply_output_mapping(output_data, mapping)

        assert "outcome" in result
        assert result["outcome"] == "success"
        assert "confidence" in result
        assert result["confidence"] == 0.9


class TestTeamHandlerNestedValue:
    """Tests for nested value retrieval."""

    @pytest.fixture
    def handler(self):
        return TeamHandler()

    def test_get_nested_value_simple(self, handler):
        """Test getting simple key."""
        data = {"key": "value"}
        result = handler._get_nested_value(data, "key")
        assert result == "value"

    def test_get_nested_value_nested(self, handler):
        """Test getting nested key."""
        data = {"level1": {"level2": {"level3": "deep_value"}}}
        result = handler._get_nested_value(data, "level1.level2.level3")
        assert result == "deep_value"

    def test_get_nested_value_missing(self, handler):
        """Test getting missing key returns None."""
        data = {"key": "value"}
        result = handler._get_nested_value(data, "nonexistent")
        assert result is None

    def test_get_nested_value_partial_path(self, handler):
        """Test partial path returns None."""
        data = {"level1": {"level2": "value"}}
        result = handler._get_nested_value(data, "level1.nonexistent.level3")
        assert result is None
