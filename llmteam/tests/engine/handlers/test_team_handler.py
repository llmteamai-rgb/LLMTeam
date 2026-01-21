import pytest
from unittest.mock import AsyncMock, MagicMock
from llmteam.engine.handlers.team_handler import TeamHandler
from llmteam.runtime import StepContext
from llmteam.team import TeamResult


class TestTeamHandler:

    @pytest.mark.asyncio
    async def test_team_execution(self):
        """Test successful team execution via handler."""
        # Mock Team with v4.0.0 API
        mock_team = MagicMock()
        mock_team.run = AsyncMock(return_value=TeamResult(
            output={"result": "success", "score": 0.95},
            success=True,
            iterations=1,
            agents_called=["agent1"],  # v4.0.0: renamed from agents_invoked
        ))

        # Mock Runtime to resolve team
        mock_runtime = MagicMock()
        mock_runtime.get_team.return_value = mock_team

        # Context - need to mock get_team method
        ctx = MagicMock(spec=StepContext)
        ctx.runtime = mock_runtime
        ctx.step_id = "step_1"
        ctx.get_team.return_value = mock_team  # v3.0.0 uses ctx.get_team()

        # Config & Input
        config = {
            "team_ref": "analyst_team",
            "input_mapping": {"query": "input.q"},
            "output_mapping": {"final_result": "result"}
        }
        input_data = {"input": {"q": "test query"}, "other": 123}

        # Handler
        handler = TeamHandler()
        result = await handler(ctx, config, input_data)

        # Verification
        ctx.get_team.assert_called_with("analyst_team")
        mock_team.run.assert_called_once()
        args, kwargs = mock_team.run.call_args
        # Handler preserves structure, so 'input' key remains even if 'input.q' was mapped
        expected_input = {
            "query": "test query",
            "other": 123,
            "input": {"q": "test query"}
        }
        assert args[0] == expected_input

        assert result["output"]["final_result"] == "success"
        # "score" is unmapped, so it should be included
        assert result["output"]["score"] == 0.95
        # v3.0.0 returns team_metadata
        assert "team_metadata" in result
        assert result["team_metadata"]["iterations"] == 1

    @pytest.mark.asyncio
    async def test_team_not_found(self):
        """Test error when team is not found."""
        mock_runtime = MagicMock()
        mock_runtime.get_team.return_value = None

        ctx = MagicMock(spec=StepContext)
        ctx.runtime = mock_runtime
        ctx.get_team.return_value = None  # v3.0.0 uses ctx.get_team()

        config = {"team_ref": "missing_team"}
        handler = TeamHandler()

        with pytest.raises(Exception) as exc:
            await handler(ctx, config, {})
        assert "not found" in str(exc.value)
