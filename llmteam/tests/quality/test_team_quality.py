"""
Tests for LLMTeam after RFC-021 Quality Simplification.

RFC-021: Quality is removed from LLMTeam.
Quality is now a design-time parameter in ConfigurationSession only.
"""

import pytest
from llmteam import LLMTeam


class TestLLMTeamNoQuality:
    """Tests that LLMTeam no longer has quality parameter. RFC-021."""

    def test_no_quality_parameter(self):
        """LLMTeam should not accept quality parameter. RFC-021."""
        team = LLMTeam(team_id="test")

        # quality should not be an attribute
        assert not hasattr(team, "quality") or not callable(getattr(team, "quality", None))

    def test_no_quality_manager(self):
        """LLMTeam should not have _quality_manager. RFC-021."""
        team = LLMTeam(team_id="test")

        assert not hasattr(team, "_quality_manager")

    def test_no_get_quality_manager(self):
        """LLMTeam should not have get_quality_manager method. RFC-021."""
        team = LLMTeam(team_id="test")

        assert not hasattr(team, "get_quality_manager")

    def test_max_cost_per_run(self):
        """max_cost_per_run should still work. RFC-021."""
        team = LLMTeam(
            team_id="test",
            max_cost_per_run=1.00,
        )

        assert team._max_cost_per_run == 1.00
        config = team.to_config()
        assert config["max_cost_per_run"] == 1.00

    def test_to_config_no_quality(self):
        """to_config() should not include quality. RFC-021."""
        team = LLMTeam(team_id="test")
        config = team.to_config()

        assert "quality" not in config

    def test_from_config_no_quality(self):
        """from_config() should work without quality. RFC-021."""
        config = {
            "team_id": "test",
            "max_cost_per_run": 2.0,
        }
        team = LLMTeam.from_config(config)

        assert team.team_id == "test"
        assert team._max_cost_per_run == 2.0


class TestLLMTeamEstimateCost:
    """Tests for LLMTeam.estimate_cost() after RFC-021."""

    @pytest.mark.asyncio
    async def test_estimate_cost_without_agents(self):
        """Estimate cost without agents uses default model. RFC-021."""
        team = LLMTeam(team_id="test")

        estimate = await team.estimate_cost(complexity="medium")

        assert estimate.min_cost > 0
        assert estimate.max_cost >= estimate.min_cost

    @pytest.mark.asyncio
    async def test_estimate_cost_with_agents(self):
        """Estimate cost with agents. RFC-021."""
        team = LLMTeam(
            team_id="test",
            agents=[
                {"type": "llm", "role": "writer", "prompt": "Write", "model": "gpt-4o"},
                {"type": "llm", "role": "editor", "prompt": "Edit", "model": "gpt-4o-mini"},
            ],
        )

        estimate = await team.estimate_cost()

        assert estimate.min_cost > 0
        assert estimate.max_cost >= estimate.min_cost


class TestLLMTeamRunSignature:
    """Tests for LLMTeam.run() signature after RFC-021."""

    def test_run_no_quality_parameter(self):
        """run() should not have quality parameter. RFC-021."""
        team = LLMTeam(team_id="test")

        import inspect
        sig = inspect.signature(team.run)

        assert "quality" not in sig.parameters
        assert "importance" not in sig.parameters
        assert "input_data" in sig.parameters

    def test_stream_no_quality_parameter(self):
        """stream() should not have quality parameter. RFC-021."""
        team = LLMTeam(team_id="test")

        import inspect
        sig = inspect.signature(team.stream)

        assert "quality" not in sig.parameters
        assert "importance" not in sig.parameters
        assert "input_data" in sig.parameters
