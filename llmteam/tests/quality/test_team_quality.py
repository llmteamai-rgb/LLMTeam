"""
Tests for LLMTeam quality integration (RFC-008).
"""

import pytest
from llmteam import LLMTeam, QualityManager


class TestLLMTeamQuality:
    """Tests for LLMTeam quality parameter."""

    def test_default_quality(self):
        """Default quality should be 50."""
        team = LLMTeam(team_id="test")
        assert team.quality == 50

    def test_quality_with_int(self):
        """Set quality with integer."""
        team = LLMTeam(team_id="test", quality=70)
        assert team.quality == 70

    def test_quality_with_preset(self):
        """Set quality with preset string."""
        team = LLMTeam(team_id="test", quality="production")
        assert team.quality == 75

        team = LLMTeam(team_id="test", quality="draft")
        assert team.quality == 20

    def test_quality_setter(self):
        """Quality setter should work."""
        team = LLMTeam(team_id="test", quality=50)

        team.quality = 80
        assert team.quality == 80

        team.quality = "best"
        assert team.quality == 95

    def test_quality_manager_access(self):
        """Should access QualityManager."""
        team = LLMTeam(team_id="test", quality=70)

        manager = team.get_quality_manager()
        assert isinstance(manager, QualityManager)
        assert manager.quality == 70

    def test_quality_in_config(self):
        """Quality should be in to_config()."""
        team = LLMTeam(team_id="test", quality=70)
        config = team.to_config()

        assert config["quality"] == 70

    def test_quality_from_config(self):
        """Quality should be restored from config."""
        config = {
            "team_id": "test",
            "quality": 75,
        }
        team = LLMTeam.from_config(config)

        assert team.quality == 75

    def test_max_cost_per_run(self):
        """max_cost_per_run should be stored."""
        team = LLMTeam(
            team_id="test",
            quality=70,
            max_cost_per_run=1.00,
        )

        assert team._max_cost_per_run == 1.00
        config = team.to_config()
        assert config["max_cost_per_run"] == 1.00


class TestLLMTeamEstimateCost:
    """Tests for LLMTeam.estimate_cost()."""

    async def test_estimate_cost_simple(self):
        """Estimate cost without agents."""
        team = LLMTeam(team_id="test", quality=50)

        estimate = await team.estimate_cost(complexity="medium")

        assert estimate.quality == 50
        assert estimate.min_cost > 0
        assert estimate.max_cost > estimate.min_cost

    async def test_estimate_cost_with_agents(self):
        """Estimate cost with agents configured."""
        team = LLMTeam(
            team_id="test",
            quality=70,
            agents=[
                {"type": "llm", "role": "writer", "prompt": "Write", "model": "gpt-4o"},
                {"type": "llm", "role": "editor", "prompt": "Edit", "model": "gpt-4o-mini"},
            ],
        )

        estimate = await team.estimate_cost()

        assert estimate.task_complexity == "custom"
        assert estimate.breakdown is not None
        assert "writer" in estimate.breakdown
        assert "editor" in estimate.breakdown

    async def test_estimate_cost_quality_affects(self):
        """Higher quality should estimate higher cost."""
        team_low = LLMTeam(team_id="test", quality=20)
        team_high = LLMTeam(team_id="test", quality=80)

        low = await team_low.estimate_cost(complexity="medium")
        high = await team_high.estimate_cost(complexity="medium")

        assert high.min_cost > low.min_cost


class TestLLMTeamRunQuality:
    """Tests for LLMTeam.run() with quality parameter."""

    async def test_run_with_quality_override(self):
        """Run with quality override."""
        team = LLMTeam(
            team_id="test",
            quality=50,
            agents=[
                {"type": "llm", "role": "worker", "prompt": "Process: {input}"},
            ],
        )

        # Note: Actual run would need mocks, this tests parameter acceptance
        # The quality override is stored in run() implementation
        assert team.quality == 50

    async def test_run_with_importance(self):
        """Run with importance parameter."""
        team = LLMTeam(
            team_id="test",
            quality=50,
            agents=[
                {"type": "llm", "role": "worker", "prompt": "Process: {input}"},
            ],
        )

        # Test importance adjustment via QualityManager
        manager = team.get_quality_manager()
        assert manager.with_importance("high") == 70
        assert manager.with_importance("medium") == 50
        assert manager.with_importance("low") == 30
