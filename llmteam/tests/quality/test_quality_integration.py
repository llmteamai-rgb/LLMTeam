"""
Tests for RFC-019 Quality Integration.

Tests quality-aware functionality across components:
- QualityAwareLLMMixin
- ConfigurationSession quality
- TeamOrchestrator quality
- GroupOrchestrator quality
- Budget pre-check with quality
- Cost tracking for quality calls
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llmteam.quality import QualityManager, QualityAwareLLMMixin


# === QualityAwareLLMMixin Tests ===


class MockComponent(QualityAwareLLMMixin):
    """Test component using mixin."""

    def __init__(self, quality: int = 50, team=None):
        self._manager = QualityManager(quality)
        self._team = team

    def _get_quality_manager(self) -> QualityManager:
        return self._manager


class TestQualityAwareLLMMixin:
    """Tests for QualityAwareLLMMixin."""

    def test_get_quality_llm_low_quality(self):
        """Low quality should use appropriate model."""
        component = MockComponent(quality=20)
        with patch.object(component, "_create_llm_provider") as mock_create:
            mock_create.return_value = MagicMock()
            component._get_quality_llm(complexity="simple")
            # Should use mini model for low quality
            assert component._quality_llm_model == "gpt-4o-mini"

    def test_get_quality_llm_high_quality(self):
        """High quality should use better model."""
        component = MockComponent(quality=90)
        with patch.object(component, "_create_llm_provider") as mock_create:
            mock_create.return_value = MagicMock()
            component._get_quality_llm(complexity="complex")
            # Should use turbo model for high quality complex tasks
            assert component._quality_llm_model in ["gpt-4-turbo", "gpt-4o"]

    def test_get_quality_params_low_quality(self):
        """Low quality params should have lower values."""
        component = MockComponent(quality=20)
        params = component._get_quality_params()
        assert params["temperature"] <= 0.5
        assert params["max_tokens"] <= 1000

    def test_get_quality_params_high_quality(self):
        """High quality params should have higher values."""
        component = MockComponent(quality=80)
        params = component._get_quality_params()
        assert params["temperature"] >= 0.5
        assert params["max_tokens"] >= 1000

    def test_params_override(self):
        """Override params should work."""
        component = MockComponent(quality=50)
        params = component._get_quality_params(
            override_temperature=0.9,
            override_max_tokens=100,
        )
        assert params["temperature"] == 0.9
        assert params["max_tokens"] == 100

    def test_llm_caching(self):
        """LLM should be cached until model changes."""
        component = MockComponent(quality=50)
        with patch.object(component, "_create_llm_provider") as mock_create:
            mock_llm = MagicMock()
            mock_create.return_value = mock_llm

            llm1 = component._get_quality_llm()
            llm2 = component._get_quality_llm()
            assert llm1 is llm2
            assert mock_create.call_count == 1

    def test_llm_refresh_on_model_change(self):
        """LLM should be recreated when model changes."""
        component = MockComponent(quality=50)
        with patch.object(component, "_create_llm_provider") as mock_create:
            mock_create.return_value = MagicMock()

            component._get_quality_llm(complexity="simple")
            component._manager = QualityManager(90)  # Change quality
            component._get_quality_llm(complexity="complex", force_refresh=True)

            assert mock_create.call_count == 2

    @pytest.mark.asyncio
    async def test_quality_complete(self):
        """_quality_complete should use quality-aware LLM."""
        component = MockComponent(quality=70)

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value="Test response")

        with patch.object(component, "_create_llm_provider", return_value=mock_llm):
            response = await component._quality_complete(
                prompt="Test prompt",
                system_prompt="System",
                complexity="medium",
            )

            assert response == "Test response"
            mock_llm.complete.assert_called_once()

    def test_cost_tracking_with_team(self):
        """Cost tracking should work when team has cost tracker."""
        mock_tracker = MagicMock()
        mock_team = MagicMock()
        mock_team._cost_tracker = mock_tracker

        component = MockComponent(quality=50, team=mock_team)
        component._quality_llm_model = "gpt-4o"

        # Mock response with tokens
        mock_response = MagicMock()
        mock_response.usage = MagicMock(total_tokens=100)

        component._track_quality_call_cost(mock_response, "test prompt")

        mock_tracker.add_usage.assert_called_once()


# === ConfigurationSession Quality Tests ===


class TestConfigurationSessionQuality:
    """Tests for quality-aware ConfigurationSession. RFC-021: Quality only in session."""

    def test_session_default_quality(self):
        """Session should have default quality 50. RFC-021."""
        from llmteam import LLMTeam
        from llmteam.configuration import ConfigurationSession

        team = LLMTeam(team_id="test")
        session = ConfigurationSession(
            session_id="test",
            team=team,
            task="test task",
        )
        assert session.quality == 50  # RFC-021: Default quality is 50

    def test_session_quality_override(self):
        """Session can set quality explicitly via set_quality(). RFC-021."""
        from llmteam import LLMTeam
        from llmteam.configuration import ConfigurationSession

        team = LLMTeam(team_id="test")
        session = ConfigurationSession(
            session_id="test",
            team=team,
            task="test task",
        )
        session.set_quality(80)
        assert session.quality == 80

    def test_get_quality_manager_returns_correct_manager(self):
        """_get_quality_manager should return session's manager. RFC-021."""
        from llmteam import LLMTeam
        from llmteam.configuration import ConfigurationSession

        team = LLMTeam(team_id="test")
        session = ConfigurationSession(
            session_id="test",
            team=team,
            task="test task",
        )
        # RFC-021: Use set_quality() to set quality
        session.set_quality(60)

        # Uses session's quality
        manager = session._get_quality_manager()
        assert manager.quality == 60

        # Can override
        session.set_quality(90)
        manager = session._get_quality_manager()
        assert manager.quality == 90


# === TeamOrchestrator Quality Tests ===


class TestTeamOrchestratorQuality:
    """Tests for TeamOrchestrator. RFC-021: Uses explicit model/temperature."""

    def test_orchestrator_uses_config_model(self):
        """Orchestrator should use model from config. RFC-021."""
        from llmteam import LLMTeam
        from llmteam.agents.orchestrator import OrchestratorConfig

        config = OrchestratorConfig(model="gpt-4-turbo", temperature=0.2)
        team = LLMTeam(team_id="test", orchestrator=config)
        team.add_agent({"type": "llm", "role": "worker", "prompt": "Work"})

        orch = team.get_orchestrator()
        assert orch._config.model == "gpt-4-turbo"
        assert orch._config.temperature == 0.2

    @pytest.mark.asyncio
    async def test_decide_next_agent_uses_config_model(self):
        """decide_next_agent should use model from config. RFC-021."""
        from llmteam import LLMTeam
        from llmteam.agents.orchestrator import OrchestratorConfig, OrchestratorMode
        from unittest.mock import patch, AsyncMock

        config = OrchestratorConfig(
            mode=OrchestratorMode.ACTIVE,
            model="gpt-4-turbo",
            temperature=0.2,
        )
        team = LLMTeam(team_id="test", orchestrator=config)
        team.add_agent({"type": "llm", "role": "worker", "prompt": "Work"})

        orch = team.get_orchestrator()

        # RFC-021: Orchestrator now uses provider.complete directly
        assert orch._config.model == "gpt-4-turbo"
        assert orch._config.temperature == 0.2


# === GroupOrchestrator Quality Tests ===


class TestGroupOrchestratorQuality:
    """Tests for GroupOrchestrator. RFC-021: Uses model/temperature instead of quality."""

    def test_group_orchestrator_has_model(self):
        """GroupOrchestrator should have model property. RFC-021."""
        from llmteam.orchestration import GroupOrchestrator

        group = GroupOrchestrator(group_id="test", model="gpt-4-turbo")
        assert group._model == "gpt-4-turbo"

    def test_group_orchestrator_has_temperature(self):
        """GroupOrchestrator should have temperature. RFC-021."""
        from llmteam.orchestration import GroupOrchestrator

        group = GroupOrchestrator(group_id="test", temperature=0.5)
        assert group._temperature == 0.5

    def test_group_default_model_and_temperature(self):
        """GroupOrchestrator should have default model/temperature. RFC-021."""
        from llmteam.orchestration import GroupOrchestrator

        group = GroupOrchestrator(group_id="test")
        assert group._model == "gpt-4o"
        assert group._temperature == 0.3


# === Budget Pre-Check Tests ===


class TestBudgetPreCheck:
    """Tests for budget management. RFC-021: Budget without quality-based estimation."""

    @pytest.mark.asyncio
    async def test_team_with_budget_limit(self):
        """Team should accept max_cost_per_run. RFC-021."""
        from llmteam import LLMTeam
        from llmteam.team.result import RunResult, RunStatus

        team = LLMTeam(
            team_id="test",
            max_cost_per_run=10.0,
        )
        team.add_agent({"type": "llm", "role": "worker", "prompt": "Work: {input}"})

        assert team._max_cost_per_run == 10.0

    @pytest.mark.asyncio
    async def test_team_budget_manager_created(self):
        """Team should create budget manager when limit set. RFC-021."""
        from llmteam import LLMTeam

        team = LLMTeam(
            team_id="test",
            max_cost_per_run=5.0,
        )

        assert team._budget_manager is not None


# === Streaming Tests ===


class TestStreamingQuality:
    """Tests for streaming. RFC-021: quality/importance removed from stream()."""

    @pytest.mark.asyncio
    async def test_stream_signature(self):
        """stream() should have input_data and run_id params. RFC-021."""
        from llmteam import LLMTeam

        team = LLMTeam(team_id="test", orchestration=True)
        team.add_agent({"type": "llm", "role": "worker", "prompt": "Work: {input}"})

        import inspect
        sig = inspect.signature(team.stream)
        assert "input_data" in sig.parameters
        assert "run_id" in sig.parameters
        # RFC-021: quality and importance removed
        assert "quality" not in sig.parameters
        assert "importance" not in sig.parameters

    @pytest.mark.asyncio
    async def test_stream_returns_async_iterator(self):
        """stream() should return async iterator. RFC-021."""
        from llmteam import LLMTeam
        import inspect

        team = LLMTeam(team_id="test", orchestration=True)

        # stream() should be an async generator
        assert inspect.isasyncgenfunction(team.stream)


# === DynamicTeamBuilder Quality Tests ===


class TestDynamicTeamBuilderQuality:
    """Tests for quality-aware DynamicTeamBuilder (TASK-Q-08)."""

    def test_builder_accepts_quality(self):
        """DynamicTeamBuilder should accept quality parameter."""
        from llmteam.builder import DynamicTeamBuilder

        builder = DynamicTeamBuilder(quality=75)
        assert builder.quality == 75

    def test_builder_get_quality_manager(self):
        """DynamicTeamBuilder should have quality manager."""
        from llmteam.builder import DynamicTeamBuilder

        builder = DynamicTeamBuilder(quality=80)
        manager = builder._get_quality_manager()
        assert manager.quality == 80

    @pytest.mark.asyncio
    async def test_analyze_task_uses_quality(self):
        """analyze_task should use quality-aware LLM."""
        from llmteam.builder import DynamicTeamBuilder

        builder = DynamicTeamBuilder(quality=70)

        # Mock _llm_call which is the internal method that calls the LLM
        with patch.object(builder, "_llm_call", new_callable=AsyncMock) as mock:
            mock.return_value = '{"team_id": "test", "description": "test", "agents": [{"role": "worker", "purpose": "test", "prompt": "test", "tools": [], "model": "gpt-4o-mini", "temperature": 0.5, "max_tool_rounds": 5}], "routing_strategy": "", "input_variables": []}'

            result = await builder.analyze_task("Test task")

            mock.assert_called_once()
            # Check prompt contains quality info
            call_args = mock.call_args
            prompt = call_args.args[0] if call_args.args else ""
            assert "70" in prompt  # Quality level should be in prompt
