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
    """Tests for quality-aware ConfigurationSession."""

    def test_session_inherits_team_quality(self):
        """Session should inherit quality from team."""
        from llmteam import LLMTeam
        from llmteam.configuration import ConfigurationSession

        team = LLMTeam(team_id="test", quality=75)
        session = ConfigurationSession(
            session_id="test",
            team=team,
            task="test task",
        )
        assert session.quality == 75

    def test_session_quality_override(self):
        """Session can override team quality."""
        from llmteam import LLMTeam
        from llmteam.configuration import ConfigurationSession

        team = LLMTeam(team_id="test", quality=30)
        session = ConfigurationSession(
            session_id="test",
            team=team,
            task="test task",
        )
        session.set_quality(80)
        assert session.quality == 80

    def test_get_quality_manager_returns_correct_manager(self):
        """_get_quality_manager should return appropriate manager."""
        from llmteam import LLMTeam
        from llmteam.configuration import ConfigurationSession

        team = LLMTeam(team_id="test", quality=60)
        session = ConfigurationSession(
            session_id="test",
            team=team,
            task="test task",
        )

        # Without override - uses team's quality
        manager = session._get_quality_manager()
        assert manager.quality == 60

        # With override - uses session's quality
        session.set_quality(90)
        manager = session._get_quality_manager()
        assert manager.quality == 90


# === TeamOrchestrator Quality Tests ===


class TestTeamOrchestratorQuality:
    """Tests for quality-aware TeamOrchestrator."""

    def test_orchestrator_uses_team_quality(self):
        """Orchestrator should use team's quality manager."""
        from llmteam import LLMTeam

        team = LLMTeam(team_id="test", quality=85)
        team.add_agent({"type": "llm", "role": "worker", "prompt": "Work"})

        orch = team.get_orchestrator()
        manager = orch._get_quality_manager()
        assert manager.quality == 85

    @pytest.mark.asyncio
    async def test_decide_next_agent_uses_quality_llm(self):
        """decide_next_agent should use quality-aware LLM."""
        from llmteam import LLMTeam
        from llmteam.agents.orchestrator import OrchestratorConfig, OrchestratorMode

        team = LLMTeam(
            team_id="test",
            quality=70,
            orchestrator=OrchestratorConfig(mode=OrchestratorMode.ACTIVE),
        )
        team.add_agent({"type": "llm", "role": "worker", "prompt": "Work"})

        orch = team.get_orchestrator()

        with patch.object(orch, "_quality_complete", new_callable=AsyncMock) as mock:
            mock.return_value = '{"next_agent": "worker", "reason": "test"}'

            await orch.decide_next_agent(
                current_state={"input": "test"},
                available_agents=["worker"],
            )

            mock.assert_called_once()
            call_kwargs = mock.call_args.kwargs
            assert call_kwargs["complexity"] == "simple"


# === GroupOrchestrator Quality Tests ===


class TestGroupOrchestratorQuality:
    """Tests for quality-aware GroupOrchestrator."""

    def test_group_orchestrator_has_quality(self):
        """GroupOrchestrator should have quality property."""
        from llmteam.orchestration import GroupOrchestrator

        group = GroupOrchestrator(group_id="test", quality=65)
        assert group.quality == 65

    def test_group_orchestrator_quality_setter(self):
        """GroupOrchestrator quality should be settable."""
        from llmteam.orchestration import GroupOrchestrator

        group = GroupOrchestrator(group_id="test", quality=50)
        group.quality = 80
        assert group.quality == 80

    def test_group_get_quality_manager(self):
        """GroupOrchestrator should return quality manager."""
        from llmteam.orchestration import GroupOrchestrator

        group = GroupOrchestrator(group_id="test", quality=75)
        manager = group._get_quality_manager()
        assert manager.quality == 75


# === Budget Pre-Check Tests ===


class TestBudgetPreCheck:
    """Tests for RFC-019 budget pre-check with quality."""

    @pytest.mark.asyncio
    async def test_budget_precheck_passes_when_under_limit(self):
        """Run should proceed when estimated cost is under budget."""
        from llmteam import LLMTeam
        from llmteam.team.result import RunResult, RunStatus

        team = LLMTeam(
            team_id="test",
            quality=50,
            max_cost_per_run=10.0,  # High limit
        )
        team.add_agent({"type": "llm", "role": "worker", "prompt": "Work: {input}"})

        # Mock agent execution (bypass actual router)
        with patch.object(team, "_run_router_mode", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = RunResult(success=True, status=RunStatus.COMPLETED)

            result = await team.run({"input": "test"})
            # Should not fail on budget pre-check
            assert "Pre-flight budget check failed" not in (result.error or "")

    @pytest.mark.asyncio
    async def test_budget_precheck_fails_when_over_limit(self):
        """Run should fail when estimated cost exceeds budget."""
        from llmteam import LLMTeam

        team = LLMTeam(
            team_id="test",
            quality=90,  # High quality = higher cost
            max_cost_per_run=0.0001,  # Very low limit
        )
        team.add_agent({"type": "llm", "role": "worker", "prompt": "Work: {input}"})

        result = await team.run({"input": "test"})

        assert result.success is False
        assert "Pre-flight budget check failed" in (result.error or "")


# === Streaming Quality Tests ===


class TestStreamingQuality:
    """Tests for quality in streaming path (TASK-Q-10)."""

    @pytest.mark.asyncio
    async def test_stream_accepts_quality_override(self):
        """stream() should accept quality parameter."""
        from llmteam import LLMTeam

        team = LLMTeam(team_id="test", quality=50, orchestration=True)
        team.add_agent({"type": "llm", "role": "worker", "prompt": "Work: {input}"})

        # Just verify the method signature accepts quality
        # Actual execution would require more complex mocking
        import inspect
        sig = inspect.signature(team.stream)
        assert "quality" in sig.parameters
        assert "importance" in sig.parameters

    @pytest.mark.asyncio
    async def test_stream_accepts_importance(self):
        """stream() should accept importance parameter."""
        from llmteam import LLMTeam

        team = LLMTeam(team_id="test", quality=50, orchestration=True)
        team.add_agent({"type": "llm", "role": "worker", "prompt": "Work: {input}"})

        import inspect
        sig = inspect.signature(team.stream)
        assert "importance" in sig.parameters


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
