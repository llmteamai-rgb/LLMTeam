"""Tests for TeamOrchestrator v4.1 architecture."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock

from llmteam.agents.report import AgentReport
from llmteam.agents.orchestrator import (
    TeamOrchestrator,
    OrchestratorMode,
    OrchestratorScope,
    OrchestratorConfig,
    RoutingDecision,
    RecoveryAction,
)


class TestAgentReport:
    """Tests for AgentReport model."""

    def test_create_report(self):
        """Test creating AgentReport."""
        report = AgentReport(
            agent_id="writer",
            agent_role="writer",
            agent_type="llm",
            started_at=datetime(2024, 1, 1, 12, 0, 0),
            completed_at=datetime(2024, 1, 1, 12, 0, 1),
        )

        assert report.agent_id == "writer"
        assert report.agent_type == "llm"
        assert report.duration_ms == 1000  # 1 second
        assert report.success is True

    def test_create_factory_method(self):
        """Test AgentReport.create factory method."""
        report = AgentReport.create(
            agent_id="reviewer",
            agent_role="reviewer",
            agent_type="llm",
            started_at=datetime.utcnow(),
            input_data={"query": "review this"},
            output={"output": "looks good"},
            success=True,
            tokens_used=150,
            model="gpt-4o",
        )

        assert report.agent_id == "reviewer"
        assert report.success is True
        assert report.tokens_used == 150
        assert report.model == "gpt-4o"
        assert "review" in report.input_summary

    def test_to_dict(self):
        """Test AgentReport serialization."""
        report = AgentReport(
            agent_id="test",
            agent_role="test",
            agent_type="llm",
            started_at=datetime(2024, 1, 1, 12, 0, 0),
            completed_at=datetime(2024, 1, 1, 12, 0, 1),
            success=True,
        )

        data = report.to_dict()

        assert data["agent_id"] == "test"
        assert data["success"] is True
        assert "started_at" in data
        assert "completed_at" in data

    def test_to_log_line(self):
        """Test AgentReport log line format."""
        report = AgentReport(
            agent_id="writer",
            agent_role="writer",
            agent_type="llm",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            duration_ms=500,
            success=True,
        )

        log = report.to_log_line()

        assert "writer" in log
        assert "LLM" in log
        assert "500ms" in log
        assert "OK" in log

    def test_failed_report(self):
        """Test failed AgentReport."""
        report = AgentReport.create(
            agent_id="failing",
            agent_role="failing",
            agent_type="llm",
            started_at=datetime.utcnow(),
            input_data={},
            output=None,
            success=False,
            error=ValueError("test error"),
        )

        assert report.success is False
        assert report.error == "test error"
        assert report.error_type == "ValueError"


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = OrchestratorConfig()

        assert config.mode == OrchestratorMode.PASSIVE
        assert config.model == "gpt-4o-mini"
        assert config.generate_report is True

    def test_active_config(self):
        """Test ACTIVE mode configuration."""
        config = OrchestratorConfig(mode=OrchestratorMode.ACTIVE)

        assert OrchestratorMode.SUPERVISOR in config.mode
        assert OrchestratorMode.REPORTER in config.mode
        assert OrchestratorMode.ROUTER in config.mode

    def test_full_config(self):
        """Test FULL mode configuration."""
        config = OrchestratorConfig(mode=OrchestratorMode.FULL)

        assert OrchestratorMode.SUPERVISOR in config.mode
        assert OrchestratorMode.REPORTER in config.mode
        assert OrchestratorMode.ROUTER in config.mode
        assert OrchestratorMode.RECOVERY in config.mode


class TestTeamOrchestrator:
    """Tests for TeamOrchestrator."""

    def test_init_passive_mode(self):
        """Test orchestrator initialization in PASSIVE mode."""
        mock_team = MagicMock()
        mock_team.team_id = "test_team"

        orch = TeamOrchestrator(team=mock_team)

        assert orch.mode == OrchestratorMode.PASSIVE
        assert orch.scope == OrchestratorScope.TEAM
        assert orch.is_router is False

    def test_init_active_mode(self):
        """Test orchestrator initialization in ACTIVE mode."""
        mock_team = MagicMock()
        mock_team.team_id = "test_team"

        config = OrchestratorConfig(mode=OrchestratorMode.ACTIVE)
        orch = TeamOrchestrator(team=mock_team, config=config)

        assert orch.is_router is True
        assert OrchestratorMode.ROUTER in orch.mode

    def test_start_end_run(self):
        """Test run lifecycle."""
        mock_team = MagicMock()
        orch = TeamOrchestrator(team=mock_team)

        orch.start_run("run-123")
        assert orch._current_run_id == "run-123"
        assert orch._reports == []

        orch.end_run()
        assert orch._current_run_id is None

    def test_receive_report(self):
        """Test receiving agent reports."""
        mock_team = MagicMock()
        orch = TeamOrchestrator(team=mock_team)

        orch.start_run("run-123")

        report = AgentReport(
            agent_id="writer",
            agent_role="writer",
            agent_type="llm",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
        )

        orch.receive_report(report)

        assert len(orch.reports) == 1
        assert orch.reports[0].agent_id == "writer"

    def test_generate_report_no_reports(self):
        """Test report generation with no reports."""
        mock_team = MagicMock()
        mock_team.team_id = "test"
        orch = TeamOrchestrator(team=mock_team)

        report = orch.generate_report()

        assert "No agent reports collected" in report

    def test_generate_markdown_report(self):
        """Test markdown report generation."""
        mock_team = MagicMock()
        mock_team.team_id = "test_team"
        orch = TeamOrchestrator(team=mock_team)

        orch.start_run("run-123")

        # Add some reports
        report1 = AgentReport(
            agent_id="writer",
            agent_role="writer",
            agent_type="llm",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            duration_ms=500,
            success=True,
            output_summary="Hello world",
        )
        orch.receive_report(report1)

        report2 = AgentReport(
            agent_id="editor",
            agent_role="editor",
            agent_type="llm",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            duration_ms=300,
            success=True,
            output_summary="Edited content",
        )
        orch.receive_report(report2)

        markdown = orch.generate_report()

        assert "# Execution Report" in markdown
        assert "writer" in markdown
        assert "editor" in markdown
        assert "test_team" in markdown

    def test_generate_json_report(self):
        """Test JSON report generation."""
        mock_team = MagicMock()
        mock_team.team_id = "test_team"

        config = OrchestratorConfig(report_format="json")
        orch = TeamOrchestrator(team=mock_team, config=config)

        orch.start_run("run-123")

        report = AgentReport(
            agent_id="writer",
            agent_role="writer",
            agent_type="llm",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
        )
        orch.receive_report(report)

        json_report = orch.generate_report()

        import json
        data = json.loads(json_report)
        assert data["team_id"] == "test_team"
        assert len(data["reports"]) == 1

    def test_get_summary(self):
        """Test execution summary."""
        mock_team = MagicMock()
        orch = TeamOrchestrator(team=mock_team)

        orch.start_run("run-123")

        # Add reports
        orch.receive_report(AgentReport(
            agent_id="a1",
            agent_role="a1",
            agent_type="llm",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            duration_ms=100,
            tokens_used=50,
            success=True,
        ))
        orch.receive_report(AgentReport(
            agent_id="a2",
            agent_role="a2",
            agent_type="llm",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            duration_ms=200,
            tokens_used=100,
            success=False,
            error="Test error",
        ))

        summary = orch.get_summary()

        assert summary["agents_executed"] == 2
        assert summary["agents_succeeded"] == 1
        assert summary["agents_failed"] == 1
        assert summary["total_duration_ms"] == 300
        assert summary["total_tokens_used"] == 150
        assert summary["execution_order"] == ["a1", "a2"]
        assert len(summary["errors"]) == 1

    def test_promote_to_group(self):
        """Test promoting to GROUP scope."""
        mock_team = MagicMock()
        orch = TeamOrchestrator(team=mock_team)

        assert orch.scope == OrchestratorScope.TEAM

        other_teams = [MagicMock(), MagicMock()]
        orch.promote_to_group("group-1", other_teams)

        assert orch.scope == OrchestratorScope.GROUP


class TestTeamOrchestratorRouting:
    """Tests for routing functionality."""

    @pytest.mark.asyncio
    async def test_decide_next_agent_no_router_mode(self):
        """Test routing when not in ROUTER mode."""
        mock_team = MagicMock()
        orch = TeamOrchestrator(team=mock_team)  # PASSIVE mode

        decision = await orch.decide_next_agent(
            current_state={"input": "test"},
            available_agents=["writer", "editor"],
        )

        # Should return first agent since ROUTER mode is off
        assert decision.next_agent == "writer"
        assert "not enabled" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_decide_next_agent_router_mode_no_llm(self):
        """Test routing in ROUTER mode without LLM available."""
        from unittest.mock import patch

        mock_team = MagicMock()

        config = OrchestratorConfig(mode=OrchestratorMode.ACTIVE)
        orch = TeamOrchestrator(team=mock_team, config=config)

        # Mock _get_llm to return None (simulating no LLM available)
        with patch.object(orch, '_get_llm', return_value=None):
            decision = await orch.decide_next_agent(
                current_state={"input": "test"},
                available_agents=["writer", "editor"],
            )

        # Should fallback to first agent
        assert decision.next_agent == "writer"
        assert "no llm" in decision.reason.lower()


class TestTeamOrchestratorRecovery:
    """Tests for recovery functionality."""

    @pytest.mark.asyncio
    async def test_decide_recovery_no_recovery_mode(self):
        """Test recovery when not in RECOVERY mode."""
        mock_team = MagicMock()
        orch = TeamOrchestrator(team=mock_team)  # PASSIVE mode

        decision = await orch.decide_recovery(
            error=ValueError("test"),
            failed_agent="writer",
            context={},
        )

        # Default: auto_retry is True
        assert decision.action == RecoveryAction.RETRY

    @pytest.mark.asyncio
    async def test_decide_recovery_no_auto_retry(self):
        """Test recovery without auto_retry."""
        mock_team = MagicMock()

        config = OrchestratorConfig(auto_retry=False)
        orch = TeamOrchestrator(team=mock_team, config=config)

        decision = await orch.decide_recovery(
            error=ValueError("test"),
            failed_agent="writer",
            context={},
        )

        # Should abort since auto_retry is off and RECOVERY mode is off
        assert decision.action == RecoveryAction.ABORT
