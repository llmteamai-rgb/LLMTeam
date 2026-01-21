"""
Tests for GroupOrchestrator (RFC-004).
"""

import pytest
from datetime import datetime

from llmteam.orchestration import (
    GroupOrchestrator,
    GroupRole,
    TeamReport,
    GroupReport,
    GroupResult,
)


class TestGroupRole:
    """Tests for GroupRole enum."""

    def test_report_collector_role(self):
        """Test REPORT_COLLECTOR is the default role."""
        assert GroupRole.REPORT_COLLECTOR.value == "report_collector"


class TestTeamReport:
    """Tests for TeamReport dataclass."""

    def test_create_report(self):
        """Test creating a team report."""
        report = TeamReport(
            team_id="team_1",
            run_id="run_123",
            success=True,
            duration_ms=100,
            agents_executed=["agent_a", "agent_b"],
        )

        assert report.team_id == "team_1"
        assert report.run_id == "run_123"
        assert report.success is True
        assert report.duration_ms == 100
        assert report.agents_executed == ["agent_a", "agent_b"]
        assert report.errors == []

    def test_to_dict(self):
        """Test converting report to dict."""
        report = TeamReport(
            team_id="team_1",
            run_id="run_123",
            success=True,
            duration_ms=100,
            agents_executed=["agent_a"],
        )
        data = report.to_dict()

        assert data["team_id"] == "team_1"
        assert data["run_id"] == "run_123"
        assert data["success"] is True
        assert "created_at" in data

    def test_from_dict(self):
        """Test creating report from dict."""
        data = {
            "team_id": "team_1",
            "run_id": "run_123",
            "success": True,
            "duration_ms": 100,
            "agents_executed": ["agent_a"],
        }
        report = TeamReport.from_dict(data)

        assert report.team_id == "team_1"
        assert report.success is True


class TestGroupReport:
    """Tests for GroupReport dataclass."""

    def test_create_report(self):
        """Test creating a group report."""
        report = GroupReport(
            group_id="group_1",
            role="report_collector",
            teams_count=2,
            teams_succeeded=2,
            teams_failed=0,
            total_duration_ms=200,
            summary="All teams succeeded",
        )

        assert report.group_id == "group_1"
        assert report.teams_count == 2
        assert report.teams_succeeded == 2
        assert report.teams_failed == 0

    def test_to_dict_from_dict(self):
        """Test round-trip serialization."""
        report = GroupReport(
            group_id="group_1",
            role="report_collector",
            teams_count=2,
            teams_succeeded=1,
            teams_failed=1,
            total_duration_ms=200,
        )

        data = report.to_dict()
        restored = GroupReport.from_dict(data)

        assert restored.group_id == report.group_id
        assert restored.teams_count == report.teams_count


class TestGroupResult:
    """Tests for GroupResult dataclass."""

    def test_create_result(self):
        """Test creating a group result."""
        result = GroupResult(
            success=True,
            output={"team_1": "output_1"},
            errors=[],
        )

        assert result.success is True
        assert result.output == {"team_1": "output_1"}
        assert result.errors == []


class TestGroupOrchestrator:
    """Tests for GroupOrchestrator class."""

    def test_init_default(self):
        """Test default initialization."""
        orch = GroupOrchestrator()

        assert orch.group_id.startswith("group_")
        assert orch.role == GroupRole.REPORT_COLLECTOR
        assert orch.teams_count == 0

    def test_init_custom_id(self):
        """Test initialization with custom ID."""
        orch = GroupOrchestrator(group_id="my_group")

        assert orch.group_id == "my_group"

    def test_add_remove_team(self):
        """Test adding and removing teams."""
        from unittest.mock import MagicMock

        orch = GroupOrchestrator(group_id="test")

        # Create mock team
        mock_team = MagicMock()
        mock_team.team_id = "team_1"

        # Add team
        orch.add_team(mock_team)
        assert orch.teams_count == 1
        assert "team_1" in orch.list_teams()

        # Remove team
        removed = orch.remove_team("team_1")
        assert removed is True
        assert orch.teams_count == 0

        # Remove non-existent team
        removed = orch.remove_team("nonexistent")
        assert removed is False

    def test_get_team(self):
        """Test getting a team by ID."""
        from unittest.mock import MagicMock

        orch = GroupOrchestrator(group_id="test")

        mock_team = MagicMock()
        mock_team.team_id = "team_1"
        orch.add_team(mock_team)

        team = orch.get_team("team_1")
        assert team is mock_team

        team = orch.get_team("nonexistent")
        assert team is None

    def test_list_teams(self):
        """Test listing team IDs."""
        from unittest.mock import MagicMock

        orch = GroupOrchestrator(group_id="test")

        for i in range(3):
            mock_team = MagicMock()
            mock_team.team_id = f"team_{i}"
            orch.add_team(mock_team)

        teams = orch.list_teams()
        assert len(teams) == 3
        assert "team_0" in teams
        assert "team_1" in teams
        assert "team_2" in teams

    def test_repr(self):
        """Test string representation."""
        orch = GroupOrchestrator(group_id="my_group")

        repr_str = repr(orch)
        assert "GroupOrchestrator" in repr_str
        assert "my_group" in repr_str
        assert "report_collector" in repr_str

    async def test_execute_sequential_empty(self):
        """Test executing with no teams."""
        orch = GroupOrchestrator(group_id="test")

        result = await orch.execute({"query": "test"})

        assert result.success is True
        assert result.output == {}
        assert result.report is not None
        assert result.report.teams_count == 0

    async def test_execute_sequential_with_teams(self):
        """Test sequential execution with mock teams."""
        from unittest.mock import MagicMock, AsyncMock

        orch = GroupOrchestrator(group_id="test")

        # Create mock teams
        for i in range(2):
            mock_team = MagicMock()
            mock_team.team_id = f"team_{i}"
            mock_team.run = AsyncMock(return_value=MagicMock(
                success=True,
                output=f"output_{i}",
                run_id=f"run_{i}",
                duration_ms=100,
            ))
            mock_team.get_orchestrator = MagicMock(return_value=None)
            orch.add_team(mock_team)

        result = await orch.execute({"query": "test"}, parallel=False)

        assert result.success is True
        assert result.report is not None
        assert result.report.teams_count == 2
        assert result.report.teams_succeeded == 2
        assert result.report.teams_failed == 0

    async def test_execute_parallel_with_teams(self):
        """Test parallel execution with mock teams."""
        from unittest.mock import MagicMock, AsyncMock

        orch = GroupOrchestrator(group_id="test")

        # Create mock teams
        for i in range(2):
            mock_team = MagicMock()
            mock_team.team_id = f"team_{i}"
            mock_team.run = AsyncMock(return_value=MagicMock(
                success=True,
                output=f"output_{i}",
                run_id=f"run_{i}",
                duration_ms=100,
            ))
            mock_team.get_orchestrator = MagicMock(return_value=None)
            orch.add_team(mock_team)

        result = await orch.execute({"query": "test"}, parallel=True)

        assert result.success is True
        assert result.report is not None
        assert result.report.teams_count == 2

    async def test_execute_with_errors(self):
        """Test execution with team errors."""
        from unittest.mock import MagicMock, AsyncMock

        orch = GroupOrchestrator(group_id="test")

        # Create mock team that raises error
        mock_team = MagicMock()
        mock_team.team_id = "failing_team"
        mock_team.run = AsyncMock(side_effect=Exception("Team failed"))
        orch.add_team(mock_team)

        result = await orch.execute({"query": "test"})

        assert result.success is False
        assert len(result.errors) == 1
        assert "failing_team" in result.errors[0]

    async def test_last_report(self):
        """Test last_report property."""
        orch = GroupOrchestrator(group_id="test")

        # Initially no report
        assert orch.last_report is None

        # After execute, report is stored
        await orch.execute({"query": "test"})
        assert orch.last_report is not None
        assert orch.last_report.group_id == "test"
