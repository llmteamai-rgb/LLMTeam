"""
Tests for CONFIGURATOR mode (RFC-005).
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from llmteam.configuration import (
    SessionState,
    AgentSuggestion,
    TestRunResult,
    TaskAnalysis,
    ConfiguratorPrompts,
    ConfigurationSession,
)


class TestSessionState:
    """Tests for SessionState enum."""

    def test_all_states_exist(self):
        """Test all expected states exist."""
        assert SessionState.CREATED.value == "created"
        assert SessionState.ANALYZING.value == "analyzing"
        assert SessionState.SUGGESTING.value == "suggesting"
        assert SessionState.CONFIGURING.value == "configuring"
        assert SessionState.TESTING.value == "testing"
        assert SessionState.READY.value == "ready"
        assert SessionState.APPLIED.value == "applied"


class TestAgentSuggestion:
    """Tests for AgentSuggestion dataclass."""

    def test_create_suggestion(self):
        """Test creating an agent suggestion."""
        suggestion = AgentSuggestion(
            role="writer",
            type="llm",
            purpose="Generate content",
            prompt_template="Write about: {topic}",
            reasoning="Needed for content generation",
        )

        assert suggestion.role == "writer"
        assert suggestion.type == "llm"
        assert suggestion.purpose == "Generate content"

    def test_to_dict(self):
        """Test converting to dict."""
        suggestion = AgentSuggestion(
            role="writer",
            type="llm",
            purpose="Generate content",
            prompt_template="Write about: {topic}",
            reasoning="Needed",
        )

        data = suggestion.to_dict()

        assert data["role"] == "writer"
        assert data["type"] == "llm"
        assert data["purpose"] == "Generate content"

    def test_from_dict(self):
        """Test creating from dict."""
        data = {
            "role": "writer",
            "type": "llm",
            "purpose": "Generate",
            "prompt_template": "Write: {input}",
            "reasoning": "Needed",
        }

        suggestion = AgentSuggestion.from_dict(data)

        assert suggestion.role == "writer"
        assert suggestion.type == "llm"


class TestTestRunResult:
    """Tests for TestRunResult dataclass."""

    def test_create_result(self):
        """Test creating a test run result."""
        result = TestRunResult(
            test_id="test_123",
            input_data={"query": "test"},
            output={"response": "result"},
            agent_outputs={"agent1": "output1"},
            duration_ms=100,
            success=True,
        )

        assert result.test_id == "test_123"
        assert result.success is True
        assert result.duration_ms == 100

    def test_to_dict(self):
        """Test converting to dict."""
        result = TestRunResult(
            test_id="test_123",
            input_data={},
            output={},
            agent_outputs={},
            duration_ms=100,
            success=True,
            analysis="Good",
            issues=[],
            recommendations=[],
        )

        data = result.to_dict()

        assert data["test_id"] == "test_123"
        assert data["success"] is True
        assert "created_at" in data

    def test_from_dict(self):
        """Test creating from dict."""
        data = {
            "test_id": "test_123",
            "input_data": {},
            "output": {},
            "agent_outputs": {},
            "duration_ms": 100,
            "success": True,
        }

        result = TestRunResult.from_dict(data)

        assert result.test_id == "test_123"
        assert result.success is True


class TestTaskAnalysis:
    """Tests for TaskAnalysis dataclass."""

    def test_create_analysis(self):
        """Test creating task analysis."""
        analysis = TaskAnalysis(
            main_goal="Generate content",
            input_type="text",
            output_type="text",
            sub_tasks=["extract", "write", "polish"],
            complexity="moderate",
        )

        assert analysis.main_goal == "Generate content"
        assert analysis.complexity == "moderate"
        assert len(analysis.sub_tasks) == 3

    def test_to_dict(self):
        """Test converting to dict."""
        analysis = TaskAnalysis(
            main_goal="Test",
            input_type="text",
            output_type="text",
            sub_tasks=["a", "b"],
            complexity="simple",
        )

        data = analysis.to_dict()

        assert data["main_goal"] == "Test"
        assert data["sub_tasks"] == ["a", "b"]


class TestConfiguratorPrompts:
    """Tests for ConfiguratorPrompts."""

    def test_task_analysis_prompt(self):
        """Test task analysis prompt template."""
        prompt = ConfiguratorPrompts.TASK_ANALYSIS

        assert "{task}" in prompt
        assert "{constraints}" in prompt
        assert "main_goal" in prompt

    def test_team_suggestion_prompt(self):
        """Test team suggestion prompt template."""
        prompt = ConfiguratorPrompts.TEAM_SUGGESTION

        assert "{task_analysis}" in prompt
        assert "agents" in prompt
        assert "flow" in prompt

    def test_test_analysis_prompt(self):
        """Test test analysis prompt template."""
        prompt = ConfiguratorPrompts.TEST_ANALYSIS

        assert "{team_config}" in prompt
        assert "{test_input}" in prompt
        assert "{final_output}" in prompt


class TestConfigurationSession:
    """Tests for ConfigurationSession."""

    def test_create_session(self):
        """Test creating a configuration session."""
        mock_team = MagicMock()
        mock_team.team_id = "test_team"

        session = ConfigurationSession(
            session_id="session_123",
            team=mock_team,
            task="Generate LinkedIn posts",
            constraints={"tone": "professional"},
        )

        assert session.session_id == "session_123"
        assert session.task == "Generate LinkedIn posts"
        assert session.constraints == {"tone": "professional"}
        assert session.state == SessionState.CREATED

    def test_add_agent(self):
        """Test adding an agent to configuration."""
        mock_team = MagicMock()
        mock_team.team_id = "test"

        session = ConfigurationSession(
            session_id="session_123",
            team=mock_team,
            task="Test",
        )

        session.add_agent(
            role="writer",
            type="llm",
            prompt="Write: {input}",
        )

        assert len(session.current_agents) == 1
        assert session.current_agents[0]["role"] == "writer"

    def test_modify_agent(self):
        """Test modifying an agent configuration."""
        mock_team = MagicMock()
        mock_team.team_id = "test"

        session = ConfigurationSession(
            session_id="session_123",
            team=mock_team,
            task="Test",
        )

        session.add_agent(role="writer", type="llm", prompt="v1")
        session.modify_agent("writer", prompt="v2")

        assert session.current_agents[0]["prompt"] == "v2"

    def test_modify_agent_not_found(self):
        """Test modifying non-existent agent raises error."""
        mock_team = MagicMock()
        mock_team.team_id = "test"

        session = ConfigurationSession(
            session_id="session_123",
            team=mock_team,
            task="Test",
        )

        with pytest.raises(ValueError, match="not found"):
            session.modify_agent("nonexistent", prompt="v2")

    def test_remove_agent(self):
        """Test removing an agent."""
        mock_team = MagicMock()
        mock_team.team_id = "test"

        session = ConfigurationSession(
            session_id="session_123",
            team=mock_team,
            task="Test",
        )

        session.add_agent(role="writer", type="llm", prompt="v1")
        session.add_agent(role="reviewer", type="llm", prompt="v1")

        session.remove_agent("writer")

        assert len(session.current_agents) == 1
        assert session.current_agents[0]["role"] == "reviewer"

    def test_set_flow(self):
        """Test setting execution flow."""
        mock_team = MagicMock()
        mock_team.team_id = "test"

        session = ConfigurationSession(
            session_id="session_123",
            team=mock_team,
            task="Test",
        )

        session.set_flow("writer -> reviewer")

        assert session.current_flow == "writer -> reviewer"

    def test_export_config(self):
        """Test exporting configuration."""
        mock_team = MagicMock()
        mock_team.team_id = "test_team"

        session = ConfigurationSession(
            session_id="session_123",
            team=mock_team,
            task="Test task",
            constraints={"tone": "formal"},
        )

        session.add_agent(role="writer", type="llm", prompt="Write")
        session.set_flow("writer")

        config = session.export_config()

        assert config["team_id"] == "test_team"
        assert config["task"] == "Test task"
        assert len(config["agents"]) == 1
        assert config["flow"] == "writer"

    def test_repr(self):
        """Test string representation."""
        mock_team = MagicMock()
        mock_team.team_id = "test"

        session = ConfigurationSession(
            session_id="session_123",
            team=mock_team,
            task="Test",
        )

        repr_str = repr(session)

        assert "ConfigurationSession" in repr_str
        assert "session_123" in repr_str
        assert "created" in repr_str

    async def test_analyze(self):
        """Test task analysis."""
        mock_team = MagicMock()
        mock_team.team_id = "test"

        session = ConfigurationSession(
            session_id="session_123",
            team=mock_team,
            task="Generate posts",
        )

        # Mock LLM provider
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value='{"main_goal": "Generate posts", "input_type": "text", "output_type": "text", "sub_tasks": ["write"], "complexity": "simple"}')
        session._llm_provider = mock_llm

        analysis = await session.analyze()

        assert session.state == SessionState.ANALYZING
        assert analysis.main_goal == "Generate posts"

    async def test_suggest(self):
        """Test team suggestion."""
        mock_team = MagicMock()
        mock_team.team_id = "test"

        session = ConfigurationSession(
            session_id="session_123",
            team=mock_team,
            task="Generate posts",
        )

        # Set up analysis
        session.task_analysis = TaskAnalysis(
            main_goal="Generate posts",
            input_type="text",
            output_type="text",
            sub_tasks=["write"],
            complexity="simple",
        )

        # Mock LLM provider
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value='{"agents": [{"role": "writer", "type": "llm", "purpose": "Write", "prompt_template": "Write: {input}", "reasoning": "needed"}], "flow": "writer", "reasoning": "simple task"}')
        session._llm_provider = mock_llm

        result = await session.suggest()

        assert session.state == SessionState.CONFIGURING
        assert len(session.suggested_agents) == 1
        assert session.suggested_agents[0].role == "writer"
        assert session.suggested_flow == "writer"


class TestOrchestratorMode:
    """Tests for CONFIGURATOR mode in OrchestratorMode."""

    def test_configurator_mode_exists(self):
        """Test CONFIGURATOR mode exists."""
        from llmteam.agents.orchestrator import OrchestratorMode

        assert hasattr(OrchestratorMode, "CONFIGURATOR")

    def test_assisted_preset(self):
        """Test ASSISTED preset includes CONFIGURATOR."""
        from llmteam.agents.orchestrator import OrchestratorMode

        assert OrchestratorMode.CONFIGURATOR in OrchestratorMode.ASSISTED
        assert OrchestratorMode.SUPERVISOR in OrchestratorMode.ASSISTED
        assert OrchestratorMode.REPORTER in OrchestratorMode.ASSISTED


class TestLLMTeamConfigure:
    """Tests for LLMTeam.configure() method."""

    async def test_configure_method_exists(self):
        """Test configure method exists on LLMTeam."""
        from llmteam import LLMTeam

        team = LLMTeam(team_id="test")
        assert hasattr(team, "configure")
        assert callable(team.configure)

    async def test_configure_creates_session(self):
        """Test configure creates a session."""
        from llmteam import LLMTeam

        team = LLMTeam(team_id="test")

        # Mock the session methods to avoid actual LLM calls
        with patch.object(ConfigurationSession, 'analyze', new_callable=AsyncMock):
            with patch.object(ConfigurationSession, 'suggest', new_callable=AsyncMock):
                session = await team.configure(
                    task="Test task",
                    constraints={"tone": "formal"},
                )

        assert isinstance(session, ConfigurationSession)
        assert session.task == "Test task"
        assert session.constraints == {"tone": "formal"}
