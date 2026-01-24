"""
Tests for RFC-021: DynamicTeamBuilder.

All tests use mock LLM providers to avoid real API calls.
"""

import json
import pytest
from unittest.mock import AsyncMock, patch

from llmteam.builder import (
    DynamicTeamBuilder,
    TeamBlueprint,
    AgentBlueprint,
    TOOL_MAP,
    BuilderParseError,
    BuilderValidationError,
)
from llmteam.builder.dynamic import (
    _strip_markdown_fences,
    _parse_blueprint_json,
    _validate_blueprint,
)


# --- Fixtures ---

VALID_BLUEPRINT_JSON = json.dumps({
    "team_id": "research-team",
    "description": "Research and summarize information",
    "agents": [
        {
            "role": "researcher",
            "purpose": "Search for information",
            "prompt": "Research the following topic: {query}",
            "tools": ["web_search"],
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "max_tool_rounds": 5,
        },
        {
            "role": "summarizer",
            "purpose": "Summarize findings",
            "prompt": "Summarize the following: {input}",
            "tools": ["text_summarize"],
            "model": "gpt-4o-mini",
            "temperature": 0.5,
            "max_tool_rounds": 2,
        },
    ],
    "routing_strategy": "research tasks to researcher, summaries to summarizer",
    "input_variables": ["query"],
})


VALID_BLUEPRINT_WITH_FENCES = f"```json\n{VALID_BLUEPRINT_JSON}\n```"


# --- Test: Parse Blueprint ---

class TestParseBlueprintValid:
    """Test parsing of valid blueprint JSON."""

    def test_parse_valid_json(self):
        data = _parse_blueprint_json(VALID_BLUEPRINT_JSON)
        assert data["team_id"] == "research-team"
        assert len(data["agents"]) == 2
        assert data["agents"][0]["role"] == "researcher"

    def test_parse_returns_dict(self):
        data = _parse_blueprint_json(VALID_BLUEPRINT_JSON)
        assert isinstance(data, dict)

    def test_parse_preserves_all_fields(self):
        data = _parse_blueprint_json(VALID_BLUEPRINT_JSON)
        assert data["description"] == "Research and summarize information"
        assert data["routing_strategy"] == "research tasks to researcher, summaries to summarizer"
        assert data["input_variables"] == ["query"]


class TestParseBlueprintMarkdownFences:
    """Test stripping markdown code fences."""

    def test_strip_json_fences(self):
        result = _strip_markdown_fences('```json\n{"key": "value"}\n```')
        assert result == '{"key": "value"}'

    def test_strip_plain_fences(self):
        result = _strip_markdown_fences('```\n{"key": "value"}\n```')
        assert result == '{"key": "value"}'

    def test_no_fences_unchanged(self):
        raw = '{"key": "value"}'
        result = _strip_markdown_fences(raw)
        assert result == raw

    def test_parse_with_fences(self):
        data = _parse_blueprint_json(VALID_BLUEPRINT_WITH_FENCES)
        assert data["team_id"] == "research-team"
        assert len(data["agents"]) == 2

    def test_strip_with_whitespace(self):
        result = _strip_markdown_fences('  ```json\n{"a": 1}\n```  ')
        assert result == '{"a": 1}'


class TestParseBlueprintInvalid:
    """Test error handling for invalid JSON."""

    def test_empty_string(self):
        with pytest.raises(BuilderParseError):
            _parse_blueprint_json("")

    def test_not_json(self):
        with pytest.raises(BuilderParseError):
            _parse_blueprint_json("This is not JSON at all")

    def test_truncated_json(self):
        with pytest.raises(BuilderParseError):
            _parse_blueprint_json('{"team_id": "test", "agents": [')

    def test_error_message_includes_context(self):
        with pytest.raises(BuilderParseError, match="Failed to parse"):
            _parse_blueprint_json("invalid json here")


# --- Test: Validate Blueprint ---

class TestValidateToolsFilter:
    """Test tool name filtering during validation."""

    def test_valid_tools_kept(self):
        data = {
            "team_id": "test",
            "agents": [{
                "role": "agent1",
                "purpose": "test",
                "prompt": "test",
                "tools": ["web_search", "code_eval"],
            }],
        }
        blueprint = _validate_blueprint(data)
        assert blueprint.agents[0].tools == ["web_search", "code_eval"]

    def test_unknown_tools_filtered(self):
        data = {
            "team_id": "test",
            "agents": [{
                "role": "agent1",
                "purpose": "test",
                "prompt": "test",
                "tools": ["web_search", "unknown_tool", "magic_spell"],
            }],
        }
        with pytest.warns(UserWarning):
            blueprint = _validate_blueprint(data)
        assert blueprint.agents[0].tools == ["web_search"]

    def test_all_unknown_tools_result_empty(self):
        data = {
            "team_id": "test",
            "agents": [{
                "role": "agent1",
                "purpose": "test",
                "prompt": "test",
                "tools": ["fake_tool"],
            }],
        }
        with pytest.warns(UserWarning):
            blueprint = _validate_blueprint(data)
        assert blueprint.agents[0].tools == []

    def test_no_tools_is_valid(self):
        data = {
            "team_id": "test",
            "agents": [{
                "role": "agent1",
                "purpose": "test",
                "prompt": "test",
                "tools": [],
            }],
        }
        blueprint = _validate_blueprint(data)
        assert blueprint.agents[0].tools == []


class TestValidateRolesUnique:
    """Test role uniqueness validation."""

    def test_duplicate_roles_error(self):
        data = {
            "team_id": "test",
            "agents": [
                {"role": "writer", "purpose": "a", "prompt": "a", "tools": []},
                {"role": "writer", "purpose": "b", "prompt": "b", "tools": []},
            ],
        }
        with pytest.raises(BuilderValidationError, match="Duplicate role"):
            _validate_blueprint(data)

    def test_reserved_role_error(self):
        data = {
            "team_id": "test",
            "agents": [
                {"role": "_internal", "purpose": "a", "prompt": "a", "tools": []},
            ],
        }
        with pytest.raises(BuilderValidationError, match="reserved"):
            _validate_blueprint(data)

    def test_unique_roles_pass(self):
        data = {
            "team_id": "test",
            "agents": [
                {"role": "writer", "purpose": "a", "prompt": "a", "tools": []},
                {"role": "editor", "purpose": "b", "prompt": "b", "tools": []},
            ],
        }
        blueprint = _validate_blueprint(data)
        assert len(blueprint.agents) == 2

    def test_empty_agents_error(self):
        data = {"team_id": "test", "agents": []}
        with pytest.raises(BuilderValidationError, match="at least 1"):
            _validate_blueprint(data)

    def test_too_many_agents_error(self):
        data = {
            "team_id": "test",
            "agents": [
                {"role": f"agent{i}", "purpose": "x", "prompt": "x", "tools": []}
                for i in range(6)
            ],
        }
        with pytest.raises(BuilderValidationError, match="max 5"):
            _validate_blueprint(data)


class TestValidateTemperatureAndRounds:
    """Test temperature and max_tool_rounds validation."""

    def test_temperature_too_high(self):
        data = {
            "team_id": "test",
            "agents": [{
                "role": "agent1",
                "purpose": "test",
                "prompt": "test",
                "tools": [],
                "temperature": 2.5,
            }],
        }
        with pytest.raises(BuilderValidationError, match="temperature"):
            _validate_blueprint(data)

    def test_temperature_negative(self):
        data = {
            "team_id": "test",
            "agents": [{
                "role": "agent1",
                "purpose": "test",
                "prompt": "test",
                "tools": [],
                "temperature": -0.1,
            }],
        }
        with pytest.raises(BuilderValidationError, match="temperature"):
            _validate_blueprint(data)

    def test_max_tool_rounds_zero(self):
        data = {
            "team_id": "test",
            "agents": [{
                "role": "agent1",
                "purpose": "test",
                "prompt": "test",
                "tools": [],
                "max_tool_rounds": 0,
            }],
        }
        with pytest.raises(BuilderValidationError, match="max_tool_rounds"):
            _validate_blueprint(data)

    def test_max_tool_rounds_too_high(self):
        data = {
            "team_id": "test",
            "agents": [{
                "role": "agent1",
                "purpose": "test",
                "prompt": "test",
                "tools": [],
                "max_tool_rounds": 11,
            }],
        }
        with pytest.raises(BuilderValidationError, match="max_tool_rounds"):
            _validate_blueprint(data)


# --- Test: Build Team ---

class TestBuildTeamFromBlueprint:
    """Test building LLMTeam from blueprint."""

    def test_build_creates_team(self):
        blueprint = TeamBlueprint(
            team_id="test-team",
            description="Test team",
            agents=[
                AgentBlueprint(
                    role="writer",
                    purpose="Write content",
                    prompt="Write about: {query}",
                    tools=[],
                    model="gpt-4o-mini",
                ),
            ],
            routing_strategy="all to writer",
            input_variables=["query"],
        )
        builder = DynamicTeamBuilder(verbose=False)
        team = builder.build_team(blueprint)

        assert team.team_id == "test-team"
        assert len(team.list_agents()) == 1
        assert team.is_router_mode is True

    def test_build_multiple_agents(self):
        blueprint = TeamBlueprint(
            team_id="multi-team",
            description="Multi agent team",
            agents=[
                AgentBlueprint(role="a", purpose="A", prompt="A: {x}", tools=[]),
                AgentBlueprint(role="b", purpose="B", prompt="B: {x}", tools=[]),
                AgentBlueprint(role="c", purpose="C", prompt="C: {x}", tools=[]),
            ],
            routing_strategy="route by type",
            input_variables=["x"],
        )
        builder = DynamicTeamBuilder(verbose=False)
        team = builder.build_team(blueprint)

        assert len(team.list_agents()) == 3

    def test_build_empty_blueprint_error(self):
        blueprint = TeamBlueprint(
            team_id="empty",
            description="Empty",
            agents=[],
        )
        builder = DynamicTeamBuilder(verbose=False)
        with pytest.raises(BuilderValidationError):
            builder.build_team(blueprint)


class TestBuildTeamToolsAttached:
    """Test that tools are attached to agents correctly."""

    def test_tools_on_agent(self):
        blueprint = TeamBlueprint(
            team_id="tools-team",
            description="Team with tools",
            agents=[
                AgentBlueprint(
                    role="researcher",
                    purpose="Research",
                    prompt="Research: {query}",
                    tools=["web_search", "http_fetch"],
                ),
            ],
            input_variables=["query"],
        )
        builder = DynamicTeamBuilder(verbose=False)
        team = builder.build_team(blueprint)

        agents = team.list_agents()
        assert len(agents) == 1
        agent = agents[0]
        # Agent should have tool executor with the tools
        assert agent._tool_executor is not None
        schemas = agent._tool_executor.get_schemas()
        tool_names = [s["function"]["name"] for s in schemas]
        assert "web_search" in tool_names
        assert "http_fetch" in tool_names

    def test_no_tools_agent(self):
        blueprint = TeamBlueprint(
            team_id="no-tools",
            description="No tools",
            agents=[
                AgentBlueprint(
                    role="thinker",
                    purpose="Think",
                    prompt="Think about: {query}",
                    tools=[],
                ),
            ],
            input_variables=["query"],
        )
        builder = DynamicTeamBuilder(verbose=False)
        team = builder.build_team(blueprint)

        agents = team.list_agents()
        agent = agents[0]
        # No tools = no executor or empty executor
        if agent._tool_executor is not None:
            schemas = agent._tool_executor.get_schemas()
            assert len(schemas) == 0


# --- Test: Refine Blueprint ---

class TestRefineBlueprint:
    """Test blueprint refinement with mock LLM."""

    @pytest.mark.asyncio
    async def test_refine_changes_agents(self):
        """LLM returns updated blueprint with different agents."""
        refined_json = json.dumps({
            "team_id": "refined-team",
            "description": "Refined team",
            "agents": [
                {
                    "role": "analyst",
                    "purpose": "Analyze data",
                    "prompt": "Analyze: {query}",
                    "tools": ["code_eval"],
                    "temperature": 0.2,
                    "max_tool_rounds": 3,
                },
            ],
            "routing_strategy": "all to analyst",
            "input_variables": ["query"],
        })

        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=refined_json)

        builder = DynamicTeamBuilder(verbose=False, provider=mock_provider)

        original = TeamBlueprint(
            team_id="original",
            description="Original",
            agents=[
                AgentBlueprint(role="writer", purpose="Write", prompt="Write: {query}", tools=[]),
            ],
            input_variables=["query"],
        )

        refined = await builder.refine_blueprint(original, "Replace writer with analyst that uses code_eval")

        assert refined.team_id == "refined-team"
        assert len(refined.agents) == 1
        assert refined.agents[0].role == "analyst"
        assert refined.agents[0].tools == ["code_eval"]


# --- Test: Analyze Task ---

class TestAnalyzeTask:
    """Test full analyze_task flow with mock LLM."""

    @pytest.mark.asyncio
    async def test_analyze_returns_blueprint(self):
        """analyze_task returns a valid TeamBlueprint."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=VALID_BLUEPRINT_JSON)

        builder = DynamicTeamBuilder(verbose=False, provider=mock_provider)
        blueprint = await builder.analyze_task("Research AI trends and summarize")

        assert isinstance(blueprint, TeamBlueprint)
        assert blueprint.team_id == "research-team"
        assert len(blueprint.agents) == 2
        assert blueprint.agents[0].role == "researcher"
        assert blueprint.agents[0].tools == ["web_search"]
        assert blueprint.agents[1].role == "summarizer"

    @pytest.mark.asyncio
    async def test_analyze_calls_provider(self):
        """analyze_task makes an LLM call with the task description."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=VALID_BLUEPRINT_JSON)

        builder = DynamicTeamBuilder(verbose=False, provider=mock_provider)
        await builder.analyze_task("Build a chatbot")

        mock_provider.complete.assert_called_once()
        call_args = mock_provider.complete.call_args
        prompt = call_args[0][0]
        assert "Build a chatbot" in prompt

    @pytest.mark.asyncio
    async def test_analyze_handles_fences(self):
        """analyze_task handles LLM output wrapped in markdown fences."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=VALID_BLUEPRINT_WITH_FENCES)

        builder = DynamicTeamBuilder(verbose=False, provider=mock_provider)
        blueprint = await builder.analyze_task("Research task")

        assert isinstance(blueprint, TeamBlueprint)
        assert blueprint.team_id == "research-team"


# --- Test: TOOL_MAP ---

class TestToolMap:
    """Test the TOOL_MAP constant."""

    def test_all_five_tools_present(self):
        assert len(TOOL_MAP) == 5
        assert "web_search" in TOOL_MAP
        assert "http_fetch" in TOOL_MAP
        assert "json_extract" in TOOL_MAP
        assert "text_summarize" in TOOL_MAP
        assert "code_eval" in TOOL_MAP

    def test_values_are_tool_definitions(self):
        from llmteam.tools import ToolDefinition
        for name, td in TOOL_MAP.items():
            assert isinstance(td, ToolDefinition), f"{name} is not ToolDefinition"
            assert td.name == name
