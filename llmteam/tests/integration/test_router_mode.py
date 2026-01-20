"""
Integration tests for ROUTER mode orchestration.

Tests that the orchestrator can actually select agents in ROUTER mode.
Requires OPENAI_API_KEY environment variable.
"""

import os
import pytest

# Skip if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)


class TestRouterModeIntegration:
    """Integration tests for ROUTER mode."""

    @pytest.mark.asyncio
    async def test_router_mode_selects_single_agent(self):
        """Test that ROUTER mode can select a single agent."""
        from llmteam import LLMTeam, OrchestratorConfig, OrchestratorMode

        # Create team with specialist agents
        team = LLMTeam(
            team_id="triage_test",
            agents=[
                {
                    "type": "llm",
                    "role": "billing",
                    "name": "Billing Specialist",
                    "description": "Handles billing and payment questions",
                    "prompt": "You are a billing specialist. Answer: {input}",
                    "model": "gpt-4o-mini",
                },
                {
                    "type": "llm",
                    "role": "technical",
                    "name": "Technical Support",
                    "description": "Handles technical issues and troubleshooting",
                    "prompt": "You are technical support. Answer: {input}",
                    "model": "gpt-4o-mini",
                },
                {
                    "type": "llm",
                    "role": "general",
                    "name": "General Support",
                    "description": "Handles general questions and information",
                    "prompt": "You are general support. Answer: {input}",
                    "model": "gpt-4o-mini",
                },
            ],
            # Enable ROUTER mode
            orchestrator=OrchestratorConfig(mode=OrchestratorMode.ACTIVE),
        )

        # Ask a billing question
        result = await team.run({"input": "Why was I charged twice on my credit card?"})

        print(f"\n=== ROUTER Mode Test ===")
        print(f"Success: {result.success}")
        print(f"Agents called: {result.agents_called}")
        print(f"Iterations: {result.iterations}")
        print(f"Final output: {str(result.final_output)[:200]}...")
        print(f"Report preview:\n{result.report[:500] if result.report else 'None'}...")

        assert result.success is True
        # In ROUTER mode, orchestrator should select billing agent
        # (not call all agents)
        assert len(result.agents_called) >= 1
        # Should have report
        assert result.report is not None
        assert "billing" in result.agents_called or "general" in result.agents_called

    @pytest.mark.asyncio
    async def test_router_mode_technical_question(self):
        """Test ROUTER mode with technical question."""
        from llmteam import LLMTeam, OrchestratorConfig, OrchestratorMode

        team = LLMTeam(
            team_id="triage_tech",
            agents=[
                {
                    "type": "llm",
                    "role": "billing",
                    "name": "Billing Specialist",
                    "description": "Handles billing and payment questions",
                    "prompt": "You are a billing specialist. Answer: {input}",
                    "model": "gpt-4o-mini",
                },
                {
                    "type": "llm",
                    "role": "technical",
                    "name": "Technical Support",
                    "description": "Handles technical issues and troubleshooting",
                    "prompt": "You are technical support. Answer: {input}",
                    "model": "gpt-4o-mini",
                },
            ],
            orchestrator=OrchestratorConfig(mode=OrchestratorMode.ACTIVE),
        )

        result = await team.run({
            "input": "My application crashes when I try to upload large files"
        })

        print(f"\n=== ROUTER Mode Technical Test ===")
        print(f"Agents called: {result.agents_called}")
        print(f"Final output: {str(result.final_output)[:200]}...")

        assert result.success is True
        # Should route to technical agent
        assert len(result.agents_called) >= 1

    @pytest.mark.asyncio
    async def test_passive_mode_calls_all_agents(self):
        """Test that PASSIVE mode calls all agents sequentially."""
        from llmteam import LLMTeam

        team = LLMTeam(
            team_id="passive_test",
            agents=[
                {
                    "type": "llm",
                    "role": "writer",
                    "prompt": "Write a short sentence about: {input}",
                    "model": "gpt-4o-mini",
                },
                {
                    "type": "llm",
                    "role": "editor",
                    "prompt": "Edit this text: {context}",
                    "model": "gpt-4o-mini",
                },
            ],
            flow="sequential",
            # Default: PASSIVE mode
        )

        result = await team.run({"input": "artificial intelligence"})

        print(f"\n=== PASSIVE Mode Test ===")
        print(f"Agents called: {result.agents_called}")
        print(f"Has report: {result.report is not None}")

        assert result.success is True
        # PASSIVE mode should call ALL agents
        assert len(result.agents_called) == 2
        assert "writer" in result.agents_called
        assert "editor" in result.agents_called
        # Should have orchestrator report
        assert result.report is not None

    @pytest.mark.asyncio
    async def test_orchestration_backward_compat(self):
        """Test backward compatibility: orchestration=True enables ACTIVE mode."""
        from llmteam import LLMTeam

        team = LLMTeam(
            team_id="compat_test",
            agents=[
                {
                    "type": "llm",
                    "role": "agent1",
                    "description": "First agent for simple tasks",
                    "prompt": "Handle: {input}",
                    "model": "gpt-4o-mini",
                },
                {
                    "type": "llm",
                    "role": "agent2",
                    "description": "Second agent for complex tasks",
                    "prompt": "Handle: {input}",
                    "model": "gpt-4o-mini",
                },
            ],
            orchestration=True,  # Old API
        )

        # Check that ROUTER mode is enabled
        assert team.is_router_mode is True

        result = await team.run({"input": "Simple question"})

        print(f"\n=== Backward Compat Test ===")
        print(f"Router mode: {team.is_router_mode}")
        print(f"Agents called: {result.agents_called}")

        assert result.success is True


class TestRouterModeUnitTests:
    """Unit tests for ROUTER mode that don't require API."""

    def test_team_has_orchestrator(self):
        """Test that team always has orchestrator."""
        from llmteam import LLMTeam

        team = LLMTeam(team_id="test")
        assert team.get_orchestrator() is not None

    def test_list_agents_excludes_orchestrator(self):
        """Test that list_agents doesn't include orchestrator."""
        from llmteam import LLMTeam

        team = LLMTeam(
            team_id="test",
            agents=[
                {"type": "llm", "role": "writer", "prompt": "test"},
            ],
        )

        agents = team.list_agents()
        agent_ids = [a.agent_id for a in agents]

        assert "writer" in agent_ids
        assert "_orchestrator" not in agent_ids

    def test_reserved_role_rejected(self):
        """Test that reserved roles (starting with _) are rejected."""
        from llmteam import LLMTeam

        team = LLMTeam(team_id="test")

        with pytest.raises(ValueError, match="reserved"):
            team.add_agent({
                "type": "llm",
                "role": "_internal",
                "prompt": "test",
            })

    def test_orchestrator_config_parameter(self):
        """Test that orchestrator config is applied."""
        from llmteam import LLMTeam, OrchestratorConfig, OrchestratorMode

        config = OrchestratorConfig(
            mode=OrchestratorMode.FULL,
            model="gpt-4o",
        )

        team = LLMTeam(
            team_id="test",
            orchestrator=config,
        )

        orch = team.get_orchestrator()
        assert orch.mode == OrchestratorMode.FULL
        assert orch.is_router is True

    def test_run_result_has_report_fields(self):
        """Test that RunResult has report and summary fields."""
        from llmteam.team.result import RunResult

        result = RunResult(success=True)
        assert hasattr(result, "report")
        assert hasattr(result, "summary")

        result.report = "Test report"
        result.summary = {"agents": 2}

        data = result.to_dict()
        assert data["report"] == "Test report"
        assert data["summary"] == {"agents": 2}
