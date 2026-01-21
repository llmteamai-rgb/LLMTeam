"""
ConfigurationSession for CONFIGURATOR mode (RFC-005).

Provides an interactive session for configuring LLMTeam via LLM assistance.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from llmteam.configuration.models import (
    AgentSuggestion,
    PipelinePreview,
    SessionState,
    TaskAnalysis,
    TestRunResult,
)
from llmteam.configuration.prompts import ConfiguratorPrompts

if TYPE_CHECKING:
    from llmteam.team import LLMTeam
    from llmteam.runtime import LLMProvider
    from llmteam.quality import QualityManager


@dataclass
class ConfigurationSession:
    """
    Session for configuring a team via CONFIGURATOR mode.

    RFC-005: Allows interactive team configuration with LLM assistance.

    Usage:
        team = LLMTeam(team_id="content")

        session = await team.configure(
            task="Generate LinkedIn posts from press releases"
        )

        # Review suggestions
        print(session.suggested_agents)

        # Test
        test = await session.test_run({"press_release": "..."})
        print(test.analysis)

        # Apply
        await session.apply()
    """

    session_id: str
    """Unique session identifier."""

    team: "LLMTeam"
    """The team being configured."""

    task: str
    """Task description."""

    constraints: Dict[str, Any] = field(default_factory=dict)
    """Task constraints (tone, length, format, etc.)."""

    # Analysis
    task_analysis: Optional[TaskAnalysis] = None
    """Analysis of the task."""

    # Suggestions
    suggested_agents: List[AgentSuggestion] = field(default_factory=list)
    """Suggested agent configurations."""

    suggested_flow: Optional[str] = None
    """Suggested agent flow."""

    suggestion_reasoning: str = ""
    """Reasoning behind suggestions."""

    # Current configuration (after edits)
    current_agents: List[Dict[str, Any]] = field(default_factory=list)
    """Current agent configurations."""

    current_flow: Optional[str] = None
    """Current flow configuration."""

    # Test runs
    test_runs: List[TestRunResult] = field(default_factory=list)
    """History of test runs."""

    # State
    state: SessionState = SessionState.CREATED
    """Current session state."""

    # Quality (RFC-008)
    _quality: Optional[int] = None
    """Quality level override (0-100). None = use team default."""

    # LLM provider (for configuration assistance)
    _llm_provider: Optional["LLMProvider"] = field(default=None, repr=False)

    created_at: datetime = field(default_factory=datetime.utcnow)
    """When the session was created."""

    # === Lifecycle Methods ===

    async def analyze(self) -> TaskAnalysis:
        """
        Analyze the task via LLM.

        Returns:
            TaskAnalysis with extracted information.
        """
        self.state = SessionState.ANALYZING

        llm = self._get_llm()
        prompt = ConfiguratorPrompts.TASK_ANALYSIS.format(
            task=self.task,
            constraints=json.dumps(self.constraints, ensure_ascii=False),
        )

        response = await llm.complete(prompt)
        data = self._parse_json(response)

        self.task_analysis = TaskAnalysis(
            main_goal=data.get("main_goal", ""),
            input_type=data.get("input_type", ""),
            output_type=data.get("output_type", ""),
            sub_tasks=data.get("sub_tasks", []),
            complexity=data.get("complexity", "moderate"),
            raw_analysis=response,
        )

        return self.task_analysis

    async def suggest(self) -> Dict[str, Any]:
        """
        Suggest team configuration via LLM.

        Returns:
            Dictionary with agents, flow, and reasoning.
        """
        self.state = SessionState.SUGGESTING

        if not self.task_analysis:
            await self.analyze()

        llm = self._get_llm()
        prompt = ConfiguratorPrompts.TEAM_SUGGESTION.format(
            task_analysis=json.dumps(self.task_analysis.to_dict(), indent=2, ensure_ascii=False),
        )

        response = await llm.complete(prompt)
        data = self._parse_json(response)

        # Parse agent suggestions
        self.suggested_agents = [
            AgentSuggestion.from_dict(agent)
            for agent in data.get("agents", [])
        ]

        self.suggested_flow = data.get("flow")
        self.suggestion_reasoning = data.get("reasoning", "")

        # Initialize current config from suggestions
        self.current_agents = [
            {
                "role": agent.role,
                "type": agent.type,
                "prompt": agent.prompt_template,
            }
            for agent in self.suggested_agents
        ]
        self.current_flow = self.suggested_flow

        self.state = SessionState.CONFIGURING

        return {
            "agents": [a.to_dict() for a in self.suggested_agents],
            "flow": self.suggested_flow,
            "reasoning": self.suggestion_reasoning,
        }

    # === Configuration Methods ===

    def add_agent(
        self,
        role: str,
        type: str,
        prompt: str,
        **config: Any,
    ) -> "ConfigurationSession":
        """
        Add an agent to the configuration.

        Args:
            role: Agent role name.
            type: Agent type ('llm', 'rag', 'kag').
            prompt: Agent prompt.
            **config: Additional configuration.

        Returns:
            Self for chaining.
        """
        agent_config = {
            "role": role,
            "type": type,
            "prompt": prompt,
            **config,
        }
        self.current_agents.append(agent_config)
        return self

    def modify_agent(self, role: str, **changes: Any) -> "ConfigurationSession":
        """
        Modify an existing agent configuration.

        Args:
            role: Agent role to modify.
            **changes: Fields to change.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If agent not found.
        """
        for agent in self.current_agents:
            if agent.get("role") == role:
                agent.update(changes)
                return self

        raise ValueError(f"Agent with role '{role}' not found")

    def remove_agent(self, role: str) -> "ConfigurationSession":
        """
        Remove an agent from the configuration.

        Args:
            role: Agent role to remove.

        Returns:
            Self for chaining.
        """
        self.current_agents = [
            a for a in self.current_agents if a.get("role") != role
        ]
        return self

    def set_flow(self, flow: str) -> "ConfigurationSession":
        """
        Set the agent execution flow.

        Args:
            flow: Flow string (e.g., "agent1 -> agent2 -> agent3").

        Returns:
            Self for chaining.
        """
        self.current_flow = flow
        return self

    # === Quality Methods (RFC-008) ===

    def set_quality(self, quality: int) -> "ConfigurationSession":
        """
        Set quality level for the configuration (RFC-008).

        Args:
            quality: Quality level (0-100).
                     0-30: Fast & cheap, basic output
                     30-70: Balanced, good for most tasks
                     70-100: Best quality, thorough analysis

        Returns:
            Self for chaining.
        """
        self._quality = max(0, min(100, int(quality)))
        return self

    @property
    def quality(self) -> int:
        """Get effective quality level."""
        if self._quality is not None:
            return self._quality
        return self.team.quality

    async def preview(self) -> PipelinePreview:
        """
        Preview the pipeline configuration with cost estimates (RFC-008).

        Returns:
            PipelinePreview with agents, flow, and cost estimates.

        Example:
            session.set_quality(60)
            preview = await session.preview()
            print(preview)
            # Pipeline Preview (quality=60):
            # Agents:
            #   1. extractor (gpt-4o-mini) — Extract key metrics...
            # Estimated cost: $0.15-0.20 per run
        """
        from llmteam.quality import QualityManager, CostEstimator

        quality = self.quality
        manager = QualityManager(quality)
        estimator = CostEstimator()

        # Determine models for agents based on quality
        agents_with_models = []
        complexity = self.task_analysis.complexity if self.task_analysis else "medium"

        for agent in self.current_agents:
            agent_copy = agent.copy()
            if "model" not in agent_copy:
                agent_copy["model"] = manager.get_model(complexity)
            agents_with_models.append(agent_copy)

        # Estimate cost
        if agents_with_models:
            estimate = estimator.estimate_detailed(
                quality=quality,
                agents=agents_with_models,
            )
        else:
            estimate = estimator.estimate(quality=quality, complexity=complexity)

        # Calculate quality rating
        if quality >= 85:
            stars, label = 5, "Excellent"
        elif quality >= 70:
            stars, label = 4, "Good"
        elif quality >= 50:
            stars, label = 3, "Adequate"
        elif quality >= 30:
            stars, label = 2, "Basic"
        else:
            stars, label = 1, "Minimal"

        return PipelinePreview(
            quality=quality,
            agents=agents_with_models,
            flow=self.current_flow,
            estimated_cost_min=estimate.min_cost,
            estimated_cost_max=estimate.max_cost,
            quality_stars=stars,
            quality_label=label,
        )

    async def generate_pipeline(self) -> Dict[str, Any]:
        """
        Generate optimized pipeline based on quality level (RFC-008).

        Adjusts number of agents and models based on quality slider.

        Returns:
            Pipeline configuration dict.
        """
        from llmteam.quality import QualityManager

        quality = self.quality
        manager = QualityManager(quality)

        # Get recommended pipeline depth
        depth = manager.get_pipeline_depth()
        min_agents, max_agents = manager.get_agent_count_range()

        # Adjust current agents to match recommended depth
        complexity = self.task_analysis.complexity if self.task_analysis else "medium"

        # Update models based on quality
        optimized_agents = []
        for agent in self.current_agents[:max_agents]:
            agent_copy = agent.copy()
            agent_copy["model"] = manager.get_model(complexity)
            # Apply generation params
            gen_params = manager.get_generation_params()
            if "max_tokens" not in agent_copy:
                agent_copy["max_tokens"] = gen_params["max_tokens"]
            if "temperature" not in agent_copy:
                agent_copy["temperature"] = gen_params["temperature"]
            optimized_agents.append(agent_copy)

        # Estimate cost
        estimate = manager.estimate_cost(complexity)

        return {
            "agents": optimized_agents,
            "flow": self.current_flow,
            "quality": quality,
            "estimated_cost": {
                "min": estimate[0],
                "max": estimate[1],
            },
            "pipeline_depth": depth.value,
        }

    # === Testing Methods ===

    async def test_run(self, input_data: Dict[str, Any]) -> TestRunResult:
        """
        Execute a test run with the current configuration.

        Args:
            input_data: Test input data.

        Returns:
            TestRunResult with execution results and analysis.
        """
        self.state = SessionState.TESTING

        test_id = f"test_{uuid.uuid4().hex[:8]}"
        start_time = datetime.utcnow()

        # Build temporary team with current config
        temp_team = self._build_temp_team()

        try:
            # Run the team
            result = await temp_team.run(input_data)

            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            # Get agent outputs
            agent_outputs = {}
            if hasattr(result, "steps") and result.steps:
                for step in result.steps:
                    if hasattr(step, "agent_role") and hasattr(step, "output"):
                        agent_outputs[step.agent_role] = step.output

            # Analyze the test run
            analysis_data = await self._analyze_test(
                input_data=input_data,
                output=result.output if hasattr(result, "output") else result.final_output,
                agent_outputs=agent_outputs,
                duration_ms=duration_ms,
            )

            test_result = TestRunResult(
                test_id=test_id,
                input_data=input_data,
                output=result.output if hasattr(result, "output") else {},
                agent_outputs=agent_outputs,
                duration_ms=duration_ms,
                success=result.success if hasattr(result, "success") else True,
                analysis=analysis_data.get("summary", ""),
                issues=analysis_data.get("issues", []),
                recommendations=analysis_data.get("recommendations", []),
                ready_for_production=analysis_data.get("ready_for_production", False),
            )

        except Exception as e:
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            test_result = TestRunResult(
                test_id=test_id,
                input_data=input_data,
                output={},
                agent_outputs={},
                duration_ms=duration_ms,
                success=False,
                analysis=f"Test failed with error: {str(e)}",
                issues=[str(e)],
                recommendations=["Fix the error before retrying"],
                ready_for_production=False,
            )

        self.test_runs.append(test_result)

        # Update state based on result
        if test_result.ready_for_production:
            self.state = SessionState.READY
        else:
            self.state = SessionState.CONFIGURING

        return test_result

    async def _analyze_test(
        self,
        input_data: Dict[str, Any],
        output: Any,
        agent_outputs: Dict[str, Any],
        duration_ms: int,
    ) -> Dict[str, Any]:
        """Analyze test run via LLM."""
        llm = self._get_llm()

        prompt = ConfiguratorPrompts.TEST_ANALYSIS.format(
            team_config=json.dumps(self.export_config(), indent=2, ensure_ascii=False),
            test_input=json.dumps(input_data, indent=2, ensure_ascii=False),
            agent_outputs=json.dumps(agent_outputs, indent=2, ensure_ascii=False),
            final_output=json.dumps(output, indent=2, ensure_ascii=False) if isinstance(output, dict) else str(output),
            duration_ms=duration_ms,
        )

        response = await llm.complete(prompt)
        return self._parse_json(response)

    # === Application Methods ===

    async def apply(self) -> None:
        """
        Apply the current configuration to the team.

        This modifies the actual LLMTeam instance.
        """
        # Clear existing agents
        for agent_id in list(self.team._agents.keys()):
            self.team.remove_agent(agent_id)

        # Add configured agents
        for agent_config in self.current_agents:
            self.team.add_agent(agent_config)

        # Set flow if specified
        if self.current_flow:
            self.team.set_flow(self.current_flow)

        self.state = SessionState.APPLIED

    def export_config(self) -> Dict[str, Any]:
        """
        Export the current configuration.

        Returns:
            Dictionary with team configuration.
        """
        return {
            "team_id": self.team.team_id,
            "task": self.task,
            "constraints": self.constraints,
            "agents": self.current_agents,
            "flow": self.current_flow,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
        }

    # === Helper Methods ===

    def _get_llm(self) -> "LLMProvider":
        """Get LLM provider for configuration."""
        if self._llm_provider:
            return self._llm_provider

        # Try to get from team's runtime context
        if hasattr(self.team, "_runtime") and self.team._runtime:
            try:
                return self.team._runtime.get_llm("default")
            except Exception:
                pass

        # Create mock provider for testing
        from llmteam.testing.mocks import MockLLMProvider
        return MockLLMProvider(responses=[
            # Default responses for testing
            '{"main_goal": "test", "input_type": "text", "output_type": "text", "sub_tasks": ["process"], "complexity": "simple"}',
            '{"agents": [{"role": "worker", "type": "llm", "purpose": "process", "prompt_template": "Process: {input}", "reasoning": "needed"}], "flow": "worker", "reasoning": "simple"}',
            '{"overall": "success", "issues": [], "recommendations": [], "ready_for_production": true, "summary": "Good"}',
        ])

    def _build_temp_team(self) -> "LLMTeam":
        """Build a temporary team with current config for testing."""
        from llmteam.team import LLMTeam

        temp_team = LLMTeam(team_id=f"{self.team.team_id}_test")

        for agent_config in self.current_agents:
            temp_team.add_agent(agent_config)

        if self.current_flow:
            temp_team.set_flow(self.current_flow)

        return temp_team

    def _parse_json(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        # Try to extract JSON from response
        response = response.strip()

        # Handle markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end].strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON object in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(response[start:end])
                except json.JSONDecodeError:
                    pass

            # Return empty dict if parsing fails
            return {}

    def __repr__(self) -> str:
        return (
            f"<ConfigurationSession "
            f"id='{self.session_id}' "
            f"state={self.state.value} "
            f"agents={len(self.current_agents)}>"
        )
