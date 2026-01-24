"""
DynamicTeamBuilder - Automatic team creation from task description.

RFC-021: LLM analyzes a user's task, optionally asks clarifying questions,
builds a team of agents with appropriate tools, and executes it.

Usage:
    from llmteam.builder import DynamicTeamBuilder

    builder = DynamicTeamBuilder(model="gpt-4o-mini")
    blueprint = await builder.analyze_task("Research AI trends and summarize findings")
    team = builder.build_team(blueprint)
    await builder.execute(team, {"query": "Latest LLM breakthroughs"})
"""

import json
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llmteam.tools.builtin import (
    web_search,
    http_fetch,
    json_extract,
    text_summarize,
    code_eval,
)
from llmteam.tools import ToolDefinition
from llmteam.builder.prompts import TASK_ANALYSIS, CLARIFYING_QUESTIONS, REFINE_BLUEPRINT


class BuilderError(Exception):
    """Base error for DynamicTeamBuilder."""
    pass


class BuilderParseError(BuilderError):
    """LLM returned invalid or unparseable JSON."""
    pass


class BuilderValidationError(BuilderError):
    """Blueprint failed validation."""
    pass


# Map of tool names to their ToolDefinition objects
TOOL_MAP: Dict[str, ToolDefinition] = {
    "web_search": web_search.tool_definition,
    "http_fetch": http_fetch.tool_definition,
    "json_extract": json_extract.tool_definition,
    "text_summarize": text_summarize.tool_definition,
    "code_eval": code_eval.tool_definition,
}


@dataclass
class AgentBlueprint:
    """Blueprint for a single agent in the team."""

    role: str
    purpose: str
    prompt: str
    tools: List[str] = field(default_factory=list)
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tool_rounds: int = 5


@dataclass
class TeamBlueprint:
    """Blueprint for the entire team."""

    team_id: str
    description: str
    agents: List[AgentBlueprint] = field(default_factory=list)
    routing_strategy: str = ""
    input_variables: List[str] = field(default_factory=list)


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    pattern = r"^```(?:json)?\s*\n?(.*?)\n?\s*```$"
    match = re.match(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _parse_blueprint_json(raw: str) -> Dict[str, Any]:
    """Parse JSON from LLM output, stripping markdown fences if needed."""
    cleaned = _strip_markdown_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise BuilderParseError(
            f"Failed to parse blueprint JSON: {e}\nRaw output: {cleaned[:200]}"
        ) from e


def _validate_blueprint(data: Dict[str, Any]) -> TeamBlueprint:
    """
    Validate and convert raw JSON to TeamBlueprint.

    Raises:
        BuilderValidationError: If validation fails.
    """
    agents_data = data.get("agents", [])

    if not agents_data:
        raise BuilderValidationError("Blueprint must have at least 1 agent")

    if len(agents_data) > 5:
        raise BuilderValidationError(
            f"Blueprint has {len(agents_data)} agents (max 5)"
        )

    # Check unique roles
    roles = [a.get("role", "") for a in agents_data]
    seen = set()
    for role in roles:
        if role in seen:
            raise BuilderValidationError(f"Duplicate role: '{role}'")
        if role.startswith("_"):
            raise BuilderValidationError(
                f"Role '{role}' is reserved (starts with '_')"
            )
        seen.add(role)

    # Build agent blueprints with tool filtering
    agent_blueprints = []
    for agent_data in agents_data:
        raw_tools = agent_data.get("tools", [])
        valid_tools = []
        for tool_name in raw_tools:
            if tool_name in TOOL_MAP:
                valid_tools.append(tool_name)
            else:
                warnings.warn(
                    f"Unknown tool '{tool_name}' for agent '{agent_data.get('role', '?')}' — skipped",
                    stacklevel=2,
                )

        temperature = agent_data.get("temperature", 0.7)
        if not (0 <= temperature <= 2):
            raise BuilderValidationError(
                f"Agent '{agent_data.get('role', '?')}': temperature must be 0-2, got {temperature}"
            )

        max_tool_rounds = agent_data.get("max_tool_rounds", 5)
        if not (1 <= max_tool_rounds <= 10):
            raise BuilderValidationError(
                f"Agent '{agent_data.get('role', '?')}': max_tool_rounds must be 1-10, got {max_tool_rounds}"
            )

        agent_blueprints.append(AgentBlueprint(
            role=agent_data.get("role", "agent"),
            purpose=agent_data.get("purpose", ""),
            prompt=agent_data.get("prompt", ""),
            tools=valid_tools,
            model=agent_data.get("model", "gpt-4o-mini"),
            temperature=temperature,
            max_tool_rounds=max_tool_rounds,
        ))

    return TeamBlueprint(
        team_id=data.get("team_id", "dynamic-team"),
        description=data.get("description", ""),
        agents=agent_blueprints,
        routing_strategy=data.get("routing_strategy", ""),
        input_variables=data.get("input_variables", []),
    )


class DynamicTeamBuilder:
    """
    Builds LLMTeam instances dynamically from task descriptions.

    Uses an LLM to analyze the task, design a team blueprint,
    and create a configured team with appropriate tools.

    Example:
        builder = DynamicTeamBuilder(model="gpt-4o-mini")
        blueprint = await builder.analyze_task("Research and summarize AI papers")
        team = builder.build_team(blueprint)
        await builder.execute(team, {"query": "transformer architectures"})
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        verbose: bool = True,
        provider: Optional[Any] = None,
    ):
        """
        Initialize DynamicTeamBuilder.

        Args:
            model: Model to use for task analysis (default: gpt-4o-mini)
            verbose: Print progress to stdout
            provider: Optional pre-configured LLM provider instance.
                      If None, creates OpenAIProvider from OPENAI_API_KEY.
        """
        self._model = model
        self._verbose = verbose
        self._provider = provider

    def _get_provider(self) -> Any:
        """Get or create the LLM provider."""
        if self._provider is not None:
            return self._provider

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise BuilderError(
                "OPENAI_API_KEY not set. Set the environment variable or pass a provider instance."
            )

        from llmteam.providers import OpenAIProvider
        self._provider = OpenAIProvider(model=self._model, api_key=api_key)
        return self._provider

    def _log(self, message: str) -> None:
        """Print if verbose mode is on."""
        if self._verbose:
            print(message)

    async def _llm_call(self, prompt: str) -> str:
        """Make a simple LLM completion call."""
        provider = self._get_provider()
        return await provider.complete(
            prompt,
            temperature=0.3,
            max_tokens=2000,
        )

    async def analyze_task(self, task_description: str) -> TeamBlueprint:
        """
        Analyze a task description and generate a TeamBlueprint.

        Args:
            task_description: Natural language description of the task.

        Returns:
            TeamBlueprint with agents, tools, and routing strategy.

        Raises:
            BuilderParseError: If LLM returns invalid JSON.
            BuilderValidationError: If blueprint fails validation.
        """
        self._log("Analyzing task...")

        prompt = TASK_ANALYSIS.format(task_description=task_description)
        raw = await self._llm_call(prompt)
        data = _parse_blueprint_json(raw)
        blueprint = _validate_blueprint(data)

        self._log(f"\n=== Team Blueprint ===")
        self._log(f"Team: {blueprint.team_id}")
        self._log(f"Description: {blueprint.description}")
        self._log(f"Agents:")
        for i, agent in enumerate(blueprint.agents, 1):
            tools_str = ", ".join(agent.tools) if agent.tools else "none"
            self._log(f"  {i}. {agent.role} (tools: {tools_str})")
        self._log(f"Routing: {blueprint.routing_strategy}")

        return blueprint

    async def ask_clarifying_questions(self, task_description: str) -> Optional[List[str]]:
        """
        Ask the LLM if it needs clarifying questions for the task.

        Args:
            task_description: The user's task description.

        Returns:
            List of questions if clarification is needed, None if task is clear.
        """
        prompt = CLARIFYING_QUESTIONS.format(task_description=task_description)
        raw = await self._llm_call(prompt)
        data = _parse_blueprint_json(raw)

        if data.get("clear", False):
            return None

        return data.get("questions", [])

    async def refine_blueprint(
        self, blueprint: TeamBlueprint, feedback: str
    ) -> TeamBlueprint:
        """
        Refine an existing blueprint based on user feedback.

        Args:
            blueprint: Current TeamBlueprint to modify.
            feedback: User's feedback/modification request.

        Returns:
            Updated TeamBlueprint.
        """
        self._log("Refining blueprint...")

        blueprint_dict = {
            "team_id": blueprint.team_id,
            "description": blueprint.description,
            "agents": [
                {
                    "role": a.role,
                    "purpose": a.purpose,
                    "prompt": a.prompt,
                    "tools": a.tools,
                    "model": a.model,
                    "temperature": a.temperature,
                    "max_tool_rounds": a.max_tool_rounds,
                }
                for a in blueprint.agents
            ],
            "routing_strategy": blueprint.routing_strategy,
            "input_variables": blueprint.input_variables,
        }

        prompt = REFINE_BLUEPRINT.format(
            blueprint_json=json.dumps(blueprint_dict, indent=2),
            feedback=feedback,
        )
        raw = await self._llm_call(prompt)
        data = _parse_blueprint_json(raw)
        refined = _validate_blueprint(data)

        self._log(f"Refined team: {refined.team_id} ({len(refined.agents)} agents)")
        return refined

    def build_team(self, blueprint: TeamBlueprint) -> Any:
        """
        Build an LLMTeam instance from a TeamBlueprint.

        Args:
            blueprint: Validated TeamBlueprint.

        Returns:
            Configured LLMTeam instance with agents and tools.

        Raises:
            BuilderValidationError: If blueprint has no agents.
        """
        from llmteam import LLMTeam

        if not blueprint.agents:
            raise BuilderValidationError("Cannot build team with no agents")

        self._log(f"\nBuilding team '{blueprint.team_id}'...")

        team = LLMTeam(
            team_id=blueprint.team_id,
            orchestration=True,  # ROUTER mode for dynamic routing
        )

        for agent_bp in blueprint.agents:
            # Resolve tool definitions
            tools = [TOOL_MAP[name] for name in agent_bp.tools if name in TOOL_MAP]

            config = {
                "type": "llm",
                "role": agent_bp.role,
                "prompt": agent_bp.prompt,
                "model": agent_bp.model,
                "temperature": agent_bp.temperature,
                "max_tool_rounds": agent_bp.max_tool_rounds,
            }
            if tools:
                config["tools"] = tools

            team.add_agent(config)

        self._log(f"Team built: {len(blueprint.agents)} agents ready")
        return team

    async def execute(self, team: Any, input_data: Dict[str, Any]) -> None:
        """
        Execute the team with streaming output.

        Args:
            team: LLMTeam instance to execute.
            input_data: Input data dict for the team run.
        """
        from llmteam.events.streaming import StreamEventType

        self._log(f"\nRunning team...")

        total_tokens = 0

        try:
            async for event in team.stream(input_data):
                if event.type == StreamEventType.AGENT_STARTED:
                    self._log(f"  [AGENT_STARTED] {event.agent_id}")
                elif event.type == StreamEventType.TOOL_CALL:
                    name = event.data.get("tool_name", "?")
                    args = event.data.get("arguments", {})
                    args_str = ", ".join(f'{k}="{v}"' for k, v in args.items())
                    self._log(f"  [TOOL_CALL] {name}({args_str})")
                elif event.type == StreamEventType.TOOL_RESULT:
                    result = event.data.get("result", "")
                    preview = result[:100] + "..." if len(str(result)) > 100 else result
                    self._log(f"  [TOOL_RESULT] {preview}")
                elif event.type == StreamEventType.AGENT_COMPLETED:
                    output = event.data.get("output", "")
                    preview = output[:200] if len(str(output)) > 200 else output
                    self._log(f"  [AGENT_COMPLETED] {event.agent_id}: {preview}")
                    total_tokens += event.data.get("tokens_used", 0)
                elif event.type == StreamEventType.RUN_COMPLETED:
                    final_output = event.data.get("output", "")
                    self._log(f"\n=== Final Result ===")
                    self._log(final_output)
                elif event.type == StreamEventType.RUN_FAILED:
                    error = event.data.get("error", "Unknown error")
                    self._log(f"\n=== Error ===")
                    self._log(f"{error}")

        except Exception as e:
            self._log(f"\nExecution error: {e}")
            self._log("You can adjust the team and retry.")
            raise

        if total_tokens > 0:
            estimated_cost = total_tokens * 0.00000015  # rough gpt-4o-mini estimate
            self._log(f"\nTokens: {total_tokens:,} | Cost: ~${estimated_cost:.4f}")

    async def run_interactive(self) -> None:
        """
        Run the full interactive CLI flow.

        Prompts for task description, optionally asks clarifying questions,
        builds and executes the team.
        """
        print("=== LLMTeam Dynamic Builder ===\n")

        # Get task description
        task = input("Describe your task:\n> ").strip()
        if not task:
            print("No task provided. Exiting.")
            return

        # Check for clarifying questions
        questions = await self.ask_clarifying_questions(task)
        if questions:
            print("\n  I have a few questions:")
            answers = []
            for i, q in enumerate(questions, 1):
                print(f"  {i}. {q}")
            answer = input("> ").strip()
            if answer:
                task = f"{task}\nAdditional context: {answer}"

        # Analyze and build blueprint
        print()
        blueprint = await self.analyze_task(task)

        # Offer refinement
        adjust = input("\nShould I adjust? (yes/no): ").strip().lower()
        if adjust in ("yes", "y"):
            feedback = input("What would you like to change?\n> ").strip()
            if feedback:
                blueprint = await self.refine_blueprint(blueprint, feedback)

        # Build team
        team = self.build_team(blueprint)

        # Get input
        vars_hint = ", ".join(blueprint.input_variables) if blueprint.input_variables else "query"
        user_input = input(f"\nEnter your {vars_hint}:\n> ").strip()
        if not user_input:
            print("No input provided. Exiting.")
            return

        # Build input data
        input_data: Dict[str, Any] = {}
        if blueprint.input_variables:
            for var in blueprint.input_variables:
                input_data[var] = user_input
        else:
            input_data["query"] = user_input

        # Execute
        print()
        await self.execute(team, input_data)
