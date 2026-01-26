"""
Configurator for automatic team creation (RFC-023).

Analyzes tasks and creates team configurations with hybrid routing.

Usage:
    configurator = Configurator(routing_mode="hybrid")
    config = await configurator.configure(
        task="Write an article about AI",
        quality=70,
    )

    # config contains:
    # - analysis: TaskAnalysis with decision points
    # - agents: List of agent configurations
    # - flow: Flow notation string
    # - decision_points: List of DecisionPointConfig
    # - estimated_cost: ConfiguratorCostEstimate
"""

import json
import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from llmteam.configuration.models import (
    TaskAnalysis,
    DecisionPointAnalysis,
    DecisionPointConfig,
    RoutingRuleConfig,
    RouteConfig,
    LLMFallbackConfigData,
    ConfiguratorCostEstimate,
    ConfiguratorOutput,
)
from llmteam.quality import QualityManager, QualityAwareLLMMixin

if TYPE_CHECKING:
    from llmteam.team import LLMTeam


# =============================================================================
# System Prompts (RFC-023)
# =============================================================================

CONFIGURATOR_SYSTEM_PROMPT = """
You are an expert Team Configurator for LLMTeam — an AI task-solving system.

Your job is to:
1. Analyze a task
2. Identify DECISION POINTS where the flow might branch
3. Design an optimal team of AI agents
4. Create a HYBRID FLOW with deterministic routing + adaptive steps

## Quality Levels Guide

| Quality | Agents | Models | Iterations | Checks | AdaptiveSteps |
|---------|--------|--------|------------|--------|---------------|
| 0-30    | 1-2    | gpt-4o-mini only | 1 | None | 0 (linear) |
| 30-50   | 2-3    | gpt-4o-mini + gpt-4o | 1-2 | Optional | 0-1 |
| 50-70   | 3-4    | gpt-4o primary | 2-3 | 1 review | 1-2 |
| 70-90   | 4-6    | gpt-4o + gpt-4-turbo | 3-5 | 2+ reviews | 2-3 |
| 90-100  | 5-8    | gpt-4-turbo primary | 5+ | Multiple experts | 3+ |

## Decision Points (KEY CONCEPT)

A DECISION POINT is where:
- The next step depends on the OUTPUT of previous step
- There are multiple valid paths forward
- A choice must be made based on context

Types of decision points:
1. **COMPLEXITY**: Simple vs complex input → different processing
2. **QUALITY_GATE**: Pass vs fail → continue vs retry/reject
3. **BRANCH**: Multiple valid paths → choose best
4. **RETRY**: Success vs failure → proceed vs retry

## Hybrid Routing Strategy

For each decision point:
1. ALWAYS try to define RULES first (cheaper, deterministic)
2. Add LLM fallback ONLY if rules can't cover all cases
3. Define clear routes with descriptions

Example rules:
- "output.word_count < 500" → simple_writer
- "output.complexity == 'high'" → researcher
- "output.score >= 80" → next_step
- "output.has_errors == true" → error_handler

## Agent Design Principles

1. **Single Responsibility**: Each agent has one clear purpose
2. **Clear Prompts**: Specific instructions, expected output format
3. **Appropriate Models**: Match model to task complexity
4. **Minimal Agents**: Don't create agents "just in case"
5. **Structured Output**: Agents at decision points output data for rules

## Output Format

Always respond with valid JSON. No additional text.
"""

TASK_ANALYSIS_PROMPT = """
Analyze this task:

Task: {task}
Quality: {quality}
Constraints: {constraints}

Identify:
1. Main goal
2. Subtasks required
3. Complexity level
4. Domain
5. Required capabilities
6. DECISION POINTS - where the flow might branch

For each decision point, determine:
- What type is it? (complexity / quality_gate / branch / retry)
- Can it be decided by simple rules? What rules?
- What are the possible routes?
- Does it need LLM fallback?

Output JSON only:
{{
    "main_goal": "Main objective",
    "input_type": "text/data/etc",
    "output_type": "text/data/etc",
    "sub_tasks": ["subtask1", ...],
    "complexity": "simple|moderate|complex",
    "domain": "content creation|analysis|...",
    "required_capabilities": ["research", "writing", ...],
    "decision_points": [
        {{
            "after_step": "step_name",
            "decision_type": "complexity|quality_gate|branch|retry",
            "description": "What decision is being made",
            "rule_feasible": true,
            "suggested_rules": [
                "output.field == 'value' → target_step"
            ],
            "needs_llm_fallback": false,
            "possible_routes": ["route_a", "route_b"]
        }}
    ]
}}
"""

TEAM_DESIGN_PROMPT = """
Design team for task:

Task: {task}
Analysis: {analysis}
Quality: {quality}
Max Cost: {max_cost}
Routing Mode: {routing_mode}

IMPORTANT: For hybrid routing, agents at decision points MUST output
structured data that rules can evaluate.

Output JSON only:
{{
    "agents": [
        {{
            "name": "agent_name",
            "purpose": "Why needed",
            "prompt": "Full prompt with {{placeholders}}. For decision points, include OUTPUT FORMAT with evaluatable fields.",
            "model": "gpt-4o-mini|gpt-4o|gpt-4-turbo",
            "temperature": 0.0-1.0,
            "max_tokens": 500-4000
        }}
    ],
    "flow": "agent1 → [adaptive:decision_id] → agent2 | agent3",
    "decision_points": [
        {{
            "decision_id": "unique_id",
            "after_step": "agent_name",
            "decision_type": "complexity|quality_gate|branch|retry",
            "rules": [
                {{
                    "condition": "output.field == 'value'",
                    "target": "target_step",
                    "description": "When to use this route"
                }}
            ],
            "llm_fallback": {{
                "model": "gpt-4o-mini",
                "prompt": "Decide routing based on...",
                "routes": [
                    {{"target": "step_a", "description": "...", "when": "..."}}
                ]
            }},
            "default_route": "fallback_step"
        }}
    ],
    "orchestration": {{"mode": "hybrid", "max_iterations": 10}},
    "estimated_cost": {{
        "min_cost": 0.05,
        "max_cost": 0.15,
        "expected_cost": 0.10,
        "adaptive_decisions": 2,
        "adaptive_cost": 0.01
    }},
    "reasoning": "Why this design",
    "warnings": []
}}
"""


class Configurator(QualityAwareLLMMixin):
    """
    Configurator for automatic team creation (RFC-023).

    Analyzes tasks and creates team configurations with hybrid routing.

    Example:
        configurator = Configurator(routing_mode="hybrid")
        config = await configurator.configure(
            task="Write an article about AI",
            quality=70,
        )
        team = configurator.build_team(config)
        result = await team.run({"topic": "AI"})
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.3,
        routing_mode: str = "hybrid",
        quality: int = 50,
    ):
        """
        Initialize Configurator.

        Args:
            model: Model for configuration LLM calls
            temperature: Temperature for LLM calls
            routing_mode: Default routing mode (sequential/hybrid/dynamic)
            quality: Default quality level (0-100)
        """
        self._model = model
        self._temperature = temperature
        self._routing_mode = routing_mode
        self._quality = quality
        self._quality_manager = QualityManager(quality)

    def _get_quality_manager(self) -> QualityManager:
        """Get QualityManager for QualityAwareLLMMixin."""
        return self._quality_manager

    async def configure(
        self,
        task: str,
        quality: int = 50,
        constraints: Optional[Dict[str, Any]] = None,
        max_cost: Optional[float] = None,
        context: Optional[str] = None,
        routing_mode: Optional[str] = None,
    ) -> ConfiguratorOutput:
        """
        Analyze task and create team configuration with hybrid routing.

        Args:
            task: Task description
            quality: Quality level 0-100
            constraints: Task constraints
            max_cost: Maximum cost budget
            context: Additional context
            routing_mode: Override routing mode

        Returns:
            ConfiguratorOutput with full team configuration
        """
        routing = routing_mode or self._routing_mode
        self._quality_manager = QualityManager(quality)

        # Step 1: Analyze task + identify decision points
        analysis = await self._analyze_task(task, quality, constraints, context)

        # Step 2: Design team with decision points
        design = await self._design_team(
            task, analysis, quality, max_cost, constraints, routing
        )

        return design

    async def _analyze_task(
        self,
        task: str,
        quality: int,
        constraints: Optional[Dict[str, Any]],
        context: Optional[str],
    ) -> TaskAnalysis:
        """Analyze task and identify decision points."""
        prompt = TASK_ANALYSIS_PROMPT.format(
            task=task,
            quality=quality,
            constraints=json.dumps(constraints or {}, ensure_ascii=False),
        )

        response = await self._quality_complete(
            prompt=prompt,
            system_prompt=CONFIGURATOR_SYSTEM_PROMPT,
            complexity="complex",
        )

        data = self._parse_json(response)

        # Parse decision points
        decision_points = []
        for dp in data.get("decision_points", []):
            decision_points.append(DecisionPointAnalysis(
                after_step=dp.get("after_step", ""),
                decision_type=dp.get("decision_type", "branch"),
                description=dp.get("description", ""),
                rule_feasible=dp.get("rule_feasible", True),
                suggested_rules=dp.get("suggested_rules", []),
                needs_llm_fallback=dp.get("needs_llm_fallback", True),
                possible_routes=dp.get("possible_routes", []),
            ))

        return TaskAnalysis(
            main_goal=data.get("main_goal", data.get("goal", "")),
            input_type=data.get("input_type", ""),
            output_type=data.get("output_type", ""),
            sub_tasks=data.get("sub_tasks", data.get("subtasks", [])),
            complexity=data.get("complexity", "moderate"),
            raw_analysis=response,
            domain=data.get("domain", ""),
            required_capabilities=data.get("required_capabilities", []),
            decision_points=decision_points,
        )

    async def _design_team(
        self,
        task: str,
        analysis: TaskAnalysis,
        quality: int,
        max_cost: Optional[float],
        constraints: Optional[Dict[str, Any]],
        routing_mode: str,
    ) -> ConfiguratorOutput:
        """Design team with agents and decision points."""
        prompt = TEAM_DESIGN_PROMPT.format(
            task=task,
            analysis=json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False),
            quality=quality,
            max_cost=max_cost or "unlimited",
            routing_mode=routing_mode,
        )

        response = await self._quality_complete(
            prompt=prompt,
            system_prompt=CONFIGURATOR_SYSTEM_PROMPT,
            complexity="complex",
        )

        data = self._parse_json(response)

        # Parse agents
        agents = []
        for agent in data.get("agents", []):
            agents.append({
                "role": agent.get("name", agent.get("role", "")),
                "type": "llm",
                "purpose": agent.get("purpose", ""),
                "prompt": agent.get("prompt", ""),
                "model": agent.get("model", "gpt-4o-mini"),
                "temperature": agent.get("temperature", 0.7),
                "max_tokens": agent.get("max_tokens", 1000),
            })

        # Parse decision points
        decision_points = []
        for dp in data.get("decision_points", []):
            rules = [
                RoutingRuleConfig(
                    condition=r.get("condition", ""),
                    target=r.get("target", ""),
                    description=r.get("description", ""),
                )
                for r in dp.get("rules", [])
            ]

            llm_fallback = None
            if dp.get("llm_fallback"):
                fb = dp["llm_fallback"]
                routes = [
                    RouteConfig(
                        target=r.get("target", ""),
                        description=r.get("description", ""),
                        when=r.get("when", ""),
                    )
                    for r in fb.get("routes", [])
                ]
                llm_fallback = LLMFallbackConfigData(
                    model=fb.get("model", "gpt-4o-mini"),
                    prompt=fb.get("prompt", ""),
                    routes=routes,
                    max_tokens=fb.get("max_tokens", 100),
                    temperature=fb.get("temperature", 0.0),
                )

            all_routes = [
                RouteConfig(
                    target=r.get("target", ""),
                    description=r.get("description", ""),
                    when=r.get("when", ""),
                )
                for r in dp.get("routes", [])
            ]

            decision_points.append(DecisionPointConfig(
                decision_id=dp.get("decision_id", f"dp_{uuid.uuid4().hex[:8]}"),
                after_step=dp.get("after_step", ""),
                decision_type=dp.get("decision_type", "branch"),
                rules=rules,
                llm_fallback=llm_fallback,
                routes=all_routes,
                default_route=dp.get("default_route"),
            ))

        # Parse cost estimate
        cost_data = data.get("estimated_cost", {})
        estimated_cost = ConfiguratorCostEstimate(
            min_cost=cost_data.get("min_cost", 0.05),
            max_cost=cost_data.get("max_cost", 0.20),
            expected_cost=cost_data.get("expected_cost", 0.10),
            breakdown=cost_data.get("breakdown", {}),
            adaptive_decisions=cost_data.get("adaptive_decisions", 0),
            adaptive_cost=cost_data.get("adaptive_cost", 0.0),
        )

        return ConfiguratorOutput(
            analysis=analysis,
            agents=agents,
            flow=data.get("flow", " → ".join(a["role"] for a in agents)),
            decision_points=decision_points,
            orchestration=data.get("orchestration", {"mode": routing_mode}),
            estimated_cost=estimated_cost,
            reasoning=data.get("reasoning", ""),
            warnings=data.get("warnings", []),
        )

    def build_team(self, config: ConfiguratorOutput) -> "LLMTeam":
        """
        Build LLMTeam from configuration.

        Args:
            config: ConfiguratorOutput from configure()

        Returns:
            LLMTeam ready to execute
        """
        from llmteam.team import LLMTeam

        team = LLMTeam(
            team_id=f"configured_{uuid.uuid4().hex[:8]}",
            orchestration=True,  # Enable ROUTER mode
        )

        # Add agents
        for agent in config.agents:
            team.add_agent(agent)

        # Set flow
        if config.flow:
            team.set_flow(config.flow)

        return team

    def _parse_json(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
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

            return {}
