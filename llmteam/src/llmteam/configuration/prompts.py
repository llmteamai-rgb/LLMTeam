"""
Prompts for CONFIGURATOR mode (RFC-005).

Provides prompt templates for task analysis, team suggestion,
and test result analysis.

RFC-019: Quality-aware prompts with pipeline depth and agent count guidance.
"""


class ConfiguratorPrompts:
    """Prompt templates for CONFIGURATOR operations."""

    TASK_ANALYSIS = """Analyze the user's task:

Task: {task}
Constraints: {constraints}

Quality Level: {quality}/100 ({quality_label})

Extract:
1. Main goal
2. Input type
3. Expected output
4. Sub-tasks needed
5. Complexity (simple/moderate/complex)

Consider the quality level when assessing complexity:
- Low quality (0-30): Prefer simpler analysis
- Medium quality (30-70): Balanced analysis
- High quality (70-100): Thorough, detailed analysis

Return JSON:
{{
    "main_goal": "...",
    "input_type": "...",
    "output_type": "...",
    "sub_tasks": ["...", "..."],
    "complexity": "simple|moderate|complex"
}}"""

    TEAM_SUGGESTION = """Based on analysis, suggest an AI agent team.

Analysis: {task_analysis}

Quality Settings (RFC-019):
- Quality Level: {quality}/100 ({quality_label})
- Pipeline Depth: {pipeline_depth}
- Agent Count: {min_agents} to {max_agents} agents
- Recommended Model: {recommended_model}

Agent types available:
- LLM: text generation, reasoning
- RAG: retrieval + generation
- KAG: knowledge graph + generation

IMPORTANT quality guidance:
- For quality < 30: Use {min_agents} agents, keep pipeline shallow
- For quality 30-70: Use 2-4 agents, balanced depth
- For quality > 70: Use up to {max_agents} agents, thorough analysis

For each agent provide:
- role: unique name
- type: llm/rag/kag
- purpose: what it does
- prompt_template: initial prompt
- reasoning: why needed

Return JSON:
{{
    "agents": [
        {{
            "role": "...",
            "type": "llm|rag|kag",
            "purpose": "...",
            "prompt_template": "...",
            "reasoning": "..."
        }}
    ],
    "flow": "agent1 -> agent2 -> agent3",
    "reasoning": "..."
}}"""

    TEST_ANALYSIS = """Analyze test run results.

Config: {team_config}
Input: {test_input}
Agent outputs: {agent_outputs}
Final output: {final_output}
Duration: {duration_ms}ms
Quality target: {quality}/100

Assess:
1. Does output match goal?
2. Did each agent work correctly?
3. Issues found?
4. Improvements needed?
5. Is quality appropriate for target level?

Return JSON:
{{
    "overall": "success|partial|failure",
    "issues": ["...", "..."],
    "recommendations": ["...", "..."],
    "ready_for_production": true|false,
    "summary": "..."
}}"""

    IMPROVE_PROMPT = """Improve the agent prompt based on test feedback.

Current prompt: {current_prompt}
Agent role: {agent_role}
Test input: {test_input}
Agent output: {agent_output}
Issues: {issues}
Recommendations: {recommendations}

Generate an improved prompt that addresses the issues.

Return JSON:
{{
    "improved_prompt": "...",
    "changes_made": ["...", "..."]
}}"""

    VALIDATE_CONFIG = """Validate the team configuration.

Team config: {team_config}
Task: {task}
Constraints: {constraints}

Check:
1. Are all required roles present?
2. Is the flow correct?
3. Are prompts appropriate for the task?
4. Any potential issues?

Return JSON:
{{
    "valid": true|false,
    "issues": ["...", "..."],
    "suggestions": ["...", "..."]
}}"""
