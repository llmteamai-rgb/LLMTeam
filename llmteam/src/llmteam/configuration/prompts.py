"""
Prompts for CONFIGURATOR mode (RFC-005).

Provides prompt templates for task analysis, team suggestion,
and test result analysis.
"""


class ConfiguratorPrompts:
    """Prompt templates for CONFIGURATOR operations."""

    TASK_ANALYSIS = """Analyze the user's task:

Task: {task}
Constraints: {constraints}

Extract:
1. Main goal
2. Input type
3. Expected output
4. Sub-tasks needed
5. Complexity (simple/moderate/complex)

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

Agent types available:
- LLM: text generation, reasoning
- RAG: retrieval + generation
- KAG: knowledge graph + generation

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

Assess:
1. Does output match goal?
2. Did each agent work correctly?
3. Issues found?
4. Improvements needed?

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
