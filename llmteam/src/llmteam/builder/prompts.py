"""
LLM prompts for DynamicTeamBuilder.

RFC-021: Prompts for task analysis, clarifying questions, and blueprint refinement.
RFC-019: Quality-aware prompts with pipeline depth and agent count guidance.
"""

TASK_ANALYSIS = """You are an AI team architect. Analyze the user's task and design a team of AI agents to accomplish it.

Available tools (use ONLY these names):
- web_search: Search the web for information
- http_fetch: Fetch content from a URL
- json_extract: Extract values from JSON using dot-notation paths
- text_summarize: Summarize text by extracting key sentences
- code_eval: Safely evaluate Python expressions (arithmetic, string ops)

Quality Settings (RFC-019):
- Quality Level: {quality}/100 ({quality_label})
- Pipeline Depth: {pipeline_depth}
- Agent Count: {min_agents} to {max_agents} agents
- Recommended Model: {recommended_model}

Rules:
1. Create {min_agents}-{max_agents} agents based on quality level
2. Roles must be unique and NOT start with "_"
3. Only use tools from the list above
4. Use model "{recommended_model}" for all agents (quality-appropriate)
5. temperature: 0.0-0.3 for factual/analytical, 0.5-0.7 for creative, 0.7-1.0 for brainstorming
6. max_tool_rounds: 1-10 (more for research-heavy agents)
7. Each agent's prompt should use {{input_variable}} placeholders
8. routing_strategy describes how the orchestrator routes tasks to agents

IMPORTANT quality guidance:
- For quality < 30 (Draft): Create {min_agents} agents, simple pipeline, quick iteration
- For quality 30-70 (Balanced): Create 2-4 agents, balanced depth
- For quality > 70 (Production): Create up to {max_agents} agents, thorough analysis

User's task: {task_description}

Respond with ONLY valid JSON (no markdown, no explanation):
{{
  "team_id": "short-descriptive-id",
  "description": "What this team does",
  "agents": [
    {{
      "role": "unique-role-name",
      "purpose": "What this agent does",
      "prompt": "Agent instruction with {{variable}} placeholders",
      "tools": ["tool_name"],
      "model": "{recommended_model}",
      "temperature": 0.3,
      "max_tool_rounds": 5
    }}
  ],
  "routing_strategy": "How to route tasks between agents",
  "input_variables": ["variable_names_used_in_prompts"]
}}"""

CLARIFYING_QUESTIONS = """You are an AI team architect. The user described a task, and you need to determine if you have enough information to design an effective agent team.

User's task: {task_description}

If the task is clear enough to design a team, respond with:
{{"clear": true}}

If you need more information, respond with (max 3 questions):
{{"clear": false, "questions": ["Question 1?", "Question 2?"]}}

Respond with ONLY valid JSON (no markdown, no explanation)."""

REFINE_BLUEPRINT = """You are an AI team architect. The user wants to modify an existing team blueprint.

Current blueprint:
{blueprint_json}

Quality Settings (RFC-019):
- Quality Level: {quality}/100 ({quality_label})
- Agent Count: {min_agents} to {max_agents} agents
- Recommended Model: {recommended_model}

User's feedback: {feedback}

Apply the user's feedback to update the blueprint. Follow these rules:
- {min_agents}-{max_agents} agents with unique roles (not starting with "_")
- Only tools: web_search, http_fetch, json_extract, text_summarize, code_eval
- Use model "{recommended_model}" (quality-appropriate)
- temperature: 0.0-2.0
- max_tool_rounds: 1-10

Respond with ONLY the updated valid JSON blueprint (no markdown, no explanation)."""
