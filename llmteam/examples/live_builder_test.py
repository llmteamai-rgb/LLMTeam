"""
LLMTeam - Live Builder Test.

Full flow:
1. DynamicTeamBuilder receives a task as prompt (configurator phase)
2. LLM analyzes the task and creates a team blueprint
3. Team is built from the blueprint
4. Team executes the task with streaming events (execution phase)
5. Final result is displayed

Usage:
    $env:OPENAI_API_KEY="sk-..."
    cd llmteam
    $env:PYTHONPATH="src"; python examples/live_builder_test.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llmteam.builder import DynamicTeamBuilder, BuilderError
from llmteam.events.streaming import StreamEventType


TASK_DESCRIPTION = """
I need a team with a single agent that can:
1. Use code_eval tool to execute Python code for mathematical calculations
2. Answer the user's math question by computing the result

The agent should receive the question in {query} variable.
Available tools: code_eval only.
Single agent, role: math_solver.
"""

QUERY = "Calculate the factorial of 15 and the sum of squares from 1 to 20. Show both results."


async def main():
    # Check API key
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        print("ERROR: OPENAI_API_KEY not set.")
        print('  $env:OPENAI_API_KEY="sk-..."  (PowerShell)')
        sys.exit(1)

    print("=" * 60)
    print("  LLMTeam - Live Builder Test")
    print("  Full flow: Task -> Configure -> Build -> Execute -> Result")
    print("=" * 60)
    print(f"\n  API Key: {key[:8]}...{key[-4:]}")
    print(f"  Model: gpt-4o-mini")

    # --- Phase 1: Configurator -----------------------------------
    print("\n" + "-" * 60)
    print("  PHASE 1: CONFIGURATOR (LLM analyzes task, builds blueprint)")
    print("-" * 60)

    builder = DynamicTeamBuilder(model="gpt-4o-mini", verbose=True)

    print(f"\n  Task: {TASK_DESCRIPTION.strip()}")
    print()

    try:
        blueprint = await builder.analyze_task(TASK_DESCRIPTION)
    except BuilderError as e:
        print(f"\n  ERROR in analysis: {e}")
        sys.exit(1)

    # Show blueprint details
    print(f"\n  Blueprint Details:")
    print(f"    Team ID: {blueprint.team_id}")
    print(f"    Description: {blueprint.description}")
    print(f"    Agents: {len(blueprint.agents)}")
    for i, agent in enumerate(blueprint.agents, 1):
        print(f"      {i}. role={agent.role}, tools={agent.tools}, "
              f"model={agent.model}, temp={agent.temperature}")
        print(f"         prompt: {agent.prompt[:80]}...")
    print(f"    Routing: {blueprint.routing_strategy}")
    print(f"    Input vars: {blueprint.input_variables}")

    # --- Phase 2: Build Team -------------------------------------
    print("\n" + "-" * 60)
    print("  PHASE 2: BUILD TEAM (from blueprint)")
    print("-" * 60)

    team = builder.build_team(blueprint)

    # Set up runtime with OpenAI provider (required for actual LLM calls)
    from llmteam.runtime import RuntimeContextFactory
    from llmteam.providers import OpenAIProvider

    factory = RuntimeContextFactory()
    provider = OpenAIProvider(model="gpt-4o-mini")
    factory.register_llm("default", provider)
    runtime = factory.create_runtime(tenant_id="default", instance_id="live-test")
    team.set_runtime(runtime)

    print(f"\n  Team ID: {team.team_id}")
    print(f"  Agents: {team.list_agents()}")
    print(f"  Router mode: {team.is_router_mode}")
    orchestrator = team.get_orchestrator()
    print(f"  Orchestrator: {orchestrator}")
    print(f"  Runtime: configured with OpenAI provider")

    # --- Phase 3: Execute ----------------------------------------
    print("\n" + "-" * 60)
    print("  PHASE 3: EXECUTE (streaming events)")
    print("-" * 60)

    print(f"\n  Query: {QUERY}")
    print("\n  Streaming:\n")

    event_count = 0
    final_output = None

    async for event in team.stream({"query": QUERY}):
        event_count += 1
        etype = event.type
        agent = event.agent_id or ""
        data = event.data

        if etype == StreamEventType.RUN_STARTED:
            print(f"    [{event_count}] RUN_STARTED (agents: {data.get('agents', [])})")

        elif etype == StreamEventType.AGENT_STARTED:
            print(f"    [{event_count}] AGENT_STARTED: {agent} "
                  f"(reason: {data.get('reason', '?')})")

        elif etype == StreamEventType.TOOL_CALL:
            tool = data.get("tool_name", "?")
            args = data.get("arguments", {})
            args_str = ", ".join(f'{k}="{v}"' for k, v in list(args.items())[:3])
            print(f"    [{event_count}] TOOL_CALL: {tool}({args_str})")

        elif etype == StreamEventType.TOOL_RESULT:
            output = str(data.get("output", ""))
            success = data.get("success", True)
            preview = output[:150] + "..." if len(output) > 150 else output
            print(f"    [{event_count}] TOOL_RESULT: {preview} (success={success})")

        elif etype == StreamEventType.AGENT_COMPLETED:
            output = str(data.get("output", ""))
            preview = output[:200] + "..." if len(output) > 200 else output
            print(f"    [{event_count}] AGENT_COMPLETED: {agent}")
            print(f"             Output: {preview}")
            final_output = output

        elif etype == StreamEventType.AGENT_FAILED:
            print(f"    [{event_count}] AGENT_FAILED: {agent} - {data.get('error', '?')}")

        elif etype == StreamEventType.COST_UPDATE:
            cost = data.get("current_cost", 0)
            tokens = data.get("tokens", 0)
            print(f"    [{event_count}] COST_UPDATE: tokens={tokens}, cost=${cost:.6f}")

        elif etype == StreamEventType.RUN_COMPLETED:
            print(f"    [{event_count}] RUN_COMPLETED (success={data.get('success')})")

        elif etype == StreamEventType.RUN_FAILED:
            print(f"    [{event_count}] RUN_FAILED: {data.get('error', '?')}")

        else:
            print(f"    [{event_count}] {etype}: {str(data)[:100]}")

    # --- Results -------------------------------------------------
    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)
    print(f"\n  Total events: {event_count}")
    if final_output:
        print(f"\n  Final output:")
        print(f"    {final_output}")
    else:
        print("\n  No output produced (check errors above)")

    print("\n  Done!")


if __name__ == "__main__":
    asyncio.run(main())
