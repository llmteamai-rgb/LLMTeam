"""
Multi-Agent Orchestration Demo.

Complex task requiring multiple agents to produce quality output.
Demonstrates: DynamicTeamBuilder -> Blueprint -> Build -> Stream Execute.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llmteam.builder import DynamicTeamBuilder


TASK = """
I need a team of 4 agents to produce a high-quality investment analysis:

1. A 'researcher' agent that uses web_search to find current data about Tesla (TSLA) stock - recent price, market cap, quarterly earnings, and major news
2. A 'calculator' agent that uses code_eval to compute financial metrics: P/E ratio (given price ~$250 and EPS ~$3.20), year-over-year growth rate, and simple moving averages
3. A 'risk_analyst' agent that uses web_search to find potential risks: competition from BYD/Rivian, regulatory issues, Elon Musk controversies, and market headwinds
4. A 'strategist' agent that uses text_summarize to compile all findings into a final 200-word investment recommendation with a clear BUY/HOLD/SELL rating

All agents should use {query} as the input variable.
Each agent: max_tool_rounds=3, temperature=0.3, model=gpt-4o-mini.
"""

QUERY = "Should I invest in Tesla (TSLA) stock in January 2025? Give me a data-driven analysis."


async def main():
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    print("=" * 70)
    print("  MULTI-AGENT ORCHESTRATION: Investment Analysis")
    print("  Task -> Configurator -> Blueprint -> Build -> Execute")
    print("=" * 70)

    builder = DynamicTeamBuilder(model="gpt-4o-mini", verbose=True)

    # Phase 1: Configurator
    print("\n--- PHASE 1: CONFIGURATOR (LLM designs the team) ---\n")
    blueprint = await builder.analyze_task(TASK)

    print(f"\n  Blueprint summary:")
    print(f"    Team: {blueprint.team_id}")
    print(f"    Agents: {len(blueprint.agents)}")
    for i, a in enumerate(blueprint.agents, 1):
        print(f"      {i}. {a.role} | tools: {a.tools} | temp: {a.temperature}")
    print(f"    Routing: {blueprint.routing_strategy[:100]}")

    # Phase 2: Build team with FLOW mode (sequential pipeline, not triage)
    print("\n--- PHASE 2: BUILD TEAM (flow mode: all agents in sequence) ---\n")

    from llmteam import LLMTeam
    from llmteam.builder import TOOL_MAP

    # Build team from blueprint manually with flow (not orchestration/router)
    agents_config = []
    for a in blueprint.agents:
        # Convert tool names to ToolDefinition objects
        tool_defs = [TOOL_MAP[t] for t in a.tools if t in TOOL_MAP]
        agents_config.append({
            "type": "llm",
            "role": a.role,
            "prompt": a.prompt,
            "model": a.model,
            "temperature": a.temperature,
            "max_tokens": 1024,
            "max_tool_rounds": a.max_tool_rounds,
            "tools": tool_defs,
        })

    # Create sequential flow: researcher -> calculator -> risk_analyst -> strategist
    agent_roles = [a.role for a in blueprint.agents]
    flow = " -> ".join(agent_roles)

    team = LLMTeam(
        team_id=blueprint.team_id,
        agents=agents_config,
        flow=flow,
    )

    # Runtime auto-created by _build_runtime() — registers OpenAI provider from env
    print(f"  Team: {team.team_id}")
    print(f"  Flow: {flow}")
    print(f"  Agents: {[a.agent_id for a in team._agents.values()]}")
    print(f"  Runtime: auto (OpenAI from OPENAI_API_KEY)")

    # Phase 3: Execute (canvas/flow mode - all agents run in sequence)
    print("\n--- PHASE 3: EXECUTE (flow pipeline) ---\n")
    print(f"  Query: {QUERY}\n")

    import time
    start = time.time()
    result = await team.run({"query": QUERY})
    elapsed = time.time() - start

    # Debug info
    if not result.success:
        print(f"  ERROR: {result.error}")
        print(f"  Status: {result.status}")

    # Show results from each agent
    print("  Pipeline execution complete.\n")

    if isinstance(result.output, dict):
        for agent_role, output in result.output.items():
            output_str = str(output)
            print(f"  --- Agent: {agent_role} ---")
            # Show first 500 chars of each agent's output
            if len(output_str) > 500:
                print(f"  {output_str[:500]}...")
            else:
                print(f"  {output_str}")
            print()
    else:
        print(f"  Output: {result.output}")

    # Final summary
    print("=" * 70)
    print("  EXECUTION SUMMARY")
    print("=" * 70)
    print(f"  Success: {result.success}")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  Agents in flow: {flow}")
    if isinstance(result.output, dict):
        print(f"  Agents that produced output: {list(result.output.keys())}")

    # Show the strategist's final recommendation
    if isinstance(result.output, dict):
        final = result.output.get("strategist", result.output.get(agent_roles[-1], ""))
        if final:
            print(f"\n{'=' * 70}")
            print("  FINAL INVESTMENT RECOMMENDATION")
            print("=" * 70)
            print(f"\n{final}\n")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
