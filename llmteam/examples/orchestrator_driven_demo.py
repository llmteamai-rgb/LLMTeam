"""
Orchestrator-Driven Multi-Agent Demo.

The TeamOrchestrator drives the entire process:
1. Receives the task
2. Decides which agent to call next (LLM-based decision)
3. Executes that agent
4. Feeds result back, decides next agent
5. Continues until all needed agents have been called
6. Returns final combined result

NO canvas, NO flow - pure orchestrator-driven execution.
"""

import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llmteam import LLMTeam
from llmteam.builder import DynamicTeamBuilder, TOOL_MAP
from llmteam.runtime import RuntimeContextFactory
from llmteam.providers import OpenAIProvider


TASK = """
I need a team of 4 agents for comprehensive investment analysis:

1. 'researcher' - uses web_search to find Tesla (TSLA) stock data, price, news, earnings
2. 'calculator' - uses code_eval to compute P/E ratio (price=$248, EPS=$3.20), growth metrics
3. 'risk_analyst' - uses web_search to find risks: competition, regulation, controversies
4. 'strategist' - uses text_summarize to compile a final BUY/HOLD/SELL recommendation

All agents use {query} as input variable.
max_tool_rounds=3, temperature=0.3, model=gpt-4o-mini.
"""

QUERY = "Should I invest in Tesla (TSLA) in January 2025? Data-driven analysis please."


async def main():
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    print("=" * 70)
    print("  ORCHESTRATOR-DRIVEN MULTI-AGENT EXECUTION")
    print("  No canvas, no flow - orchestrator decides everything")
    print("=" * 70)

    # Phase 1: Configurator builds blueprint
    print("\n--- PHASE 1: CONFIGURATOR ---\n")
    builder = DynamicTeamBuilder(model="gpt-4o-mini", verbose=True)
    blueprint = await builder.analyze_task(TASK)

    print(f"\n  Agents: {[a.role for a in blueprint.agents]}")

    # Phase 2: Build team in ROUTER mode (orchestrator-driven)
    print("\n--- PHASE 2: BUILD TEAM (ROUTER mode) ---\n")

    agents_config = []
    for a in blueprint.agents:
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

    team = LLMTeam(
        team_id=blueprint.team_id,
        agents=agents_config,
        orchestration=True,  # ROUTER mode - orchestrator decides
    )

    # Set up runtime manually for ROUTER mode
    factory = RuntimeContextFactory()
    provider = OpenAIProvider(model="gpt-4o-mini")
    factory.register_llm("default", provider)
    runtime = factory.create_runtime(tenant_id="default", instance_id="orch-run")
    team.set_runtime(runtime)

    orchestrator = team.get_orchestrator()
    agent_roles = [a.role for a in blueprint.agents]

    print(f"  Team: {team.team_id}")
    print(f"  Router mode: {team.is_router_mode}")
    print(f"  Orchestrator: {orchestrator}")
    print(f"  Agents: {agent_roles}")

    # Phase 3: Orchestrator-driven execution loop
    print("\n--- PHASE 3: ORCHESTRATOR-DRIVEN EXECUTION ---\n")
    print(f"  Query: {QUERY}\n")

    run_id = "orch-run-001"
    orchestrator.start_run(run_id)

    outputs = {}
    agents_called = []
    current_state = {"input": {"query": QUERY}, "outputs": outputs}
    max_iterations = len(team._agents) + 1  # Safety limit

    total_start = time.time()
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        available_agents = [a for a in agent_roles if a not in agents_called]

        if not available_agents:
            print(f"  All agents have been called. Stopping.\n")
            break

        # Ask orchestrator which agent to call next
        print(f"  [Iteration {iteration}] Orchestrator deciding...")
        print(f"    Available: {available_agents}")
        print(f"    Already called: {agents_called}")

        decision = await orchestrator.decide_next_agent(
            current_state=current_state,
            available_agents=available_agents,
        )

        if decision.next_agent is None or decision.next_agent == "":
            print(f"    Decision: DONE (no more agents needed)")
            break

        print(f"    Decision: call '{decision.next_agent}'")
        print(f"    Reason: {decision.reason[:150]}")

        # Execute the chosen agent
        agent = team._agents.get(decision.next_agent)
        if agent is None:
            print(f"    ERROR: Agent '{decision.next_agent}' not found!")
            break

        print(f"\n  >>> Executing agent: {decision.next_agent} <<<")
        agent_start = time.time()

        # Set up event callback for tool visibility
        tool_events = []

        async def _tool_cb(event_type, data, agent_id):
            tool_events.append((event_type, data))

        if hasattr(agent, "_event_callback"):
            agent._event_callback = _tool_cb

        # Build context with previous outputs
        context = {"outputs": outputs}
        if agents_called:
            context["previous_agent"] = agents_called[-1]
            context["previous_output"] = outputs.get(agents_called[-1], "")

        # Execute
        result = await agent.execute(
            input_data={"query": QUERY},
            context=context,
            run_id=run_id,
        )

        agent_elapsed = time.time() - agent_start

        # Clear callback
        if hasattr(agent, "_event_callback"):
            agent._event_callback = None

        # Show tool calls
        for evt_type, evt_data in tool_events:
            if evt_type == "tool_call":
                tool = evt_data.get("tool_name", "?")
                args = str(evt_data.get("arguments", {}))[:80]
                print(f"    [TOOL_CALL] {tool}({args})")
            elif evt_type == "tool_result":
                output = str(evt_data.get("output", ""))[:100]
                print(f"    [TOOL_RESULT] {output}")

        # Store result
        outputs[decision.next_agent] = result.output
        agents_called.append(decision.next_agent)

        # Show agent output
        output_preview = str(result.output)[:300]
        print(f"    Tokens: {result.tokens_used}")
        print(f"    Duration: {agent_elapsed:.1f}s")
        print(f"    Success: {result.success}")
        print(f"    Output: {output_preview}")
        print()

        # Update state for orchestrator's next decision
        current_state["outputs"] = outputs
        current_state["last_agent"] = decision.next_agent
        current_state["last_output"] = result.output

    total_elapsed = time.time() - total_start
    orchestrator.end_run()

    # Final report
    print("=" * 70)
    print("  EXECUTION COMPLETE")
    print("=" * 70)
    print(f"  Iterations: {iteration}")
    print(f"  Agents called: {agents_called}")
    print(f"  Total duration: {total_elapsed:.1f}s")
    print()

    # Show final output from last agent (strategist)
    if agents_called:
        last = agents_called[-1]
        print(f"  Final agent: {last}")
        print(f"  Final output:")
        print(f"  {'-' * 60}")
        print(f"  {outputs[last]}")
        print(f"  {'-' * 60}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
