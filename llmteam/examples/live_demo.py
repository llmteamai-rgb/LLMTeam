"""
LLMTeam v5.4.0 — Live Demo with OpenAI + Built-in Tools.

Demonstrates:
- RFC-015: Provider Function Calling (OpenAI tool_calls)
- RFC-016: Agent Tool Execution Loop (multi-round tool calling)
- RFC-017: Streaming Events (tool_call / tool_result)
- RFC-018: Built-in Tools (web_search, json_extract, code_eval, text_summarize)
- RFC-019: Budget Per-Period (cost tracking)

Requirements:
    pip install openai
    set OPENAI_API_KEY=sk-...

Usage:
    cd llmteam
    set PYTHONPATH=src
    python examples/live_demo.py
"""

import asyncio
import os
import sys
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llmteam import LLMTeam
from llmteam.agents import OrchestratorConfig, OrchestratorMode
from llmteam.tools.builtin import web_search, http_fetch, json_extract, code_eval, text_summarize


def check_api_key():
    """Check that OPENAI_API_KEY is set."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("ERROR: OPENAI_API_KEY not set.")
        print("  set OPENAI_API_KEY=sk-...")
        print("  (or export OPENAI_API_KEY=sk-... on Linux/Mac)")
        sys.exit(1)
    print(f"  API Key: {key[:8]}...{key[-4:]}")
    return key


async def demo_single_agent_tools():
    """
    Demo 1: Single agent with multiple tools.

    Shows RFC-016 tool loop: the agent calls tools in sequence,
    building up context until it has enough info to answer.
    """
    print("\n" + "=" * 60)
    print("  DEMO 1: Single Agent + Multiple Tools (RFC-016 Tool Loop)")
    print("=" * 60)

    # Create team with one agent that has all tools
    team = LLMTeam(
        team_id="research_assistant",
        model="gpt-4o-mini",
        orchestration=True,  # Router mode
    )

    # Add agent with built-in tools
    team.add_agent({
        "type": "llm",
        "role": "assistant",
        "prompt": (
            "You are a helpful research assistant. "
            "Use the available tools to answer the user's question.\n\n"
            "User question: {query}"
        ),
        "tools": [
            web_search.tool_definition,
            json_extract.tool_definition,
            code_eval.tool_definition,
            text_summarize.tool_definition,
        ],
        "max_tool_rounds": 5,
        "model": "gpt-4o-mini",
        "temperature": 0.3,
    })

    print(f"\n  Team: {team.team_id}")
    print(f"  Agents: {team.list_agents()}")
    print(f"  Model: gpt-4o-mini")
    print(f"  Tools: web_search, json_extract, code_eval, text_summarize")

    # Run task
    query = "Search for 'machine learning trends 2025', then calculate 2**10 + 3**5 using code_eval."
    print(f"\n  Task: {query}")
    print("\n  Executing...\n")

    result = await team.run(
        input_data={"query": query},
        run_id="demo-run-1",
    )

    # Display results
    print(f"  Success: {result.success}")
    print(f"  Status: {result.status}")
    print(f"  Tokens Used: {result.tokens_used}")

    if result.error:
        print(f"  Error: {result.error}")
        return result

    if result.output:
        print(f"\n  --- Agent Outputs ---")
        if isinstance(result.output, dict):
            for agent_id, output in result.output.items():
                print(f"\n  [{agent_id}]:")
                # Check for tool calls in output
                if isinstance(output, str):
                    # Truncate long output
                    display = output[:500] + "..." if len(output) > 500 else output
                    print(f"    {display}")
                else:
                    print(f"    {output}")
        else:
            print(f"    {result.output}")

    if result.summary:
        print(f"\n  --- Cost Summary ---")
        print(f"    {json.dumps(result.summary, indent=4, default=str)}")

    if result.report:
        print(f"\n  --- Orchestrator Report ---")
        report_str = result.report if isinstance(result.report, str) else json.dumps(result.report, indent=2, default=str)
        print(f"    {report_str[:300]}")

    return result


async def demo_multi_agent_routing():
    """
    Demo 2: Multi-agent team with orchestrator routing.

    Shows how the orchestrator selects the right agent for the task.
    Each agent has specialized tools.
    """
    print("\n" + "=" * 60)
    print("  DEMO 2: Multi-Agent Team + Orchestrator Routing")
    print("=" * 60)

    team = LLMTeam(
        team_id="specialist_team",
        model="gpt-4o-mini",
        orchestration=True,
    )

    # Agent 1: Researcher (web search)
    team.add_agent({
        "type": "llm",
        "role": "researcher",
        "prompt": (
            "You are a web researcher. Use web_search to find information.\n\n"
            "Research topic: {query}"
        ),
        "tools": [web_search.tool_definition],
        "max_tool_rounds": 3,
        "model": "gpt-4o-mini",
    })

    # Agent 2: Calculator (code eval)
    team.add_agent({
        "type": "llm",
        "role": "calculator",
        "prompt": (
            "You are a math calculator. Use code_eval to compute expressions.\n\n"
            "Calculate: {query}"
        ),
        "tools": [code_eval.tool_definition],
        "max_tool_rounds": 3,
        "model": "gpt-4o-mini",
    })

    # Agent 3: Data Analyst (json + summarize)
    team.add_agent({
        "type": "llm",
        "role": "analyst",
        "prompt": (
            "You are a data analyst. Use json_extract and text_summarize tools.\n\n"
            "Analyze: {query}"
        ),
        "tools": [json_extract.tool_definition, text_summarize.tool_definition],
        "max_tool_rounds": 3,
        "model": "gpt-4o-mini",
    })

    print(f"\n  Team: {team.team_id}")
    print(f"  Agents:")
    print(f"    - researcher (tools: web_search)")
    print(f"    - calculator (tools: code_eval)")
    print(f"    - analyst   (tools: json_extract, text_summarize)")

    # Task that should route to calculator
    query = "What is the factorial of 12?"
    print(f"\n  Task: {query}")
    print(f"  Expected routing: calculator")
    print("\n  Executing...\n")

    result = await team.run(
        input_data={"query": query},
        run_id="demo-run-2",
    )

    print(f"  Success: {result.success}")
    print(f"  Agents Called: {getattr(result, 'agents_called', 'N/A')}")
    print(f"  Tokens Used: {result.tokens_used}")

    if result.output:
        print(f"\n  --- Output ---")
        if isinstance(result.output, dict):
            for agent_id, output in result.output.items():
                display = str(output)[:300]
                print(f"  [{agent_id}]: {display}")
        else:
            print(f"    {str(result.output)[:300]}")

    return result


async def demo_streaming():
    """
    Demo 3: Streaming events (RFC-017).

    Shows real-time TOOL_CALL and TOOL_RESULT events during execution.
    """
    print("\n" + "=" * 60)
    print("  DEMO 3: Streaming Events (RFC-017)")
    print("=" * 60)

    team = LLMTeam(
        team_id="streaming_demo",
        model="gpt-4o-mini",
        orchestration=True,
    )

    team.add_agent({
        "type": "llm",
        "role": "math_agent",
        "prompt": (
            "You are a math helper. Use code_eval to compute the answer.\n\n"
            "Question: {query}"
        ),
        "tools": [code_eval.tool_definition],
        "max_tool_rounds": 3,
        "model": "gpt-4o-mini",
        "temperature": 0.1,
    })

    query = "Calculate the sum of squares from 1 to 10: sum(x**2 for x in range(1, 11))"
    print(f"\n  Task: {query}")
    print("\n  Streaming events:\n")

    event_count = 0
    async for event in team.stream({"query": query}):
        event_count += 1
        etype = event.type
        agent = getattr(event, "agent_id", None) or ""
        data = event.data if hasattr(event, "data") else {}

        if etype == "tool_call":
            print(f"    [{event_count}] TOOL_CALL: {data.get('tool_name', '?')}({data.get('arguments', {})})")
        elif etype == "tool_result":
            print(f"    [{event_count}] TOOL_RESULT: {data.get('output', '?')} (success={data.get('success')})")
        elif etype == "agent_started":
            print(f"    [{event_count}] AGENT_STARTED: {agent}")
        elif etype == "agent_completed":
            print(f"    [{event_count}] AGENT_COMPLETED: {agent}")
            if "output" in data:
                display = str(data["output"])[:200]
                print(f"             Output: {display}")
        elif etype == "run_started":
            print(f"    [{event_count}] RUN_STARTED")
        elif etype == "run_completed":
            print(f"    [{event_count}] RUN_COMPLETED (success={data.get('success')})")
        else:
            print(f"    [{event_count}] {etype}: {str(data)[:100]}")

    print(f"\n  Total events: {event_count}")


async def main():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("  LLMTeam v5.4.0 — Live Demo")
    print("  Provider: OpenAI (gpt-4o-mini)")
    print("#" * 60)

    check_api_key()

    try:
        # Demo 1: Single agent with multiple tools
        await demo_single_agent_tools()

        # Demo 2: Multi-agent routing
        await demo_multi_agent_routing()

        # Demo 3: Streaming events
        await demo_streaming()

    except Exception as e:
        print(f"\n  ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "#" * 60)
    print("  Demo complete!")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
