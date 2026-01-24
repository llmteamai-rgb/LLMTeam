"""
Dynamic Team Builder - Interactive CLI Demo.

RFC-021: Demonstrates automatic team creation from task descriptions.

Usage:
    # Set your API key
    export OPENAI_API_KEY="sk-..."  # bash
    $env:OPENAI_API_KEY="sk-..."    # PowerShell

    # Run the demo
    python examples/dynamic_demo.py

    # Or with a custom model
    python examples/dynamic_demo.py --model gpt-4o
"""

import asyncio
import argparse
import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llmteam.builder import DynamicTeamBuilder, BuilderError


async def main(model: str = "gpt-4o-mini") -> None:
    """Run the interactive dynamic builder."""
    # Check API key early
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print()
        print("Set it with:")
        print('  export OPENAI_API_KEY="sk-..."  # bash/zsh')
        print('  $env:OPENAI_API_KEY="sk-..."    # PowerShell')
        sys.exit(1)

    builder = DynamicTeamBuilder(model=model, verbose=True)

    try:
        await builder.run_interactive()
    except BuilderError as e:
        print(f"\nBuilder error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMTeam Dynamic Builder Demo")
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model for task analysis (default: gpt-4o-mini)",
    )
    args = parser.parse_args()

    asyncio.run(main(model=args.model))
