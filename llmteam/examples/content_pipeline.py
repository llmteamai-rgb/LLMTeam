"""
Content Pipeline Example - v4.0.0 Test Task

Demonstrates LLMTeam with 3 LLM Agents and TeamOrchestrator:
- Writer: Creates initial article draft
- Editor: Improves and polishes the text
- Publisher: Formats for publication with meta-tags

Usage:
    cd llmteam
    PYTHONPATH=src python examples/content_pipeline.py

    # Or with custom topic:
    PYTHONPATH=src python examples/content_pipeline.py --topic "Machine Learning"
"""

import asyncio
import argparse
from typing import Dict, Any

# Import from llmteam v4.0.0
from llmteam import LLMTeam


def create_content_pipeline() -> LLMTeam:
    """
    Create a content pipeline team with 3 LLM agents.

    Flow: Writer → Editor → Publisher
    """
    team = LLMTeam(
        team_id="content_pipeline",
        agents=[
            # Agent 1: Writer - creates initial draft
            {
                "type": "llm",
                "role": "writer",
                "prompt": """You are a professional content writer.

Write a comprehensive article about: {topic}

Requirements:
- 3-4 paragraphs
- Informative and engaging tone
- Include key facts and insights
- Structure with introduction, body, and conclusion

Output the article text only.""",
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 1500,
            },

            # Agent 2: Editor - improves the text
            {
                "type": "llm",
                "role": "editor",
                "prompt": """You are a professional editor.

Review and improve the following article:

{context}

Tasks:
- Fix any grammar or spelling errors
- Improve sentence flow and readability
- Ensure consistent tone throughout
- Enhance clarity where needed
- Keep the original meaning intact

Output the improved article text only.""",
                "model": "gpt-4o-mini",
                "temperature": 0.3,
                "max_tokens": 1500,
                "use_context": True,  # Use output from previous agent
            },

            # Agent 3: Publisher - formats for publication
            {
                "type": "llm",
                "role": "publisher",
                "prompt": """You are a content publisher.

Format the following article for web publication:

{context}

Tasks:
1. Add an engaging headline (H1)
2. Add a brief summary/excerpt (2-3 sentences)
3. Add 5-7 relevant meta keywords
4. Add a meta description (150-160 chars)
5. Format the article with proper HTML-like structure

Output format:
---
HEADLINE: [headline]
SUMMARY: [summary]
KEYWORDS: [comma-separated keywords]
META_DESCRIPTION: [description]
---
[formatted article with ## subheadings]
---""",
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "max_tokens": 2000,
                "use_context": True,
            },
        ],
        # Sequential flow: writer -> editor -> publisher
        flow="writer -> editor -> publisher",
    )

    return team


async def run_pipeline(topic: str) -> Dict[str, Any]:
    """
    Run the content pipeline with a given topic.

    Args:
        topic: The article topic

    Returns:
        Pipeline result with output and metadata
    """
    # Create team
    team = create_content_pipeline()

    print(f"\n{'='*60}")
    print(f"Content Pipeline - v4.0.0 Demo")
    print(f"{'='*60}")
    print(f"\nTopic: {topic}")
    print(f"Agents: {[a['role'] for a in team.to_config()['agents']]}")
    print(f"Flow: writer → editor → publisher")
    print(f"\n{'='*60}")

    # Run pipeline
    print("\nRunning pipeline...")
    result = await team.run({"topic": topic})

    # Display results
    print(f"\n{'='*60}")
    print(f"RESULT")
    print(f"{'='*60}")
    print(f"\nStatus: {result.status}")
    print(f"Success: {result.success}")
    print(f"Agents called: {result.agents_called}")
    print(f"Iterations: {result.iterations}")
    print(f"Duration: {result.duration_ms}ms")

    if result.error:
        print(f"\nError: {result.error}")

    print(f"\n{'='*60}")
    print(f"FINAL OUTPUT")
    print(f"{'='*60}")
    print(f"\n{result.final_output}")

    return {
        "success": result.success,
        "status": str(result.status),
        "output": result.output,
        "final_output": result.final_output,
        "agents_called": result.agents_called,
        "duration_ms": result.duration_ms,
    }


async def run_with_orchestrator(topic: str) -> Dict[str, Any]:
    """
    Alternative: Run with adaptive orchestration.

    Uses orchestration=True for TeamOrchestrator-driven flow.
    """
    team = LLMTeam(
        team_id="content_adaptive",
        agents=[
            {
                "type": "llm",
                "role": "writer",
                "prompt": "Write an article about: {topic}",
                "model": "gpt-4o-mini",
            },
            {
                "type": "llm",
                "role": "editor",
                "prompt": "Edit and improve: {context}",
                "model": "gpt-4o-mini",
                "use_context": True,
            },
            {
                "type": "llm",
                "role": "publisher",
                "prompt": "Format for publication: {context}",
                "model": "gpt-4o-mini",
                "use_context": True,
            },
        ],
        # Adaptive orchestration - TeamOrchestrator decides flow
        orchestration=True,
    )

    result = await team.run({"topic": topic})
    return {
        "success": result.success,
        "final_output": result.final_output,
    }


def main():
    parser = argparse.ArgumentParser(description="Content Pipeline Demo")
    parser.add_argument(
        "--topic",
        default="Искусственный интеллект в медицине",
        help="Article topic (default: AI in medicine)",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Use adaptive orchestration instead of fixed flow",
    )
    args = parser.parse_args()

    if args.adaptive:
        result = asyncio.run(run_with_orchestrator(args.topic))
    else:
        result = asyncio.run(run_pipeline(args.topic))

    print(f"\n{'='*60}")
    print("Pipeline completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
