"""
Integration tests for Content Pipeline with real OpenAI API.

These tests require OPENAI_API_KEY environment variable.

Run:
    OPENAI_API_KEY=sk-... PYTHONPATH=src pytest tests/integration/test_content_pipeline_api.py -v
"""

import os
import pytest
from typing import Dict, Any

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)


from llmteam import LLMTeam
from llmteam.team import RunStatus


class TestContentPipelineIntegration:
    """Integration tests with real OpenAI API."""

    async def test_single_llm_agent(self):
        """Test single LLM agent generates output."""
        team = LLMTeam(
            team_id="single_test",
            agents=[
                {
                    "type": "llm",
                    "role": "greeter",
                    "prompt": "Say hello to {name} in one short sentence.",
                    "model": "gpt-4o-mini",
                    "max_tokens": 50,
                },
            ],
        )

        result = await team.run({"name": "World"})

        assert result.success is True
        assert result.status == RunStatus.COMPLETED
        assert "greeter" in result.agents_called
        assert result.final_output is not None
        assert len(result.final_output) > 0
        print(f"\nOutput: {result.final_output}")

    async def test_two_agents_sequential(self):
        """Test two agents in sequence pass context."""
        team = LLMTeam(
            team_id="two_agents",
            agents=[
                {
                    "type": "llm",
                    "role": "writer",
                    "prompt": "Write one sentence about: {topic}",
                    "model": "gpt-4o-mini",
                    "max_tokens": 100,
                },
                {
                    "type": "llm",
                    "role": "translator",
                    "prompt": "Translate to Russian: {context}",
                    "model": "gpt-4o-mini",
                    "max_tokens": 100,
                    "use_context": True,
                },
            ],
            flow="writer -> translator",
        )

        result = await team.run({"topic": "artificial intelligence"})

        assert result.success is True
        assert result.agents_called == ["writer", "translator"]
        assert result.final_output is not None
        # Should contain Cyrillic characters (Russian)
        assert any(ord(c) > 1024 for c in result.final_output), "Expected Russian text"
        print(f"\nOutput: {result.final_output}")

    async def test_content_pipeline_full(self):
        """Test full content pipeline: Writer -> Editor -> Publisher."""
        team = LLMTeam(
            team_id="content_pipeline",
            agents=[
                {
                    "type": "llm",
                    "role": "writer",
                    "prompt": """Write a short paragraph (3-4 sentences) about: {topic}
Be informative and engaging.""",
                    "model": "gpt-4o-mini",
                    "temperature": 0.7,
                    "max_tokens": 200,
                },
                {
                    "type": "llm",
                    "role": "editor",
                    "prompt": """Improve this text, fix any errors:

{context}

Output only the improved text.""",
                    "model": "gpt-4o-mini",
                    "temperature": 0.3,
                    "max_tokens": 250,
                    "use_context": True,
                },
                {
                    "type": "llm",
                    "role": "publisher",
                    "prompt": """Format for publication:

{context}

Add:
1. A headline (prefix with "HEADLINE: ")
2. 3 keywords (prefix with "KEYWORDS: ")

Then output the article.""",
                    "model": "gpt-4o-mini",
                    "temperature": 0.2,
                    "max_tokens": 350,
                    "use_context": True,
                },
            ],
            flow="writer -> editor -> publisher",
        )

        result = await team.run({"topic": "benefits of morning exercise"})

        assert result.success is True
        assert result.status == RunStatus.COMPLETED
        assert result.agents_called == ["writer", "editor", "publisher"]
        assert result.iterations == 3
        assert result.final_output is not None
        assert len(result.final_output) > 100, "Expected substantial output"

        # Check for expected formatting
        output_lower = result.final_output.lower()
        assert "headline" in output_lower or "keyword" in output_lower, \
            "Expected formatted output with headline or keywords"

        print(f"\n{'='*60}")
        print("CONTENT PIPELINE RESULT")
        print(f"{'='*60}")
        print(f"Status: {result.status}")
        print(f"Agents: {result.agents_called}")
        print(f"Duration: {result.duration_ms}ms")
        print(f"{'='*60}")
        print(result.final_output)
        print(f"{'='*60}")

    async def test_pipeline_with_orchestration(self):
        """Test pipeline with adaptive orchestration."""
        team = LLMTeam(
            team_id="adaptive_pipeline",
            agents=[
                {
                    "type": "llm",
                    "role": "analyst",
                    "prompt": "Analyze this topic in 2-3 sentences: {topic}",
                    "model": "gpt-4o-mini",
                    "max_tokens": 150,
                },
                {
                    "type": "llm",
                    "role": "summarizer",
                    "prompt": "Summarize in one sentence: {context}",
                    "model": "gpt-4o-mini",
                    "max_tokens": 100,
                    "use_context": True,
                },
            ],
            orchestration=True,
        )

        result = await team.run({"topic": "quantum computing"})

        assert result.success is True
        assert result.final_output is not None
        print(f"\nOrchestrated output: {result.final_output}")

    async def test_error_handling(self):
        """Test that errors are handled gracefully."""
        team = LLMTeam(
            team_id="error_test",
            agents=[
                {
                    "type": "llm",
                    "role": "test",
                    "prompt": "Test: {query}",
                    "model": "gpt-4o-mini",
                    "max_tokens": 10,
                },
            ],
        )

        # Empty input should still work
        result = await team.run({"query": ""})
        assert result is not None

    async def test_russian_language(self):
        """Test pipeline with Russian language."""
        team = LLMTeam(
            team_id="russian_pipeline",
            agents=[
                {
                    "type": "llm",
                    "role": "writer",
                    "prompt": """Напиши короткий абзац (2-3 предложения) на тему: {topic}
Пиши на русском языке.""",
                    "model": "gpt-4o-mini",
                    "temperature": 0.7,
                    "max_tokens": 200,
                },
                {
                    "type": "llm",
                    "role": "editor",
                    "prompt": """Улучши этот текст, исправь ошибки:

{context}

Выведи только улучшенный текст на русском.""",
                    "model": "gpt-4o-mini",
                    "temperature": 0.3,
                    "max_tokens": 250,
                    "use_context": True,
                },
            ],
            flow="writer -> editor",
        )

        result = await team.run({"topic": "искусственный интеллект в медицине"})

        assert result.success is True
        assert result.agents_called == ["writer", "editor"]
        # Should contain Cyrillic
        assert any(ord(c) > 1024 for c in result.final_output), "Expected Russian text"

        print(f"\n{'='*60}")
        print("RUSSIAN PIPELINE RESULT")
        print(f"{'='*60}")
        print(result.final_output)
        print(f"{'='*60}")

    async def test_result_metadata(self):
        """Test that result contains proper metadata."""
        team = LLMTeam(
            team_id="metadata_test",
            agents=[
                {
                    "type": "llm",
                    "role": "test",
                    "prompt": "Say: {word}",
                    "model": "gpt-4o-mini",
                    "max_tokens": 20,
                },
            ],
        )

        result = await team.run({"word": "hello"})

        # Check all metadata fields
        assert result.success is True
        assert result.status == RunStatus.COMPLETED
        assert result.agents_called == ["test"]
        assert result.iterations >= 1
        assert result.duration_ms > 0
        assert result.error is None
        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.completed_at >= result.started_at

        print(f"\nMetadata:")
        print(f"  Duration: {result.duration_ms}ms")
        print(f"  Iterations: {result.iterations}")
        print(f"  Started: {result.started_at}")
        print(f"  Completed: {result.completed_at}")
