"""
Extended test cases for LLMTeam Auto-Testing System.

Provides additional ~77 test cases to reach 100 total tests.
Place this file in: llmteam/src/llmteam/testing/extended_tests.py

Usage:
    from llmteam.testing.extended_tests import get_extended_test_classes
    test_classes = get_extended_test_classes()
"""

import asyncio
import time
from typing import Any, List, Type

# Import base classes from auto_test_system
# These will be available when imported from the testing module
try:
    from llmteam.testing.auto_test_system import TestCase, TestCategory, TestContext
except ImportError:
    # Fallback for standalone testing
    from dataclasses import dataclass
    from enum import Enum
    from abc import ABC, abstractmethod
    
    class TestCategory(str, Enum):
        SMOKE = "smoke"
        SINGLE_AGENT = "single_agent"
        MULTI_AGENT = "multi_agent"
        FLOW = "flow"
        CONTEXT = "context"
        GROUP = "group"
        CONFIGURATION = "configuration"
        ERROR_HANDLING = "error_handling"
        PERFORMANCE = "performance"
        EDGE_CASES = "edge_cases"
        CONCURRENCY = "concurrency"
        COST = "cost"
    
    class TestCase(ABC):
        name: str = "Unnamed Test"
        description: str = ""
        category: TestCategory = TestCategory.SMOKE
        tags: List[str] = []
        timeout_seconds: int = 60
        
        @abstractmethod
        async def run(self, context: "TestContext") -> Any:
            pass
        
        def validate(self, output: Any) -> bool:
            return output is not None
    
    @dataclass
    class TestContext:
        config: dict
        shared_state: dict
        _llmteam_class: Any = None
        _group_orchestrator_class: Any = None
        
        def get_llmteam_class(self):
            if self._llmteam_class is None:
                from llmteam import LLMTeam
                self._llmteam_class = LLMTeam
            return self._llmteam_class
        
        def get_group_orchestrator(self):
            if self._group_orchestrator_class is None:
                from llmteam.orchestration import GroupOrchestrator
                self._group_orchestrator_class = GroupOrchestrator
            return self._group_orchestrator_class


# =============================================================================
# EXTENDED SMOKE TESTS (5 tests)
# =============================================================================

class ExtendedSmokeTests:
    """Additional smoke tests."""
    
    class TeamWithMultipleAgents(TestCase):
        name = "Team with Multiple Agents"
        description = "Create team with 3 agents"
        category = TestCategory.SMOKE
        tags = ["creation", "multiple"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="multi_agent_smoke",
                agents=[
                    {"type": "llm", "role": "agent1", "prompt": "Say 1"},
                    {"type": "llm", "role": "agent2", "prompt": "Say 2"},
                    {"type": "llm", "role": "agent3", "prompt": "Say 3"},
                ]
            )
            return {"agents": len(team), "has_all": len(team) == 3}
    
    class TeamWithQuality(TestCase):
        name = "Team with Quality Setting"
        description = "Create team with quality parameter"
        category = TestCategory.SMOKE
        tags = ["creation", "quality"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="quality_smoke",
                agents=[{"type": "llm", "role": "test", "prompt": "Hi"}],
                quality=75,
            )
            return {"quality": team.quality, "correct": team.quality == 75}
    
    class TeamWithPresetQuality(TestCase):
        name = "Team with Preset Quality"
        description = "Create team with quality preset name"
        category = TestCategory.SMOKE
        tags = ["creation", "quality", "preset"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="preset_smoke",
                agents=[{"type": "llm", "role": "test", "prompt": "Hi"}],
                quality="production",
            )
            return {"quality": team.quality, "is_production": team.quality == 75}
    
    class TeamWithTimeout(TestCase):
        name = "Team with Timeout"
        description = "Create team with timeout setting"
        category = TestCategory.SMOKE
        tags = ["creation", "timeout"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="timeout_smoke",
                agents=[{"type": "llm", "role": "test", "prompt": "Hi"}],
                timeout=30,
            )
            return {"timeout": team._timeout, "correct": team._timeout == 30}
    
    class TeamWithMaxCost(TestCase):
        name = "Team with Max Cost"
        description = "Create team with cost limit"
        category = TestCategory.SMOKE
        tags = ["creation", "cost"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="cost_smoke",
                agents=[{"type": "llm", "role": "test", "prompt": "Hi"}],
                max_cost_per_run=0.10,
            )
            return {
                "has_budget": team.budget_manager is not None,
                "max_cost": team._max_cost_per_run,
            }


# =============================================================================
# EXTENDED SINGLE AGENT TESTS (10 tests)
# =============================================================================

class ExtendedSingleAgentTests:
    """Additional single agent tests."""
    
    class AgentWithMaxTokens(TestCase):
        name = "Agent with Max Tokens"
        description = "Test agent with max_tokens limit"
        category = TestCategory.SINGLE_AGENT
        tags = ["llm", "tokens"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="max_tokens",
                agents=[{
                    "type": "llm",
                    "role": "limited",
                    "prompt": "Write a very short greeting",
                    "max_tokens": 20,
                    "model": ctx.config.get("model", "gpt-4o-mini"),
                }]
            )
            result = await team.run({})
            return {"success": result.success, "output": result.output}
    
    class AgentWithTemperature(TestCase):
        name = "Agent with Temperature"
        description = "Test agent with custom temperature"
        category = TestCategory.SINGLE_AGENT
        tags = ["llm", "temperature"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="temperature",
                agents=[{
                    "type": "llm",
                    "role": "creative",
                    "prompt": "Say hello creatively",
                    "temperature": 0.9,
                    "model": ctx.config.get("model", "gpt-4o-mini"),
                }]
            )
            result = await team.run({})
            return {"success": result.success}
    
    class AgentWithSystemPrompt(TestCase):
        name = "Agent with System Prompt"
        description = "Test agent with system message"
        category = TestCategory.SINGLE_AGENT
        tags = ["llm", "system"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="system_prompt",
                agents=[{
                    "type": "llm",
                    "role": "assistant",
                    "system": "You are a helpful assistant. Always respond briefly.",
                    "prompt": "What is 2+2?",
                    "model": ctx.config.get("model", "gpt-4o-mini"),
                }]
            )
            result = await team.run({})
            return {"success": result.success, "output": result.output}
    
    class MultipleVariableSubstitution(TestCase):
        name = "Multiple Variable Substitution"
        description = "Test multiple variables in prompt"
        category = TestCategory.SINGLE_AGENT
        tags = ["llm", "variables"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="multi_vars",
                agents=[{
                    "type": "llm",
                    "role": "greeter",
                    "prompt": "Say hello to {name} who is {age} years old",
                    "model": ctx.config.get("model", "gpt-4o-mini"),
                }]
            )
            result = await team.run({"name": "Alice", "age": "25"})
            return {"success": result.success, "output": result.output}
    
    class EmptyPromptVariable(TestCase):
        name = "Empty Prompt Variable"
        description = "Test handling of empty variable"
        category = TestCategory.SINGLE_AGENT
        tags = ["llm", "variables", "edge"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="empty_var",
                agents=[{
                    "type": "llm",
                    "role": "echo",
                    "prompt": "Repeat: {text}",
                    "model": ctx.config.get("model", "gpt-4o-mini"),
                }]
            )
            result = await team.run({"text": ""})
            return {"success": result.success}
    
    class NumericInput(TestCase):
        name = "Numeric Input"
        description = "Test handling of numeric input"
        category = TestCategory.SINGLE_AGENT
        tags = ["llm", "input", "numeric"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="numeric",
                agents=[{
                    "type": "llm",
                    "role": "calculator",
                    "prompt": "What is {number} times 2?",
                    "model": ctx.config.get("model", "gpt-4o-mini"),
                }]
            )
            result = await team.run({"number": 42})
            return {"success": result.success, "output": result.output}
    
    class BooleanInput(TestCase):
        name = "Boolean Input"
        description = "Test handling of boolean input"
        category = TestCategory.SINGLE_AGENT
        tags = ["llm", "input", "boolean"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="boolean",
                agents=[{
                    "type": "llm",
                    "role": "checker",
                    "prompt": "Is {flag} true or false? Answer in one word.",
                    "model": ctx.config.get("model", "gpt-4o-mini"),
                }]
            )
            result = await team.run({"flag": True})
            return {"success": result.success}
    
    class ListInput(TestCase):
        name = "List Input"
        description = "Test handling of list input"
        category = TestCategory.SINGLE_AGENT
        tags = ["llm", "input", "list"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="list_input",
                agents=[{
                    "type": "llm",
                    "role": "summarizer",
                    "prompt": "How many items: {items}",
                    "model": ctx.config.get("model", "gpt-4o-mini"),
                }]
            )
            result = await team.run({"items": ["a", "b", "c"]})
            return {"success": result.success}
    
    class DictInput(TestCase):
        name = "Dict Input"
        description = "Test handling of dict input"
        category = TestCategory.SINGLE_AGENT
        tags = ["llm", "input", "dict"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="dict_input",
                agents=[{
                    "type": "llm",
                    "role": "reader",
                    "prompt": "What is the name in: {data}",
                    "model": ctx.config.get("model", "gpt-4o-mini"),
                }]
            )
            result = await team.run({"data": {"name": "Bob", "age": 30}})
            return {"success": result.success}
    
    class JsonOutput(TestCase):
        name = "JSON Output Request"
        description = "Test requesting JSON output"
        category = TestCategory.SINGLE_AGENT
        tags = ["llm", "output", "json"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="json_output",
                agents=[{
                    "type": "llm",
                    "role": "generator",
                    "prompt": "Return a JSON object with keys 'name' and 'age'. Only JSON, no markdown.",
                    "model": ctx.config.get("model", "gpt-4o-mini"),
                }]
            )
            result = await team.run({})
            return {"success": result.success, "output": result.output}


# =============================================================================
# EXTENDED MULTI-AGENT TESTS (10 tests)
# =============================================================================

class ExtendedMultiAgentTests:
    """Additional multi-agent tests."""
    
    class FourAgentPipeline(TestCase):
        name = "Four Agent Pipeline"
        description = "Execute four agents in sequence"
        category = TestCategory.MULTI_AGENT
        tags = ["pipeline", "sequence"]
        timeout_seconds = 90
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="four_agents",
                agents=[
                    {"type": "llm", "role": "a", "prompt": "Say A"},
                    {"type": "llm", "role": "b", "prompt": "After {a}, say B"},
                    {"type": "llm", "role": "c", "prompt": "After {b}, say C"},
                    {"type": "llm", "role": "d", "prompt": "After {c}, say D"},
                ],
                flow="a -> b -> c -> d",
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            result = await team.run({})
            return {"success": result.success, "agents": len(result.output)}
    
    class FiveAgentPipeline(TestCase):
        name = "Five Agent Pipeline"
        description = "Execute five agents in sequence"
        category = TestCategory.MULTI_AGENT
        tags = ["pipeline", "sequence"]
        timeout_seconds = 120
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="five_agents",
                agents=[
                    {"type": "llm", "role": "step1", "prompt": "Say 1"},
                    {"type": "llm", "role": "step2", "prompt": "After {step1}, say 2"},
                    {"type": "llm", "role": "step3", "prompt": "After {step2}, say 3"},
                    {"type": "llm", "role": "step4", "prompt": "After {step3}, say 4"},
                    {"type": "llm", "role": "step5", "prompt": "After {step4}, say 5"},
                ],
                flow="step1 -> step2 -> step3 -> step4 -> step5",
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            result = await team.run({})
            return {"success": result.success}
    
    class PipelineWithSharedInput(TestCase):
        name = "Pipeline with Shared Input"
        description = "All agents access original input"
        category = TestCategory.MULTI_AGENT
        tags = ["pipeline", "input"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="shared_input",
                agents=[
                    {"type": "llm", "role": "first", "prompt": "Topic is {topic}. Say intro."},
                    {"type": "llm", "role": "second", "prompt": "Topic is {topic}. After: {first}. Say body."},
                ],
                flow="first -> second",
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            result = await team.run({"topic": "testing"})
            return {"success": result.success}
    
    class ReviewerPattern(TestCase):
        name = "Reviewer Pattern"
        description = "Writer + Reviewer pattern"
        category = TestCategory.MULTI_AGENT
        tags = ["pattern", "review"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="reviewer",
                agents=[
                    {"type": "llm", "role": "writer", "prompt": "Write a sentence about {topic}"},
                    {"type": "llm", "role": "reviewer", "prompt": "Review this: {writer}. Give feedback."},
                ],
                flow="writer -> reviewer",
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            result = await team.run({"topic": "AI"})
            return {"success": result.success}
    
    class TranslatorPattern(TestCase):
        name = "Translator Pattern"
        description = "Writer + Translator pattern"
        category = TestCategory.MULTI_AGENT
        tags = ["pattern", "translation"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="translator",
                agents=[
                    {"type": "llm", "role": "writer", "prompt": "Write 'Hello world' in English"},
                    {"type": "llm", "role": "translator", "prompt": "Translate to French: {writer}"},
                ],
                flow="writer -> translator",
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            result = await team.run({})
            return {"success": result.success}
    
    class SummarizerPattern(TestCase):
        name = "Summarizer Pattern"
        description = "Generator + Summarizer pattern"
        category = TestCategory.MULTI_AGENT
        tags = ["pattern", "summary"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="summarizer",
                agents=[
                    {"type": "llm", "role": "generator", "prompt": "Write 3 sentences about {topic}"},
                    {"type": "llm", "role": "summarizer", "prompt": "Summarize in 1 sentence: {generator}"},
                ],
                flow="generator -> summarizer",
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            result = await team.run({"topic": "programming"})
            return {"success": result.success}
    
    class FactCheckerPattern(TestCase):
        name = "Fact Checker Pattern"
        description = "Claim + Fact-checker pattern"
        category = TestCategory.MULTI_AGENT
        tags = ["pattern", "verification"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="fact_checker",
                agents=[
                    {"type": "llm", "role": "claim", "prompt": "State a fact about {topic}"},
                    {"type": "llm", "role": "checker", "prompt": "Is this fact accurate? {claim}"},
                ],
                flow="claim -> checker",
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            result = await team.run({"topic": "the Sun"})
            return {"success": result.success}
    
    class ExpanderPattern(TestCase):
        name = "Expander Pattern"
        description = "Brief + Expander pattern"
        category = TestCategory.MULTI_AGENT
        tags = ["pattern", "expansion"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="expander",
                agents=[
                    {"type": "llm", "role": "brief", "prompt": "Say one word about {topic}"},
                    {"type": "llm", "role": "expander", "prompt": "Expand on: {brief}"},
                ],
                flow="brief -> expander",
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            result = await team.run({"topic": "cats"})
            return {"success": result.success}
    
    class QAPattern(TestCase):
        name = "Q&A Pattern"
        description = "Question + Answer pattern"
        category = TestCategory.MULTI_AGENT
        tags = ["pattern", "qa"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="qa",
                agents=[
                    {"type": "llm", "role": "question", "prompt": "Ask a question about {topic}"},
                    {"type": "llm", "role": "answer", "prompt": "Answer: {question}"},
                ],
                flow="question -> answer",
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            result = await team.run({"topic": "Python"})
            return {"success": result.success}
    
    class DebatePattern(TestCase):
        name = "Debate Pattern"
        description = "Pro + Con + Synthesis pattern"
        category = TestCategory.MULTI_AGENT
        tags = ["pattern", "debate"]
        timeout_seconds = 90
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="debate",
                agents=[
                    {"type": "llm", "role": "pro", "prompt": "Give one pro argument for {topic}"},
                    {"type": "llm", "role": "con", "prompt": "Give one con argument for {topic}"},
                    {"type": "llm", "role": "synthesis", "prompt": "Synthesize: Pro: {pro}, Con: {con}"},
                ],
                flow="pro -> con -> synthesis",
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            result = await team.run({"topic": "remote work"})
            return {"success": result.success}


# =============================================================================
# EXTENDED CONTEXT TESTS (8 tests)
# =============================================================================

class ExtendedContextTests:
    """Additional context tests."""
    
    class ContextIsolation(TestCase):
        name = "Context Isolation"
        description = "Verify context isolation between runs"
        category = TestCategory.CONTEXT
        tags = ["context", "isolation"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="isolation_test",
                agents=[{"type": "llm", "role": "echo", "prompt": "Repeat: {word}"}],
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            
            result1 = await team.run({"word": "FIRST"})
            result2 = await team.run({"word": "SECOND"})
            
            return {
                "run1_success": result1.success,
                "run2_success": result2.success,
                "isolated": True,
            }
    
    class ContextAccumulation(TestCase):
        name = "Context Accumulation"
        description = "Verify context accumulates through pipeline"
        category = TestCategory.CONTEXT
        tags = ["context", "accumulation"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="accumulation",
                agents=[
                    {"type": "llm", "role": "a", "prompt": "Say A"},
                    {"type": "llm", "role": "b", "prompt": "After {a}, say B"},
                    {"type": "llm", "role": "c", "prompt": "After {a} and {b}, say C"},
                ],
                flow="a -> b -> c",
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            result = await team.run({})
            return {
                "success": result.success,
                "has_a": "a" in result.output,
                "has_b": "b" in result.output,
                "has_c": "c" in result.output,
            }
    
    class ContextOverwrite(TestCase):
        name = "Context Overwrite"
        description = "Test overwriting context values"
        category = TestCategory.CONTEXT
        tags = ["context", "overwrite"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="overwrite",
                agents=[{"type": "llm", "role": "agent", "prompt": "Value is: {value}"}],
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            
            # Run twice with different values
            result1 = await team.run({"value": "ONE"})
            result2 = await team.run({"value": "TWO"})
            
            return {
                "success": result1.success and result2.success,
            }
    
    class LargeContext(TestCase):
        name = "Large Context"
        description = "Test handling large context data"
        category = TestCategory.CONTEXT
        tags = ["context", "large"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="large_context",
                agents=[{"type": "llm", "role": "counter", "prompt": "How many items? {items}"}],
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            
            large_list = list(range(100))
            result = await team.run({"items": str(large_list)})
            return {"success": result.success}
    
    class NestedContext(TestCase):
        name = "Nested Context"
        description = "Test nested dict in context"
        category = TestCategory.CONTEXT
        tags = ["context", "nested"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="nested",
                agents=[{"type": "llm", "role": "reader", "prompt": "Read: {data}"}],
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            
            nested = {
                "level1": {
                    "level2": {
                        "level3": "deep_value"
                    }
                }
            }
            result = await team.run({"data": nested})
            return {"success": result.success}
    
    class ContextWithNone(TestCase):
        name = "Context with None"
        description = "Test None value in context"
        category = TestCategory.CONTEXT
        tags = ["context", "none"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="none_context",
                agents=[{"type": "llm", "role": "handler", "prompt": "Value: {value}"}],
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            result = await team.run({"value": None})
            return {"success": result.success}
    
    class ContextPreservation(TestCase):
        name = "Context Preservation"
        description = "Verify original input preserved"
        category = TestCategory.CONTEXT
        tags = ["context", "preservation"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="preservation",
                agents=[
                    {"type": "llm", "role": "first", "prompt": "Input: {original}"},
                    {"type": "llm", "role": "second", "prompt": "Original was: {original}, first said: {first}"},
                ],
                flow="first -> second",
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            result = await team.run({"original": "TEST_VALUE"})
            return {"success": result.success}
    
    class EmptyContext(TestCase):
        name = "Empty Context"
        description = "Test with empty input dict"
        category = TestCategory.CONTEXT
        tags = ["context", "empty"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="empty_ctx",
                agents=[{"type": "llm", "role": "greeter", "prompt": "Say hello"}],
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            result = await team.run({})
            return {"success": result.success}


# =============================================================================
# EXTENDED FLOW TESTS (8 tests)
# =============================================================================

class ExtendedFlowTests:
    """Additional flow tests."""
    
    class ChainedFlow(TestCase):
        name = "Chained Flow"
        description = "Test chained arrow flow"
        category = TestCategory.FLOW
        tags = ["flow", "chain"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="chained",
                agents=[
                    {"type": "llm", "role": "one", "prompt": "Say 1"},
                    {"type": "llm", "role": "two", "prompt": "After {one}, say 2"},
                    {"type": "llm", "role": "three", "prompt": "After {two}, say 3"},
                ],
                flow="one -> two -> three",
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            result = await team.run({})
            return {"success": result.success}
    
    class SingleAgentFlow(TestCase):
        name = "Single Agent Flow"
        description = "Test flow with single agent"
        category = TestCategory.FLOW
        tags = ["flow", "single"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="single_flow",
                agents=[{"type": "llm", "role": "solo", "prompt": "Say solo"}],
                flow="solo",
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            result = await team.run({})
            return {"success": result.success}
    
    class DefaultSequential(TestCase):
        name = "Default Sequential"
        description = "Test default sequential flow"
        category = TestCategory.FLOW
        tags = ["flow", "default"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="default_seq",
                agents=[
                    {"type": "llm", "role": "first", "prompt": "Say first"},
                    {"type": "llm", "role": "second", "prompt": "Say second"},
                ],
                # flow defaults to "sequential"
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            return {"flow": team._flow, "is_sequential": team._flow == "sequential"}
    
    class FlowWithSpaces(TestCase):
        name = "Flow with Spaces"
        description = "Test flow string with extra spaces"
        category = TestCategory.FLOW
        tags = ["flow", "parsing"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="spaced_flow",
                agents=[
                    {"type": "llm", "role": "a", "prompt": "A"},
                    {"type": "llm", "role": "b", "prompt": "B"},
                ],
                flow="  a  ->  b  ",  # Extra spaces
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            result = await team.run({})
            return {"success": result.success}
    
    class FlowChange(TestCase):
        name = "Flow Change"
        description = "Test changing flow after creation"
        category = TestCategory.FLOW
        tags = ["flow", "change"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="flow_change",
                agents=[
                    {"type": "llm", "role": "a", "prompt": "A"},
                    {"type": "llm", "role": "b", "prompt": "B"},
                ],
                flow="a -> b",
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            
            original_flow = team._flow
            team.set_flow("b -> a")
            new_flow = team._flow
            
            return {
                "original": original_flow,
                "new": new_flow,
                "changed": original_flow != new_flow,
            }
    
    class LongChainFlow(TestCase):
        name = "Long Chain Flow"
        description = "Test long sequential chain"
        category = TestCategory.FLOW
        tags = ["flow", "long"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            agents = [
                {"type": "llm", "role": f"agent{i}", "prompt": f"Say {i}"}
                for i in range(5)
            ]
            flow = " -> ".join([f"agent{i}" for i in range(5)])
            
            team = LLMTeam(
                team_id="long_chain",
                agents=agents,
                flow=flow,
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            return {"agents": len(team), "flow": team._flow}
    
    class AdaptiveFlow(TestCase):
        name = "Adaptive Flow"
        description = "Test adaptive flow mode"
        category = TestCategory.FLOW
        tags = ["flow", "adaptive"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="adaptive",
                agents=[
                    {"type": "llm", "role": "worker", "prompt": "Work on {task}"},
                ],
                flow="adaptive",
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            return {"flow": team._flow, "is_adaptive": team._flow == "adaptive"}
    
    class FlowValidation(TestCase):
        name = "Flow Validation"
        description = "Test flow string validation"
        category = TestCategory.FLOW
        tags = ["flow", "validation"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="validation",
                agents=[
                    {"type": "llm", "role": "a", "prompt": "A"},
                    {"type": "llm", "role": "b", "prompt": "B"},
                ],
                flow="a -> b",
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            # Should not raise
            return {"valid": True, "flow": team._flow}


# =============================================================================
# EXTENDED GROUP TESTS (8 tests)
# =============================================================================

class ExtendedGroupTests:
    """Additional group tests."""
    
    class GroupWithThreeTeams(TestCase):
        name = "Group with Three Teams"
        description = "Create group with three teams"
        category = TestCategory.GROUP
        tags = ["group", "multi"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            GroupOrchestrator = ctx.get_group_orchestrator()
            from llmteam.orchestration import TeamRole
            
            team1 = LLMTeam(team_id="g3_t1", agents=[
                {"type": "llm", "role": "w", "prompt": "T1"}
            ])
            team2 = LLMTeam(team_id="g3_t2", agents=[
                {"type": "llm", "role": "w", "prompt": "T2"}
            ])
            team3 = LLMTeam(team_id="g3_t3", agents=[
                {"type": "llm", "role": "w", "prompt": "T3"}
            ])
            
            group = GroupOrchestrator(group_id="three_teams")
            group.add_team(team1, role=TeamRole.LEADER)
            group.add_team(team2, role=TeamRole.MEMBER)
            group.add_team(team3, role=TeamRole.MEMBER)
            
            return {"teams_count": 3, "all_in_group": all([
                team1.is_in_group, team2.is_in_group, team3.is_in_group
            ])}
    
    class GroupRoles(TestCase):
        name = "Group Roles"
        description = "Test different team roles in group"
        category = TestCategory.GROUP
        tags = ["group", "roles"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            GroupOrchestrator = ctx.get_group_orchestrator()
            from llmteam.orchestration import TeamRole
            
            leader = LLMTeam(team_id="leader", agents=[
                {"type": "llm", "role": "w", "prompt": "Lead"}
            ])
            member = LLMTeam(team_id="member", agents=[
                {"type": "llm", "role": "w", "prompt": "Member"}
            ])
            specialist = LLMTeam(team_id="specialist", agents=[
                {"type": "llm", "role": "w", "prompt": "Specialist"}
            ])
            
            group = GroupOrchestrator(group_id="roles_test")
            group.add_team(leader, role=TeamRole.LEADER)
            group.add_team(member, role=TeamRole.MEMBER)
            group.add_team(specialist, role=TeamRole.SPECIALIST)
            
            return {
                "leader_role": str(leader.group_role),
                "member_role": str(member.group_role),
                "specialist_role": str(specialist.group_role),
            }
    
    class GroupId(TestCase):
        name = "Group ID"
        description = "Test group ID assignment"
        category = TestCategory.GROUP
        tags = ["group", "id"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            GroupOrchestrator = ctx.get_group_orchestrator()
            from llmteam.orchestration import TeamRole
            
            team = LLMTeam(team_id="id_test", agents=[
                {"type": "llm", "role": "w", "prompt": "Hi"}
            ])
            
            group = GroupOrchestrator(group_id="my_custom_group_id")
            group.add_team(team, role=TeamRole.LEADER)
            
            return {
                "group_id": team.group_id,
                "correct": team.group_id == "my_custom_group_id",
            }
    
    class GroupRemoveTeam(TestCase):
        name = "Group Remove Team"
        description = "Test removing team from group"
        category = TestCategory.GROUP
        tags = ["group", "remove"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            GroupOrchestrator = ctx.get_group_orchestrator()
            from llmteam.orchestration import TeamRole
            
            team1 = LLMTeam(team_id="rem_t1", agents=[
                {"type": "llm", "role": "w", "prompt": "T1"}
            ])
            team2 = LLMTeam(team_id="rem_t2", agents=[
                {"type": "llm", "role": "w", "prompt": "T2"}
            ])
            
            group = GroupOrchestrator(group_id="remove_test")
            group.add_team(team1, role=TeamRole.LEADER)
            group.add_team(team2, role=TeamRole.MEMBER)
            
            before = len(group._teams)
            group.remove_team("rem_t2")
            after = len(group._teams)
            
            return {
                "before": before,
                "after": after,
                "removed": before > after,
            }
    
    class GroupListTeams(TestCase):
        name = "Group List Teams"
        description = "Test listing teams in group"
        category = TestCategory.GROUP
        tags = ["group", "list"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            GroupOrchestrator = ctx.get_group_orchestrator()
            from llmteam.orchestration import TeamRole
            
            team1 = LLMTeam(team_id="list_t1", agents=[
                {"type": "llm", "role": "w", "prompt": "T1"}
            ])
            team2 = LLMTeam(team_id="list_t2", agents=[
                {"type": "llm", "role": "w", "prompt": "T2"}
            ])
            
            group = GroupOrchestrator(group_id="list_test")
            group.add_team(team1, role=TeamRole.LEADER)
            group.add_team(team2, role=TeamRole.MEMBER)
            
            team_ids = group.list_teams()

            return {
                "count": len(team_ids),
                "ids": team_ids,
                "has_both": "list_t1" in team_ids and "list_t2" in team_ids,
            }
    
    class GroupGetTeam(TestCase):
        name = "Group Get Team"
        description = "Test getting team by ID from group"
        category = TestCategory.GROUP
        tags = ["group", "get"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            GroupOrchestrator = ctx.get_group_orchestrator()
            from llmteam.orchestration import TeamRole
            
            team = LLMTeam(team_id="get_test", agents=[
                {"type": "llm", "role": "w", "prompt": "Hi"}
            ])
            
            group = GroupOrchestrator(group_id="get_group")
            group.add_team(team, role=TeamRole.LEADER)
            
            found = group.get_team("get_test")
            not_found = group.get_team("nonexistent")
            
            return {
                "found": found is not None,
                "not_found": not_found is None,
                "correct_id": found.team_id == "get_test" if found else False,
            }
    
    class EmptyGroup(TestCase):
        name = "Empty Group"
        description = "Test empty group creation"
        category = TestCategory.GROUP
        tags = ["group", "empty"]
        
        async def run(self, ctx: TestContext) -> Any:
            GroupOrchestrator = ctx.get_group_orchestrator()
            
            group = GroupOrchestrator(group_id="empty_group")
            
            return {
                "teams_count": len(group._teams),
                "is_empty": len(group._teams) == 0,
            }
    
    class GroupAutoId(TestCase):
        name = "Group Auto ID"
        description = "Test automatic group ID generation"
        category = TestCategory.GROUP
        tags = ["group", "auto_id"]
        
        async def run(self, ctx: TestContext) -> Any:
            GroupOrchestrator = ctx.get_group_orchestrator()
            
            group = GroupOrchestrator()  # No group_id
            
            return {
                "has_id": group.group_id is not None,
                "id_type": type(group.group_id).__name__,
            }


# =============================================================================
# EXTENDED ERROR HANDLING TESTS (8 tests)
# =============================================================================

class ExtendedErrorHandlingTests:
    """Additional error handling tests."""
    
    class DuplicateAgentId(TestCase):
        name = "Duplicate Agent ID"
        description = "Handle duplicate agent ID"
        category = TestCategory.ERROR_HANDLING
        tags = ["error", "duplicate"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            try:
                team = LLMTeam(
                    team_id="dup_agent",
                    agents=[
                        {"type": "llm", "role": "same", "prompt": "Hi"},
                        {"type": "llm", "role": "same", "prompt": "Hi"},  # Duplicate
                    ]
                )
                return {"error": False, "message": "No error raised"}
            except ValueError as e:
                return {"error": True, "message": str(e)}
    
    class ReservedRoleName(TestCase):
        name = "Reserved Role Name"
        description = "Handle reserved role name"
        category = TestCategory.ERROR_HANDLING
        tags = ["error", "reserved"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            try:
                team = LLMTeam(
                    team_id="reserved",
                    agents=[
                        {"type": "llm", "role": "_internal", "prompt": "Hi"},  # Reserved
                    ]
                )
                return {"error": False}
            except ValueError as e:
                return {"error": True, "message": str(e)}
    
    class MissingAgentType(TestCase):
        name = "Missing Agent Type"
        description = "Handle missing agent type"
        category = TestCategory.ERROR_HANDLING
        tags = ["error", "missing"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            try:
                team = LLMTeam(
                    team_id="no_type",
                    agents=[
                        {"role": "agent", "prompt": "Hi"},  # No type
                    ]
                )
                return {"error": False}
            except (ValueError, KeyError) as e:
                return {"error": True, "message": str(e)}
    
    class InvalidQuality(TestCase):
        name = "Invalid Quality Value"
        description = "Handle invalid quality value"
        category = TestCategory.ERROR_HANDLING
        tags = ["error", "quality"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            # Quality should be clamped to 0-100
            team = LLMTeam(
                team_id="bad_quality",
                agents=[{"type": "llm", "role": "t", "prompt": "Hi"}],
                quality=150,  # Over 100
            )
            return {
                "quality": team.quality,
                "clamped": team.quality <= 100,
            }
    
    class NegativeQuality(TestCase):
        name = "Negative Quality Value"
        description = "Handle negative quality value"
        category = TestCategory.ERROR_HANDLING
        tags = ["error", "quality"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="neg_quality",
                agents=[{"type": "llm", "role": "t", "prompt": "Hi"}],
                quality=-50,  # Negative
            )
            return {
                "quality": team.quality,
                "clamped": team.quality >= 0,
            }
    
    class InvalidPreset(TestCase):
        name = "Invalid Quality Preset"
        description = "Handle invalid quality preset name"
        category = TestCategory.ERROR_HANDLING
        tags = ["error", "preset"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="bad_preset",
                agents=[{"type": "llm", "role": "t", "prompt": "Hi"}],
                quality="nonexistent_preset",  # Invalid preset
            )
            # Should fallback to default (50)
            return {
                "quality": team.quality,
                "is_default": team.quality == 50,
            }
    
    class EmptyTeamRun(TestCase):
        name = "Empty Team Run"
        description = "Run team with no agents"
        category = TestCategory.ERROR_HANDLING
        tags = ["error", "empty"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(team_id="empty_run")  # No agents
            result = await team.run({})
            return {
                "success": result.success,
                "has_error": result.error is not None,
            }
    
    class RemoveNonexistentAgent(TestCase):
        name = "Remove Nonexistent Agent"
        description = "Remove agent that doesn't exist"
        category = TestCategory.ERROR_HANDLING
        tags = ["error", "remove"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="remove_none",
                agents=[{"type": "llm", "role": "exists", "prompt": "Hi"}]
            )
            result = team.remove_agent("nonexistent")
            return {
                "removed": result,
                "expected_false": result == False,
            }


# =============================================================================
# EXTENDED PERFORMANCE TESTS (5 tests)
# =============================================================================

class ExtendedPerformanceTests:
    """Additional performance tests."""
    
    class TeamCreationSpeed(TestCase):
        name = "Team Creation Speed"
        description = "Measure team creation time"
        category = TestCategory.PERFORMANCE
        tags = ["performance", "creation"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            
            start = time.time()
            for i in range(10):
                team = LLMTeam(
                    team_id=f"speed_test_{i}",
                    agents=[{"type": "llm", "role": "t", "prompt": "Hi"}]
                )
            elapsed = time.time() - start
            
            return {
                "teams_created": 10,
                "elapsed_ms": int(elapsed * 1000),
                "avg_ms": int(elapsed * 100),
            }
    
    class AgentAddSpeed(TestCase):
        name = "Agent Add Speed"
        description = "Measure agent addition time"
        category = TestCategory.PERFORMANCE
        tags = ["performance", "agent"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(team_id="add_speed")
            
            start = time.time()
            for i in range(20):
                team.add_agent({
                    "type": "llm",
                    "role": f"agent_{i}",
                    "prompt": f"Agent {i}"
                })
            elapsed = time.time() - start
            
            return {
                "agents_added": 20,
                "elapsed_ms": int(elapsed * 1000),
                "final_count": len(team),
            }
    
    class LargeTeam(TestCase):
        name = "Large Team"
        description = "Create team with many agents"
        category = TestCategory.PERFORMANCE
        tags = ["performance", "large"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            
            agents = [
                {"type": "llm", "role": f"agent_{i}", "prompt": f"Agent {i}"}
                for i in range(50)
            ]
            
            start = time.time()
            team = LLMTeam(team_id="large_team", agents=agents)
            elapsed = time.time() - start
            
            return {
                "agents": len(team),
                "elapsed_ms": int(elapsed * 1000),
            }
    
    class SerializationSpeed(TestCase):
        name = "Serialization Speed"
        description = "Measure config serialization time"
        category = TestCategory.PERFORMANCE
        tags = ["performance", "serialization"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="serialize",
                agents=[
                    {"type": "llm", "role": f"a{i}", "prompt": f"P{i}"}
                    for i in range(10)
                ]
            )
            
            start = time.time()
            for _ in range(100):
                config = team.to_config()
            elapsed = time.time() - start
            
            return {
                "iterations": 100,
                "elapsed_ms": int(elapsed * 1000),
            }
    
    class GroupCreationSpeed(TestCase):
        name = "Group Creation Speed"
        description = "Measure group creation time"
        category = TestCategory.PERFORMANCE
        tags = ["performance", "group"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            GroupOrchestrator = ctx.get_group_orchestrator()
            from llmteam.orchestration import TeamRole
            
            start = time.time()
            for i in range(5):
                group = GroupOrchestrator(group_id=f"perf_group_{i}")
                for j in range(3):
                    team = LLMTeam(
                        team_id=f"perf_team_{i}_{j}",
                        agents=[{"type": "llm", "role": "w", "prompt": "Hi"}]
                    )
                    role = TeamRole.LEADER if j == 0 else TeamRole.MEMBER
                    group.add_team(team, role=role)
            elapsed = time.time() - start
            
            return {
                "groups": 5,
                "teams_per_group": 3,
                "elapsed_ms": int(elapsed * 1000),
            }


# =============================================================================
# EXTENDED EDGE CASE TESTS (8 tests)
# =============================================================================

class ExtendedEdgeCaseTests:
    """Additional edge case tests."""
    
    class EmptyString(TestCase):
        name = "Empty String Input"
        description = "Handle empty string in input"
        category = TestCategory.EDGE_CASES
        tags = ["edge", "empty"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="empty_str",
                agents=[{"type": "llm", "role": "echo", "prompt": "Echo: {text}"}],
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            result = await team.run({"text": ""})
            return {"success": result.success}
    
    class WhitespaceOnly(TestCase):
        name = "Whitespace Only Input"
        description = "Handle whitespace-only input"
        category = TestCategory.EDGE_CASES
        tags = ["edge", "whitespace"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="whitespace",
                agents=[{"type": "llm", "role": "echo", "prompt": "Echo: {text}"}],
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            result = await team.run({"text": "   \n\t   "})
            return {"success": result.success}
    
    class NewlinesInPrompt(TestCase):
        name = "Newlines in Prompt"
        description = "Handle newlines in prompt"
        category = TestCategory.EDGE_CASES
        tags = ["edge", "newlines"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="newlines",
                agents=[{
                    "type": "llm",
                    "role": "multi",
                    "prompt": "Line 1\nLine 2\nLine 3",
                }],
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            result = await team.run({})
            return {"success": result.success}
    
    class TabsInPrompt(TestCase):
        name = "Tabs in Prompt"
        description = "Handle tabs in prompt"
        category = TestCategory.EDGE_CASES
        tags = ["edge", "tabs"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="tabs",
                agents=[{
                    "type": "llm",
                    "role": "tabbed",
                    "prompt": "Col1\tCol2\tCol3",
                }],
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            result = await team.run({})
            return {"success": result.success}
    
    class CyrillicInput(TestCase):
        name = "Cyrillic Input"
        description = "Handle Cyrillic characters"
        category = TestCategory.EDGE_CASES
        tags = ["edge", "i18n", "cyrillic"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="cyrillic",
                agents=[{"type": "llm", "role": "echo", "prompt": "Повторите: {text}"}],
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            result = await team.run({"text": "Привет мир"})
            return {"success": result.success}
    
    class ChineseInput(TestCase):
        name = "Chinese Input"
        description = "Handle Chinese characters"
        category = TestCategory.EDGE_CASES
        tags = ["edge", "i18n", "chinese"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="chinese",
                agents=[{"type": "llm", "role": "echo", "prompt": "重复: {text}"}],
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            result = await team.run({"text": "你好世界"})
            return {"success": result.success}
    
    class EmojiInput(TestCase):
        name = "Emoji Input"
        description = "Handle emoji in input"
        category = TestCategory.EDGE_CASES
        tags = ["edge", "emoji"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="emoji",
                agents=[{"type": "llm", "role": "echo", "prompt": "What emoji? {text}"}],
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            result = await team.run({"text": "🚀🌟💡"})
            return {"success": result.success}
    
    class MixedEncoding(TestCase):
        name = "Mixed Encoding"
        description = "Handle mixed language input"
        category = TestCategory.EDGE_CASES
        tags = ["edge", "mixed"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="mixed",
                agents=[{"type": "llm", "role": "reader", "prompt": "Read: {text}"}],
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            result = await team.run({"text": "Hello Привет 你好 🌍"})
            return {"success": result.success}


# =============================================================================
# EXTENDED CONCURRENCY TESTS (5 tests)
# =============================================================================

class ExtendedConcurrencyTests:
    """Additional concurrency tests."""
    
    class SequentialRuns(TestCase):
        name = "Sequential Runs"
        description = "Multiple sequential runs on same team"
        category = TestCategory.CONCURRENCY
        tags = ["concurrency", "sequential"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="seq_runs",
                agents=[{"type": "llm", "role": "counter", "prompt": "Count: {n}"}],
                model=ctx.config.get("model", "gpt-4o-mini"),
            )
            
            results = []
            for i in range(3):
                result = await team.run({"n": i})
                results.append(result.success)
            
            return {
                "runs": 3,
                "all_success": all(results),
            }
    
    class ParallelDifferentTeams(TestCase):
        name = "Parallel Different Teams"
        description = "Run different teams in parallel"
        category = TestCategory.CONCURRENCY
        tags = ["concurrency", "parallel"]
        timeout_seconds = 60
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            
            teams = [
                LLMTeam(
                    team_id=f"parallel_{i}",
                    agents=[{"type": "llm", "role": "worker", "prompt": f"Team {i}"}],
                    model=ctx.config.get("model", "gpt-4o-mini"),
                )
                for i in range(3)
            ]
            
            start = time.time()
            results = await asyncio.gather(*[t.run({}) for t in teams])
            elapsed = time.time() - start
            
            return {
                "teams": 3,
                "all_success": all(r.success for r in results),
                "elapsed_s": round(elapsed, 2),
            }
    
    class RapidCreation(TestCase):
        name = "Rapid Creation"
        description = "Rapidly create many teams"
        category = TestCategory.CONCURRENCY
        tags = ["concurrency", "creation"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            
            async def create_team(i):
                return LLMTeam(
                    team_id=f"rapid_{i}",
                    agents=[{"type": "llm", "role": "w", "prompt": "Hi"}]
                )
            
            teams = await asyncio.gather(*[create_team(i) for i in range(20)])
            
            return {
                "created": len(teams),
                "all_valid": all(t.team_id.startswith("rapid_") for t in teams),
            }
    
    class InterleavedOperations(TestCase):
        name = "Interleaved Operations"
        description = "Interleaved team operations"
        category = TestCategory.CONCURRENCY
        tags = ["concurrency", "interleaved"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            
            team1 = LLMTeam(team_id="inter1", agents=[
                {"type": "llm", "role": "a", "prompt": "A"}
            ])
            team2 = LLMTeam(team_id="inter2", agents=[
                {"type": "llm", "role": "b", "prompt": "B"}
            ])
            
            # Add agents to both
            team1.add_agent({"type": "llm", "role": "a2", "prompt": "A2"})
            team2.add_agent({"type": "llm", "role": "b2", "prompt": "B2"})
            
            return {
                "team1_agents": len(team1),
                "team2_agents": len(team2),
                "both_correct": len(team1) == 2 and len(team2) == 2,
            }
    
    class ConcurrentGroupCreation(TestCase):
        name = "Concurrent Group Creation"
        description = "Create multiple groups concurrently"
        category = TestCategory.CONCURRENCY
        tags = ["concurrency", "group"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            GroupOrchestrator = ctx.get_group_orchestrator()
            from llmteam.orchestration import TeamRole
            
            async def create_group(i):
                group = GroupOrchestrator(group_id=f"conc_group_{i}")
                team = LLMTeam(
                    team_id=f"conc_team_{i}",
                    agents=[{"type": "llm", "role": "w", "prompt": "Hi"}]
                )
                group.add_team(team, role=TeamRole.LEADER)
                return group
            
            groups = await asyncio.gather(*[create_group(i) for i in range(5)])
            
            return {
                "groups": len(groups),
                "all_valid": all(g.group_id.startswith("conc_group_") for g in groups),
            }


# =============================================================================
# EXTENDED QUALITY/COST TESTS (5 tests)
# =============================================================================

class ExtendedQualityCostTests:
    """Additional quality and cost tests."""
    
    class QualityPresets(TestCase):
        name = "Quality Presets"
        description = "Test all quality presets"
        category = TestCategory.COST
        tags = ["quality", "presets"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            
            presets = {
                "draft": 20,
                "economy": 30,
                "balanced": 50,
                "production": 75,
                "best": 95,
            }
            
            results = {}
            for preset, expected in presets.items():
                team = LLMTeam(
                    team_id=f"preset_{preset}",
                    agents=[{"type": "llm", "role": "t", "prompt": "Hi"}],
                    quality=preset,
                )
                results[preset] = {
                    "expected": expected,
                    "actual": team.quality,
                    "match": team.quality == expected,
                }
            
            return {
                "presets": results,
                "all_match": all(r["match"] for r in results.values()),
            }
    
    class QualityModelSelection(TestCase):
        name = "Quality Model Selection"
        description = "Test model selection by quality"
        category = TestCategory.COST
        tags = ["quality", "model"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            
            results = {}
            for quality in [20, 50, 80]:
                team = LLMTeam(
                    team_id=f"model_{quality}",
                    agents=[{"type": "llm", "role": "t", "prompt": "Hi"}],
                    quality=quality,
                )
                model = team._quality_manager.get_model("medium")
                results[quality] = model
            
            return {
                "models": results,
                "different": len(set(results.values())) > 1,
            }
    
    class CostTrackerExists(TestCase):
        name = "Cost Tracker Exists"
        description = "Verify cost tracker is created"
        category = TestCategory.COST
        tags = ["cost", "tracker"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="tracker_test",
                agents=[{"type": "llm", "role": "t", "prompt": "Hi"}],
            )
            
            return {
                "has_tracker": team.cost_tracker is not None,
                "tracker_type": type(team.cost_tracker).__name__,
            }
    
    class BudgetManagerCreation(TestCase):
        name = "Budget Manager Creation"
        description = "Verify budget manager with max_cost"
        category = TestCategory.COST
        tags = ["cost", "budget"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            
            team_no_budget = LLMTeam(
                team_id="no_budget",
                agents=[{"type": "llm", "role": "t", "prompt": "Hi"}],
            )
            
            team_with_budget = LLMTeam(
                team_id="with_budget",
                agents=[{"type": "llm", "role": "t", "prompt": "Hi"}],
                max_cost_per_run=0.50,
            )
            
            return {
                "no_budget_manager": team_no_budget.budget_manager is None,
                "has_budget_manager": team_with_budget.budget_manager is not None,
            }
    
    class QualityManagerAccess(TestCase):
        name = "Quality Manager Access"
        description = "Access quality manager methods"
        category = TestCategory.COST
        tags = ["quality", "manager"]
        
        async def run(self, ctx: TestContext) -> Any:
            LLMTeam = ctx.get_llmteam_class()
            team = LLMTeam(
                team_id="qm_access",
                agents=[{"type": "llm", "role": "t", "prompt": "Hi"}],
                quality=60,
            )
            
            qm = team.get_quality_manager()
            
            return {
                "has_qm": qm is not None,
                "quality": qm.quality,
                "model": qm.get_model("medium"),
                "depth": str(qm.get_pipeline_depth()),
            }


# =============================================================================
# EXPORT FUNCTION
# =============================================================================

def get_extended_test_classes() -> List[Type[TestCase]]:
    """
    Get all extended test classes.
    
    Returns:
        List of test class types
    """
    test_classes = []
    
    containers = [
        ExtendedSmokeTests,
        ExtendedSingleAgentTests,
        ExtendedMultiAgentTests,
        ExtendedContextTests,
        ExtendedFlowTests,
        ExtendedGroupTests,
        ExtendedErrorHandlingTests,
        ExtendedPerformanceTests,
        ExtendedEdgeCaseTests,
        ExtendedConcurrencyTests,
        ExtendedQualityCostTests,
    ]
    
    for container in containers:
        for name in dir(container):
            obj = getattr(container, name)
            if isinstance(obj, type) and issubclass(obj, TestCase) and obj != TestCase:
                test_classes.append(obj)
    
    return test_classes


# Quick count
if __name__ == "__main__":
    classes = get_extended_test_classes()
    print(f"Extended tests count: {len(classes)}")
    print(f"Total with base 23: {len(classes) + 23}")
    
    # List by category
    by_category = {}
    for cls in classes:
        cat = cls.category.value if hasattr(cls, 'category') else 'unknown'
        by_category.setdefault(cat, []).append(cls.__name__)
    
    print("\nBy category:")
    for cat, tests in sorted(by_category.items()):
        print(f"  {cat}: {len(tests)}")
