"""
Testing Utilities module.

Provides mocks, test runners, and harnesses for testing LLMTeam workflows.

Usage:
    from llmteam.testing import MockLLMProvider, SegmentTestRunner

    # Create a mock provider with deterministic responses
    mock_llm = MockLLMProvider(responses=["Hello!", "How can I help?"])

    # Run segment in test mode
    runner = SegmentTestRunner()
    result = await runner.run(segment, input_data)
"""

from llmteam.testing.mocks import (
    MockLLMProvider,
    MockHTTPClient,
    MockStore,
    MockSecretsProvider,
    MockEventEmitter,
)

from llmteam.testing.runner import (
    SegmentTestRunner,
    SegmentTestConfig,
    SegmentTestResult,
)

# Backwards compatibility aliases
TestRunConfig = SegmentTestConfig
TestResult = SegmentTestResult

from llmteam.testing.harness import (
    StepTestHarness,
    HandlerTestCase,
)

__all__ = [
    # Mocks
    "MockLLMProvider",
    "MockHTTPClient",
    "MockStore",
    "MockSecretsProvider",
    "MockEventEmitter",
    # Runner
    "SegmentTestRunner",
    "SegmentTestConfig",
    "SegmentTestResult",
    # Backwards compatibility
    "TestRunConfig",
    "TestResult",
    # Harness
    "StepTestHarness",
    "HandlerTestCase",
]
