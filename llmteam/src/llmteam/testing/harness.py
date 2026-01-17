"""
Step Test Harness.

Provides utilities for testing individual step handlers.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Callable, Awaitable
import asyncio


@dataclass
class HandlerTestCase:
    """
    A single test case for a step handler.

    Usage:
        case = HandlerTestCase(
            name="basic_transform",
            input_data={"value": 42},
            expected_output={"result": 84},
            step_config={"expression": "value * 2"},
        )
    """

    name: str
    input_data: dict[str, Any]
    expected_output: Optional[dict[str, Any]] = None
    step_config: dict[str, Any] = field(default_factory=dict)
    should_fail: bool = False
    expected_error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.expected_output is None and not self.should_fail:
            self.expected_output = {}


class StepTestHarness:
    """
    Test harness for step handlers.

    Provides a convenient way to test step handlers in isolation.

    Usage:
        harness = StepTestHarness(handler=my_handler)

        # Add test cases
        harness.add_case(
            HandlerTestCase(
                name="basic",
                input_data={"x": 1},
                expected_output={"y": 2},
            )
        )

        # Run all tests
        results = await harness.run_all()
        harness.assert_all_passed(results)
    """

    def __init__(
        self,
        handler: Callable[..., Awaitable[dict[str, Any]]],
        step_type: str = "test_step",
        default_config: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize the test harness.

        Args:
            handler: The step handler function to test.
            step_type: Step type identifier.
            default_config: Default step configuration.
        """
        self.handler = handler
        self.step_type = step_type
        self.default_config = default_config or {}
        self.cases: list[HandlerTestCase] = []

    def add_case(self, case: HandlerTestCase) -> "StepTestHarness":
        """Add a test case."""
        self.cases.append(case)
        return self

    def add_cases(self, cases: list[HandlerTestCase]) -> "StepTestHarness":
        """Add multiple test cases."""
        self.cases.extend(cases)
        return self

    async def run_case(self, case: HandlerTestCase) -> dict[str, Any]:
        """
        Run a single test case.

        Returns:
            dict with keys: passed, output, error, case_name
        """
        from llmteam.canvas import StepDefinition
        from llmteam.runtime import RuntimeContextFactory
        from llmteam.testing.mocks import MockLLMProvider, MockStore

        # Merge configs
        config = {**self.default_config, **case.step_config}

        # Create step definition
        step = StepDefinition(
            step_id="test_step",
            type=self.step_type,
            config=config,
        )

        # Create mock runtime context
        factory = RuntimeContextFactory()
        factory.register_llm("default", MockLLMProvider())
        factory.register_store("default", MockStore())

        runtime = factory.create_runtime(
            tenant_id="test",
            instance_id="test-harness",
        )
        context = runtime.child_context("test_step")

        try:
            # Call handler
            output = await self.handler(step, case.input_data, context)

            if case.should_fail:
                return {
                    "passed": False,
                    "output": output,
                    "error": "Expected failure but handler succeeded",
                    "case_name": case.name,
                }

            # Check expected output
            if case.expected_output is not None:
                for key, expected_value in case.expected_output.items():
                    actual_value = output.get(key)
                    if actual_value != expected_value:
                        return {
                            "passed": False,
                            "output": output,
                            "error": f"Output mismatch for '{key}': expected {expected_value!r}, got {actual_value!r}",
                            "case_name": case.name,
                        }

            return {
                "passed": True,
                "output": output,
                "error": None,
                "case_name": case.name,
            }

        except Exception as e:
            error_str = str(e)

            if case.should_fail:
                if case.expected_error and case.expected_error not in error_str:
                    return {
                        "passed": False,
                        "output": None,
                        "error": f"Expected error containing '{case.expected_error}', got: {error_str}",
                        "case_name": case.name,
                    }
                return {
                    "passed": True,
                    "output": None,
                    "error": error_str,
                    "case_name": case.name,
                }

            return {
                "passed": False,
                "output": None,
                "error": error_str,
                "case_name": case.name,
            }

    async def run_all(self) -> list[dict[str, Any]]:
        """
        Run all test cases.

        Returns:
            List of results for each case.
        """
        results = []
        for case in self.cases:
            result = await self.run_case(case)
            results.append(result)
        return results

    def assert_all_passed(self, results: list[dict[str, Any]]) -> None:
        """Assert that all test cases passed."""
        failed = [r for r in results if not r["passed"]]
        if failed:
            messages = [f"  - {r['case_name']}: {r['error']}" for r in failed]
            raise AssertionError(
                f"{len(failed)} test case(s) failed:\n" + "\n".join(messages)
            )

    @classmethod
    def from_handler(
        cls,
        handler: Callable[..., Awaitable[dict[str, Any]]],
        step_type: str = "test_step",
    ) -> "StepTestHarness":
        """Create a harness from a handler function."""
        return cls(handler=handler, step_type=step_type)

    @classmethod
    def for_transform(
        cls,
        expression: str,
        test_cases: list[tuple[dict[str, Any], dict[str, Any]]],
    ) -> "StepTestHarness":
        """
        Create a harness for testing transform expressions.

        Args:
            expression: Transform expression.
            test_cases: List of (input, expected_output) tuples.

        Returns:
            Configured harness.
        """
        from llmteam.canvas.handlers import TransformHandler

        harness = cls(
            handler=TransformHandler(),
            step_type="transform",
            default_config={"expression": expression},
        )

        for i, (input_data, expected) in enumerate(test_cases):
            harness.add_case(
                HandlerTestCase(
                    name=f"case_{i}",
                    input_data=input_data,
                    expected_output=expected,
                )
            )

        return harness

    @classmethod
    def for_condition(
        cls,
        condition_config: dict[str, Any],
        test_cases: list[tuple[dict[str, Any], str]],
    ) -> "StepTestHarness":
        """
        Create a harness for testing condition handlers.

        Args:
            condition_config: Condition configuration.
            test_cases: List of (input, expected_branch) tuples.

        Returns:
            Configured harness.
        """
        from llmteam.canvas.handlers import ConditionHandler

        harness = cls(
            handler=ConditionHandler(),
            step_type="condition",
            default_config=condition_config,
        )

        for i, (input_data, expected_branch) in enumerate(test_cases):
            harness.add_case(
                HandlerTestCase(
                    name=f"case_{i}",
                    input_data=input_data,
                    expected_output={"branch": expected_branch},
                )
            )

        return harness
