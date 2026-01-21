"""
Segment Test Runner.

Provides isolated segment execution for testing with mock resources.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime


@dataclass
class SegmentTestConfig:
    """Configuration for test runs."""

    timeout_seconds: float = 30.0
    capture_events: bool = True
    fail_on_error: bool = True
    mock_llm_responses: list[str] = field(default_factory=list)
    mock_http_responses: dict[tuple[str, str], Any] = field(default_factory=dict)
    mock_store_data: dict[str, Any] = field(default_factory=dict)
    mock_secrets: dict[str, str] = field(default_factory=dict)


@dataclass
class SegmentTestResult:
    """Result from a test run."""

    success: bool
    output: dict[str, Any]
    events: list[Any]
    step_results: dict[str, Any]
    duration_ms: float
    error: Optional[str] = None

    @property
    def step_count(self) -> int:
        """Number of steps executed."""
        return len(self.step_results)

    def get_step_output(self, step_id: str) -> Any:
        """Get output from a specific step."""
        return self.step_results.get(step_id, {}).get("output")

    def assert_success(self) -> None:
        """Assert that the test run succeeded."""
        assert self.success, f"Test run failed: {self.error}"

    def assert_step_executed(self, step_id: str) -> None:
        """Assert that a specific step was executed."""
        assert step_id in self.step_results, f"Step '{step_id}' was not executed"

    def assert_output_contains(self, key: str) -> None:
        """Assert that the output contains a key."""
        assert key in self.output, f"Output does not contain key '{key}'"


class SegmentTestRunner:
    """
    Test runner for segment execution.

    Provides isolated execution environment with mock resources.

    Usage:
        runner = SegmentTestRunner()

        # Configure mocks
        runner.configure(SegmentTestConfig(
            mock_llm_responses=["Hello!"],
            mock_http_responses={("GET", "/api"): {"data": 1}},
        ))

        # Run segment
        result = await runner.run(segment, {"input": "test"})
        result.assert_success()
    """

    def __init__(self) -> None:
        self._config = SegmentTestConfig()
        self._custom_handlers: dict[str, Any] = {}

    def configure(self, config: SegmentTestConfig) -> "SegmentTestRunner":
        """Configure the test runner."""
        self._config = config
        return self

    def register_handler(
        self,
        step_type: str,
        handler: Any,
    ) -> "SegmentTestRunner":
        """Register a custom step handler for testing."""
        self._custom_handlers[step_type] = handler
        return self

    async def run(
        self,
        segment: Any,
        input_data: dict[str, Any],
        config: Optional[SegmentTestConfig] = None,
    ) -> SegmentTestResult:
        """
        Run a segment in test mode.

        Args:
            segment: SegmentDefinition to run.
            input_data: Input data for the segment.
            config: Optional config override.

        Returns:
            SegmentTestResult with execution details.
        """
        from llmteam.testing.mocks import (
            MockLLMProvider,
            MockHTTPClient,
            MockStore,
            MockSecretsProvider,
            MockEventEmitter,
        )
        from llmteam.runtime import RuntimeContextFactory
        from llmteam.engine import ExecutionEngine, ExecutionStatus

        run_config = config or self._config
        start_time = datetime.now()

        # Create mock resources
        mock_llm = MockLLMProvider(
            responses=run_config.mock_llm_responses or ["Mock LLM response"]
        )
        mock_http = MockHTTPClient(responses=run_config.mock_http_responses)
        mock_store = MockStore(initial_data=run_config.mock_store_data)
        mock_secrets = MockSecretsProvider(secrets=run_config.mock_secrets)
        mock_emitter = MockEventEmitter()

        # Create runtime context with mocks
        factory = RuntimeContextFactory()
        factory.register_llm("default", mock_llm)
        factory.register_llm("mock", mock_llm)
        factory.register_store("default", mock_store)
        factory.register_client("http", mock_http)

        runtime = factory.create_runtime(
            tenant_id="test",
            instance_id=f"test-{datetime.now().isoformat()}",
        )

        # Create execution engine
        runner = ExecutionEngine(event_emitter=mock_emitter)

        # Register custom handlers
        for step_type, handler in self._custom_handlers.items():
            runner.register_handler(step_type, handler)

        # Run segment
        step_results: dict[str, Any] = {}
        error: Optional[str] = None

        try:
            result = await runner.run(
                segment=segment,
                input_data=input_data,
                runtime=runtime,
            )

            # Collect step results
            for step in segment.steps:
                step_results[step.step_id] = {
                    "executed": True,
                    "output": result.step_outputs.get(step.step_id),
                }

            success = result.status == ExecutionStatus.COMPLETED

            if not success and result.error:
                error = str(result.error)

            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000

            return SegmentTestResult(
                success=success,
                output=result.output or {},
                events=mock_emitter.events,
                step_results=step_results,
                duration_ms=duration_ms,
                error=error,
            )

        except Exception as e:
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000

            if run_config.fail_on_error:
                raise

            return SegmentTestResult(
                success=False,
                output={},
                events=mock_emitter.events,
                step_results=step_results,
                duration_ms=duration_ms,
                error=str(e),
            )

    async def run_step(
        self,
        step: Any,
        input_data: dict[str, Any],
        config: Optional[SegmentTestConfig] = None,
    ) -> dict[str, Any]:
        """
        Run a single step in isolation.

        Args:
            step: StepDefinition to run.
            input_data: Input data for the step.
            config: Optional config override.

        Returns:
            Step output.
        """
        from llmteam.engine import WorkflowDefinition, EdgeDefinition

        # Wrap step in minimal workflow
        segment = WorkflowDefinition(
            workflow_id="test_segment",
            name="Test Segment",
            entrypoint=step.step_id,
            steps=[step],
            edges=[],
        )

        result = await self.run(segment, input_data, config)
        return result.get_step_output(step.step_id) or {}
