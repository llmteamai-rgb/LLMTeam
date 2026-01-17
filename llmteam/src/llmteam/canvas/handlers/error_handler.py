"""
Error Handler.

Provides error handling, recovery, and fallback mechanisms for workflows.

Supports:
- Try-catch-finally patterns
- Fallback values
- Retry with backoff
- Error transformation
- Compensation (undo) actions
"""

from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

from llmteam.runtime import StepContext


@dataclass
class ErrorConfig:
    """Configuration for error handler."""

    # Error handling mode: "catch", "fallback", "retry", "compensate"
    mode: str = "catch"

    # Catch: error types to catch (empty = all)
    catch_types: list[str] = field(default_factory=list)

    # Fallback: default value to return on error
    fallback_value: Any = None

    # Retry configuration
    max_retries: int = 3
    retry_delay_ms: int = 1000
    retry_backoff: float = 2.0  # Exponential backoff multiplier
    max_retry_delay_ms: int = 30000

    # Whether to rethrow error after handling
    rethrow: bool = False

    # Error transformation
    transform_error: bool = False
    error_template: str = ""

    # Logging
    log_errors: bool = True

    # Output field names
    error_field: str = "error"
    success_field: str = "success"
    result_field: str = "result"


class ErrorHandler:
    """
    Handler for error handling and recovery.

    This handler wraps step execution with error handling capabilities.

    Step Type: "error_handler"

    Config:
        mode: Error handling mode
            - "catch": Catch errors and continue
            - "fallback": Return fallback value on error
            - "retry": Retry failed operations
            - "compensate": Execute compensation on error
        catch_types: List of error types to catch
        fallback_value: Value to return on error (for fallback mode)
        max_retries: Maximum retry attempts (for retry mode)
        retry_delay_ms: Initial retry delay in milliseconds
        retry_backoff: Exponential backoff multiplier
        rethrow: Whether to rethrow error after handling
        log_errors: Whether to log errors

    Input:
        Any input passed to wrapped step

    Output:
        success: Whether execution succeeded
        result: Step result (if successful)
        error: Error information (if failed)
        retries: Number of retries attempted (for retry mode)

    Usage in segment JSON:
        {
            "step_id": "safe_api_call",
            "type": "error_handler",
            "config": {
                "mode": "retry",
                "max_retries": 3,
                "retry_delay_ms": 1000,
                "fallback_value": {"status": "unavailable"}
            }
        }
    """

    STEP_TYPE = "error_handler"
    DISPLAY_NAME = "Error Handler"
    DESCRIPTION = "Handle errors with retry, fallback, or catch"
    CATEGORY = "flow_control"

    async def __call__(
        self,
        ctx: StepContext,
        config: dict[str, Any],
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute with error handling."""
        error_config = self._parse_config(config)

        if error_config.mode == "catch":
            return await self._execute_catch(ctx, error_config, input_data)
        elif error_config.mode == "fallback":
            return await self._execute_fallback(ctx, error_config, input_data)
        elif error_config.mode == "retry":
            return await self._execute_retry(ctx, error_config, input_data)
        elif error_config.mode == "compensate":
            return await self._execute_compensate(ctx, error_config, input_data)
        else:
            raise ValueError(f"Unknown error handling mode: {error_config.mode}")

    def _parse_config(self, config: dict) -> ErrorConfig:
        """Parse configuration dict into ErrorConfig."""
        return ErrorConfig(
            mode=config.get("mode", "catch"),
            catch_types=config.get("catch_types", []),
            fallback_value=config.get("fallback_value"),
            max_retries=config.get("max_retries", 3),
            retry_delay_ms=config.get("retry_delay_ms", 1000),
            retry_backoff=config.get("retry_backoff", 2.0),
            max_retry_delay_ms=config.get("max_retry_delay_ms", 30000),
            rethrow=config.get("rethrow", False),
            transform_error=config.get("transform_error", False),
            error_template=config.get("error_template", ""),
            log_errors=config.get("log_errors", True),
            error_field=config.get("error_field", "error"),
            success_field=config.get("success_field", "success"),
            result_field=config.get("result_field", "result"),
        )

    async def _execute_catch(
        self,
        ctx: StepContext,
        config: ErrorConfig,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute with catch mode - catch errors and continue."""
        try:
            # In a real implementation, this would execute the wrapped step
            # For now, we pass through the input as the result
            result = input_data.get("body_result", input_data)

            return {
                config.success_field: True,
                config.result_field: result,
                config.error_field: None,
            }

        except Exception as e:
            error_type = type(e).__name__

            # Check if we should catch this error type
            if config.catch_types and error_type not in config.catch_types:
                raise

            error_info = self._create_error_info(e, config)

            if config.log_errors:
                self._log_error(ctx, e)

            if config.rethrow:
                raise

            return {
                config.success_field: False,
                config.result_field: None,
                config.error_field: error_info,
            }

    async def _execute_fallback(
        self,
        ctx: StepContext,
        config: ErrorConfig,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute with fallback mode - return fallback value on error."""
        try:
            result = input_data.get("body_result", input_data)

            return {
                config.success_field: True,
                config.result_field: result,
                config.error_field: None,
                "used_fallback": False,
            }

        except Exception as e:
            error_type = type(e).__name__

            if config.catch_types and error_type not in config.catch_types:
                raise

            if config.log_errors:
                self._log_error(ctx, e)

            return {
                config.success_field: True,  # Fallback is considered success
                config.result_field: config.fallback_value,
                config.error_field: self._create_error_info(e, config),
                "used_fallback": True,
            }

    async def _execute_retry(
        self,
        ctx: StepContext,
        config: ErrorConfig,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute with retry mode - retry failed operations."""
        last_error: Optional[Exception] = None
        attempts = 0
        delay_ms = config.retry_delay_ms

        for attempt in range(config.max_retries + 1):
            attempts += 1

            try:
                # In a real implementation, this would execute the wrapped step
                result = input_data.get("body_result", input_data)

                # Simulate potential failure for testing
                if input_data.get("_simulate_failure") and attempt <= input_data.get("_fail_count", 0):
                    raise RuntimeError(f"Simulated failure on attempt {attempt}")

                return {
                    config.success_field: True,
                    config.result_field: result,
                    config.error_field: None,
                    "retries": attempts - 1,
                    "final_attempt": attempt + 1,
                }

            except Exception as e:
                last_error = e
                error_type = type(e).__name__

                if config.catch_types and error_type not in config.catch_types:
                    raise

                if config.log_errors:
                    self._log_error(ctx, e, attempt=attempt, max_attempts=config.max_retries + 1)

                if attempt < config.max_retries:
                    # Wait before retry
                    await asyncio.sleep(delay_ms / 1000)
                    # Apply backoff
                    delay_ms = min(
                        int(delay_ms * config.retry_backoff),
                        config.max_retry_delay_ms,
                    )

        # All retries exhausted
        error_info = self._create_error_info(last_error, config) if last_error else None

        if config.rethrow and last_error:
            raise last_error

        # Try fallback
        if config.fallback_value is not None:
            return {
                config.success_field: True,
                config.result_field: config.fallback_value,
                config.error_field: error_info,
                "retries": attempts - 1,
                "used_fallback": True,
            }

        return {
            config.success_field: False,
            config.result_field: None,
            config.error_field: error_info,
            "retries": attempts - 1,
        }

    async def _execute_compensate(
        self,
        ctx: StepContext,
        config: ErrorConfig,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute with compensate mode - run compensation on error."""
        try:
            result = input_data.get("body_result", input_data)

            return {
                config.success_field: True,
                config.result_field: result,
                config.error_field: None,
                "compensated": False,
            }

        except Exception as e:
            error_type = type(e).__name__

            if config.catch_types and error_type not in config.catch_types:
                raise

            if config.log_errors:
                self._log_error(ctx, e)

            # Execute compensation
            compensation_result = None
            try:
                compensation_input = input_data.get("compensation", {})
                if compensation_input:
                    # In a real implementation, this would execute compensation step
                    compensation_result = {"compensated": True}
            except Exception as comp_error:
                if config.log_errors:
                    self._log_error(ctx, comp_error, context="compensation")

            if config.rethrow:
                raise

            return {
                config.success_field: False,
                config.result_field: compensation_result,
                config.error_field: self._create_error_info(e, config),
                "compensated": compensation_result is not None,
            }

    def _create_error_info(self, error: Exception, config: ErrorConfig) -> dict[str, Any]:
        """Create error information dict."""
        error_info = {
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.now().isoformat(),
        }

        if config.transform_error and config.error_template:
            # Apply error template transformation
            error_info["transformed"] = config.error_template.format(
                type=error_info["type"],
                message=error_info["message"],
            )

        return error_info

    def _log_error(
        self,
        ctx: StepContext,
        error: Exception,
        attempt: int = 0,
        max_attempts: int = 1,
        context: str = "",
    ) -> None:
        """Log error information."""
        # In a real implementation, use structured logging
        error_msg = f"[{ctx.step_id}] Error"
        if context:
            error_msg += f" ({context})"
        if max_attempts > 1:
            error_msg += f" (attempt {attempt + 1}/{max_attempts})"
        error_msg += f": {type(error).__name__}: {error}"

        # Would use ctx.logger or observability module
        # For now, just track in metadata
        ctx.metadata.setdefault("errors", []).append({
            "message": error_msg,
            "timestamp": datetime.now().isoformat(),
        })


class TryCatchHandler:
    """
    Handler for try-catch-finally patterns.

    Provides structured error handling with try, catch, and finally blocks.

    Step Type: "try_catch"

    Config:
        try_step: Step to try executing
        catch_steps: Dict mapping error types to handler steps
        finally_step: Step to always execute
        default_catch: Default catch step for unhandled errors

    Input:
        Input for try step

    Output:
        success: Whether try succeeded
        result: Result from try or catch
        error: Error info if caught
        finally_result: Result from finally step
    """

    STEP_TYPE = "try_catch"
    DISPLAY_NAME = "Try-Catch"
    DESCRIPTION = "Structured error handling with try-catch-finally"
    CATEGORY = "flow_control"

    async def __call__(
        self,
        ctx: StepContext,
        config: dict[str, Any],
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute try-catch-finally."""
        result = None
        error = None
        caught = False
        finally_result = None

        try:
            # Execute try block
            try_input = input_data.get("try_input", input_data)
            result = try_input  # Placeholder - actual execution by runner

        except Exception as e:
            error = {
                "type": type(e).__name__,
                "message": str(e),
            }

            # Find matching catch handler
            catch_steps = config.get("catch_steps", {})
            error_type = type(e).__name__

            if error_type in catch_steps:
                caught = True
                # Would execute catch step
                result = {"caught": error_type}
            elif "default" in catch_steps or config.get("default_catch"):
                caught = True
                result = {"caught": "default"}
            else:
                # Re-raise if no handler
                raise

        finally:
            # Execute finally block
            if config.get("finally_step"):
                finally_result = {"executed": True}

        return {
            "success": error is None or caught,
            "result": result,
            "error": error,
            "caught": caught,
            "finally_result": finally_result,
        }
