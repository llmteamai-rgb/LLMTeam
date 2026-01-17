"""
Loop Handler.

Provides iterative execution with configurable loop control.

Supports:
- For-each loops over collections
- While loops with conditions
- Until loops with termination conditions
- Numeric range loops
"""

from typing import Any, Optional
from dataclasses import dataclass

from llmteam.runtime import StepContext


@dataclass
class LoopConfig:
    """Configuration for loop handler."""

    # Loop type: "foreach", "while", "until", "range"
    loop_type: str = "foreach"

    # For-each: input field containing collection
    collection_field: str = "items"

    # For-each: variable name for current item
    item_variable: str = "item"

    # For-each: variable name for current index
    index_variable: str = "index"

    # While/Until: condition expression
    condition: str = ""

    # Range: start, end, step
    range_start: int = 0
    range_end: int = 10
    range_step: int = 1

    # Maximum iterations (safety limit)
    max_iterations: int = 1000

    # Whether to continue on item error
    continue_on_error: bool = False

    # Aggregate results
    aggregate: bool = True

    # Break condition (expression evaluated after each iteration)
    break_condition: str = ""


class LoopHandler:
    """
    Handler for iterative step execution.

    This handler manages loop control flow and iteration over collections.

    Step Type: "loop"

    Config:
        loop_type: Type of loop ("foreach", "while", "until", "range")
        collection_field: Field containing items (for foreach)
        item_variable: Variable name for current item
        index_variable: Variable name for current index
        condition: Condition expression (for while/until)
        range_start/end/step: Range parameters (for range)
        max_iterations: Maximum iterations allowed
        continue_on_error: Continue iteration on item errors
        aggregate: Whether to aggregate results

    Input:
        items: Collection to iterate (for foreach)
        body: Step configuration to execute for each iteration
        <condition_data>: Data for condition evaluation

    Output:
        results: List of iteration results
        iterations: Number of iterations executed
        errors: List of errors (if continue_on_error)

    Usage in segment JSON:
        {
            "step_id": "process_items",
            "type": "loop",
            "config": {
                "loop_type": "foreach",
                "collection_field": "items",
                "item_variable": "item",
                "max_iterations": 100
            }
        }
    """

    STEP_TYPE = "loop"
    DISPLAY_NAME = "Loop"
    DESCRIPTION = "Iterate over collections or conditions"
    CATEGORY = "flow_control"

    async def __call__(
        self,
        ctx: StepContext,
        config: dict[str, Any],
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute loop."""
        loop_config = self._parse_config(config)

        if loop_config.loop_type == "foreach":
            return await self._execute_foreach(ctx, loop_config, input_data)
        elif loop_config.loop_type == "while":
            return await self._execute_while(ctx, loop_config, input_data)
        elif loop_config.loop_type == "until":
            return await self._execute_until(ctx, loop_config, input_data)
        elif loop_config.loop_type == "range":
            return await self._execute_range(ctx, loop_config, input_data)
        else:
            raise ValueError(f"Unknown loop type: {loop_config.loop_type}")

    def _parse_config(self, config: dict) -> LoopConfig:
        """Parse configuration dict into LoopConfig."""
        return LoopConfig(
            loop_type=config.get("loop_type", "foreach"),
            collection_field=config.get("collection_field", "items"),
            item_variable=config.get("item_variable", "item"),
            index_variable=config.get("index_variable", "index"),
            condition=config.get("condition", ""),
            range_start=config.get("range_start", 0),
            range_end=config.get("range_end", 10),
            range_step=config.get("range_step", 1),
            max_iterations=config.get("max_iterations", 1000),
            continue_on_error=config.get("continue_on_error", False),
            aggregate=config.get("aggregate", True),
            break_condition=config.get("break_condition", ""),
        )

    async def _execute_foreach(
        self,
        ctx: StepContext,
        config: LoopConfig,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute for-each loop."""
        collection = input_data.get(config.collection_field, [])

        if not isinstance(collection, (list, tuple)):
            raise ValueError(
                f"Expected list for '{config.collection_field}', "
                f"got {type(collection).__name__}"
            )

        results = []
        errors = []
        iterations = 0

        for index, item in enumerate(collection):
            if iterations >= config.max_iterations:
                break

            iterations += 1

            try:
                # Create iteration context
                iteration_data = {
                    **input_data,
                    config.item_variable: item,
                    config.index_variable: index,
                    "_loop_iteration": index,
                    "_loop_total": len(collection),
                }

                # Execute body (placeholder - actual body execution handled by runner)
                result = {
                    config.item_variable: item,
                    config.index_variable: index,
                    "processed": True,
                }

                results.append(result)

                # Check break condition
                if config.break_condition and self._evaluate_condition(
                    config.break_condition, {**iteration_data, "result": result}
                ):
                    break

            except Exception as e:
                if config.continue_on_error:
                    errors.append({
                        "index": index,
                        "item": item,
                        "error": str(e),
                    })
                else:
                    raise

        return {
            "results": results if config.aggregate else results[-1] if results else None,
            "iterations": iterations,
            "errors": errors if errors else None,
            "completed": iterations == len(collection),
        }

    async def _execute_while(
        self,
        ctx: StepContext,
        config: LoopConfig,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute while loop."""
        results = []
        errors = []
        iterations = 0

        while iterations < config.max_iterations:
            # Evaluate condition
            if not self._evaluate_condition(config.condition, input_data):
                break

            iterations += 1

            try:
                # Create iteration context
                iteration_data = {
                    **input_data,
                    config.index_variable: iterations - 1,
                    "_loop_iteration": iterations - 1,
                }

                result = {
                    config.index_variable: iterations - 1,
                    "processed": True,
                }

                results.append(result)

                # Check break condition
                if config.break_condition and self._evaluate_condition(
                    config.break_condition, {**iteration_data, "result": result}
                ):
                    break

            except Exception as e:
                if config.continue_on_error:
                    errors.append({
                        "iteration": iterations - 1,
                        "error": str(e),
                    })
                else:
                    raise

        return {
            "results": results if config.aggregate else results[-1] if results else None,
            "iterations": iterations,
            "errors": errors if errors else None,
        }

    async def _execute_until(
        self,
        ctx: StepContext,
        config: LoopConfig,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute until loop (do-while with negated condition)."""
        results = []
        errors = []
        iterations = 0

        while iterations < config.max_iterations:
            iterations += 1

            try:
                iteration_data = {
                    **input_data,
                    config.index_variable: iterations - 1,
                    "_loop_iteration": iterations - 1,
                }

                result = {
                    config.index_variable: iterations - 1,
                    "processed": True,
                }

                results.append(result)

                # Check termination condition
                if self._evaluate_condition(config.condition, {**iteration_data, "result": result}):
                    break

            except Exception as e:
                if config.continue_on_error:
                    errors.append({
                        "iteration": iterations - 1,
                        "error": str(e),
                    })
                else:
                    raise

        return {
            "results": results if config.aggregate else results[-1] if results else None,
            "iterations": iterations,
            "errors": errors if errors else None,
        }

    async def _execute_range(
        self,
        ctx: StepContext,
        config: LoopConfig,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute range loop."""
        results = []
        errors = []
        iterations = 0

        for i in range(config.range_start, config.range_end, config.range_step):
            if iterations >= config.max_iterations:
                break

            iterations += 1

            try:
                iteration_data = {
                    **input_data,
                    config.index_variable: i,
                    "_loop_iteration": iterations - 1,
                }

                result = {
                    config.index_variable: i,
                    "processed": True,
                }

                results.append(result)

                # Check break condition
                if config.break_condition and self._evaluate_condition(
                    config.break_condition, {**iteration_data, "result": result}
                ):
                    break

            except Exception as e:
                if config.continue_on_error:
                    errors.append({
                        "index": i,
                        "error": str(e),
                    })
                else:
                    raise

        return {
            "results": results if config.aggregate else results[-1] if results else None,
            "iterations": iterations,
            "errors": errors if errors else None,
        }

    def _evaluate_condition(self, condition: str, data: dict) -> bool:
        """
        Evaluate a condition expression.

        Supports:
        - "true" / "false" literals
        - "<key>" - check if key exists and is truthy
        - "<key> == <value>" - equality check
        - "<key> != <value>" - inequality check
        - "<key> > <value>" - greater than (numeric)
        - "<key> < <value>" - less than (numeric)
        """
        condition = condition.strip()

        if condition.lower() == "true":
            return True
        if condition.lower() == "false":
            return False

        # Parse comparison operators
        for op in ["==", "!=", ">=", "<=", ">", "<"]:
            if op in condition:
                parts = condition.split(op)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()

                    # Get actual value
                    actual = data.get(key)
                    if actual is None:
                        return False

                    # Parse expected value
                    expected: Any
                    if value.lower() == "true":
                        expected = True
                    elif value.lower() == "false":
                        expected = False
                    elif value.isdigit():
                        expected = int(value)
                    else:
                        try:
                            expected = float(value)
                        except ValueError:
                            expected = value.strip("'\"")

                    # Compare
                    if op == "==":
                        return actual == expected
                    elif op == "!=":
                        return actual != expected
                    elif op == ">":
                        return actual > expected
                    elif op == "<":
                        return actual < expected
                    elif op == ">=":
                        return actual >= expected
                    elif op == "<=":
                        return actual <= expected

        # Simple key check
        return bool(data.get(condition))
