"""
Parallel Handlers.

Handles parallel split (fan-out) and join (fan-in) operations.
"""

from typing import Any, List

from llmteam.runtime import StepContext
from llmteam.observability import get_logger


logger = get_logger(__name__)


class ParallelSplitHandler:
    """
    Handler for parallel_split step type.

    Fans out input data to multiple output ports for parallel execution.
    """

    def __init__(self) -> None:
        """Initialize handler."""
        pass

    async def __call__(
        self,
        ctx: StepContext,
        config: dict[str, Any],
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Split input to parallel branches.

        Args:
            ctx: Step context
            config: Step configuration:
                - branches: Number of branches (default: 2)
                - split_field: Field to split if input is a list
                - copy_mode: "reference" or "copy" (default: "reference")
            input_data: Input data to distribute

        Returns:
            Dict with branch_1, branch_2, ... keys containing data for each branch
        """
        branches = config.get("branches", 2)
        split_field = config.get("split_field")
        copy_mode = config.get("copy_mode", "reference")

        logger.debug(f"Parallel Split: {branches} branches, split_field={split_field}")

        result = {}

        if split_field and isinstance(input_data, dict):
            # Split a list field across branches
            items = input_data.get(split_field, [])
            if isinstance(items, list):
                # Distribute items across branches
                for i in range(branches):
                    branch_items = items[i::branches]  # Round-robin distribution
                    branch_data = dict(input_data) if copy_mode == "copy" else input_data
                    result[f"branch_{i + 1}"] = {
                        **branch_data,
                        split_field: branch_items,
                        "_branch_index": i,
                        "_total_branches": branches,
                    }
            else:
                # Not a list, just copy to all branches
                for i in range(branches):
                    result[f"branch_{i + 1}"] = input_data
        else:
            # Copy input to all branches
            for i in range(branches):
                branch_data = dict(input_data) if copy_mode == "copy" and isinstance(input_data, dict) else input_data
                if isinstance(branch_data, dict):
                    result[f"branch_{i + 1}"] = {
                        **branch_data,
                        "_branch_index": i,
                        "_total_branches": branches,
                    }
                else:
                    result[f"branch_{i + 1}"] = branch_data

        logger.debug(f"Parallel Split completed: {len(result)} branches")
        return result


class ParallelJoinHandler:
    """
    Handler for parallel_join step type.

    Joins results from parallel branches using a merge strategy.
    """

    def __init__(self) -> None:
        """Initialize handler."""
        pass

    async def __call__(
        self,
        ctx: StepContext,
        config: dict[str, Any],
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Join parallel branch results.

        Args:
            ctx: Step context
            config: Step configuration:
                - merge_strategy: "all", "any", or "first" (default: "all")
                - timeout_per_branch: Timeout for each branch in seconds
            input_data: Dict with branch_1, branch_2, ... results

        Returns:
            Dict with 'output' containing merged results
        """
        merge_strategy = config.get("merge_strategy", "all")

        logger.debug(f"Parallel Join: strategy={merge_strategy}")

        # Collect branch results
        branch_results: List[Any] = []
        for key in sorted(input_data.keys()):
            if key.startswith("branch_"):
                branch_results.append(input_data[key])

        # Also check for numbered ports (branch_1, branch_2, etc.)
        if not branch_results:
            # Try input ports directly
            for key, value in input_data.items():
                if isinstance(value, dict) or value is not None:
                    branch_results.append(value)

        logger.debug(f"Parallel Join: {len(branch_results)} branch results")

        # Apply merge strategy
        if merge_strategy == "first":
            # Return first available result
            result = branch_results[0] if branch_results else {}
        elif merge_strategy == "any":
            # Return first non-null/non-error result
            result = None
            for br in branch_results:
                if br and not (isinstance(br, dict) and "error" in br):
                    result = br
                    break
            if result is None:
                result = branch_results[0] if branch_results else {}
        else:  # "all" strategy
            # Return all results as array
            result = branch_results

        return {
            "output": result,
        }
