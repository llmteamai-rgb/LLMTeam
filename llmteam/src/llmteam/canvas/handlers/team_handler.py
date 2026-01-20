"""
Team Handler for Canvas.

Allows Canvas to invoke entire agent teams as workflow steps,
delegating orchestration to the team's LLMTeam container (v3.0.0).
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

from llmteam.runtime import StepContext
from llmteam.observability import get_logger

if TYPE_CHECKING:
    from llmteam.team import LLMTeam

logger = get_logger(__name__)


class TeamNotFoundError(Exception):
    """Raised when referenced team is not found."""
    pass


class TeamHandler:
    """
    Handler for team step type.

    Enables Canvas workflows to invoke agent teams as steps,
    bridging the Canvas execution engine with team orchestration.

    Config:
        team_ref: Reference to registered team (required)
        input_mapping: Optional mapping from step input to team input
        output_mapping: Optional mapping from team output to step output
        timeout: Optional timeout in seconds

    Example segment JSON:
        {
            "step_id": "analysis_team",
            "type": "team",
            "config": {
                "team_ref": "data_analysis_team",
                "input_mapping": {
                    "query": "input.user_query",
                    "context": "input.context"
                },
                "output_mapping": {
                    "result": "analysis_result",
                    "confidence": "confidence_score"
                }
            }
        }

    Usage:
        # Register team in runtime context
        runtime.register_team("data_analysis_team", analysis_team)

        # TeamHandler will resolve and invoke it
        handler = TeamHandler()
        result = await handler(ctx, config, input_data)
    """

    def __init__(self) -> None:
        """Initialize TeamHandler."""
        pass

    async def __call__(
        self,
        ctx: StepContext,
        config: Dict[str, Any],
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute team step.

        Args:
            ctx: Step context with runtime resources
            config: Step configuration:
                - team_ref: Team reference name (required)
                - input_mapping: Input field mappings (optional)
                - output_mapping: Output field mappings (optional)
                - timeout: Execution timeout in seconds (optional)
            input_data: Input data from previous step

        Returns:
            Dict with 'output' key containing team execution result

        Raises:
            TeamNotFoundError: If team_ref is not found in runtime
        """
        team_ref = config.get("team_ref")
        if not team_ref:
            raise ValueError("team_ref is required in team step config")

        input_mapping = config.get("input_mapping", {})
        output_mapping = config.get("output_mapping", {})

        logger.debug(f"TeamHandler: resolving team '{team_ref}'")

        # Resolve team from runtime context
        team = self._resolve_team(ctx, team_ref)
        if team is None:
            raise TeamNotFoundError(f"Team '{team_ref}' not found in runtime context")

        # Map input data
        mapped_input = self._apply_input_mapping(input_data, input_mapping)
        logger.debug(f"TeamHandler: mapped input for '{team_ref}': {list(mapped_input.keys())}")

        # Generate run_id from context
        run_id = f"{ctx.step_id}_{ctx.instance_id}" if hasattr(ctx, 'instance_id') else ctx.step_id

        # Execute team using new LLMTeam.run() API (v3.0.0)
        logger.info(f"TeamHandler: invoking team '{team_ref}' with run_id '{run_id}'")
        try:
            team_result = await team.run(mapped_input, run_id=run_id)
        except Exception as e:
            logger.error(f"TeamHandler: team '{team_ref}' execution failed: {e}")
            raise

        # Check for team execution failure
        if not team_result.success:
            error_msg = team_result.error or "Unknown team execution error"
            logger.error(f"TeamHandler: team '{team_ref}' failed: {error_msg}")
            raise RuntimeError(f"Team execution failed: {error_msg}")

        # Map output data from TeamResult
        mapped_output = self._apply_output_mapping(team_result.output, output_mapping)
        logger.debug(
            f"TeamHandler: team '{team_ref}' completed successfully "
            f"(iterations={team_result.iterations}, agents={team_result.agents_called})"
        )

        return {
            "output": mapped_output,
            "team_metadata": {
                "iterations": team_result.iterations,
                "agents_called": team_result.agents_called,  # v4.0.0: renamed from agents_invoked
                "escalations": team_result.escalations,
            }
        }

    def _resolve_team(self, ctx: StepContext, team_ref: str) -> Optional["LLMTeam"]:
        """
        Resolve team from runtime context.

        Uses the StepContext.get_team() method added in v3.0.0.

        Args:
            ctx: Step context
            team_ref: Team reference name

        Returns:
            LLMTeam instance or None if not found
        """
        # Use StepContext.get_team() which delegates to RuntimeContext (v3.0.0)
        team = ctx.get_team(team_ref)
        if team is not None:
            return team

        # Fallback: try runtime directly for backwards compatibility
        runtime = getattr(ctx, 'runtime', None)
        if runtime is None:
            logger.warning("No runtime context available for team resolution")
            return None

        # Check if runtime has team registry via get_team
        if hasattr(runtime, 'get_team'):
            return runtime.get_team(team_ref)

        return None

    def _apply_input_mapping(
        self,
        input_data: Dict[str, Any],
        mapping: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Apply input mapping to transform step input to team input.

        Args:
            input_data: Original input data
            mapping: Field mapping (target_field -> source_path)

        Returns:
            Mapped input data
        """
        if not mapping:
            return input_data.copy()

        result = {}
        for target_field, source_path in mapping.items():
            value = self._get_nested_value(input_data, source_path)
            if value is not None:
                result[target_field] = value

        # Include unmapped fields from input
        for key, value in input_data.items():
            if key not in result and key not in mapping.values():
                result[key] = value

        return result

    def _apply_output_mapping(
        self,
        output_data: Dict[str, Any],
        mapping: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Apply output mapping to transform team output to step output.

        Args:
            output_data: Team output data
            mapping: Field mapping (target_field -> source_field)

        Returns:
            Mapped output data
        """
        if not mapping:
            return output_data

        result = {}
        for target_field, source_field in mapping.items():
            if source_field in output_data:
                result[target_field] = output_data[source_field]

        # Include unmapped fields from output
        for key, value in output_data.items():
            if key not in result and key not in mapping.values():
                result[key] = value

        return result

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """
        Get value from nested dictionary using dot notation.

        Args:
            data: Source dictionary
            path: Dot-separated path (e.g., "input.user_query")

        Returns:
            Value at path or None if not found
        """
        parts = path.split(".")
        current = data

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current
