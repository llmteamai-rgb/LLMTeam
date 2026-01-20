"""
LLM Agent Handler.

Executes LLM prompts using the configured LLM provider.
"""

from typing import Any, Optional
import json

from llmteam.runtime import StepContext
from llmteam.observability import get_logger


logger = get_logger(__name__)


class LLMAgentHandler:
    """
    Handler for llm_agent step type.

    Resolves LLM provider from runtime context, renders prompt template,
    and executes completion.
    """

    def __init__(
        self,
        default_temperature: float = 0.7,
        default_max_tokens: int = 1000,
    ) -> None:
        """
        Initialize handler.

        Args:
            default_temperature: Default temperature for LLM calls
            default_max_tokens: Default max tokens for LLM calls
        """
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

    async def __call__(
        self,
        ctx: StepContext,
        config: dict[str, Any],
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute LLM agent step.

        Args:
            ctx: Step context with runtime resources
            config: Step configuration:
                - llm_ref: Reference to LLM provider (required)
                - prompt_template_id: Prompt template ID (optional)
                - prompt: Direct prompt string (optional)
                - system_prompt: System prompt (optional)
                - temperature: Temperature (optional)
                - max_tokens: Max tokens (optional)
            input_data: Input data for prompt templating

        Returns:
            Dict with 'output' containing LLM response
        """
        llm_ref = config.get("llm_ref", "default")
        prompt_template_id = config.get("prompt_template_id")
        direct_prompt = config.get("prompt", "")
        system_prompt = config.get("system_prompt", "")
        temperature = config.get("temperature", self.default_temperature)
        max_tokens = config.get("max_tokens", self.default_max_tokens)

        logger.debug(
            f"LLM Agent executing: llm_ref={llm_ref}, "
            f"template={prompt_template_id}, temp={temperature}"
        )

        try:
            # Resolve LLM provider from context
            llm = ctx.get_llm(llm_ref)

            # Build prompt
            if prompt_template_id:
                # Load and render template
                prompt = await self._render_template(prompt_template_id, input_data)
            elif direct_prompt:
                # Use direct prompt with variable substitution
                prompt = self._substitute_variables(direct_prompt, input_data)
            else:
                # Default: convert input to prompt
                prompt = self._input_to_prompt(input_data)

            # Execute LLM call
            response = await llm.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            logger.debug(f"LLM Agent completed: response length={len(response)}")

            return {
                "output": response,
            }

        except Exception as e:
            logger.error(f"LLM Agent failed: {e}")
            return {
                "output": "",
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                },
            }

    async def _render_template(
        self,
        template_id: str,
        variables: dict[str, Any],
    ) -> str:
        """
        Render a prompt template with variables.

        Args:
            template_id: Template identifier
            variables: Variables for substitution

        Returns:
            Rendered prompt string
        """
        # Simple template rendering (can be extended with Jinja2 or other engines)
        # For now, just return template_id as prompt with variables
        return f"Template: {template_id}\nVariables: {json.dumps(variables, default=str)}"

    def _substitute_variables(
        self,
        prompt: str,
        variables: dict[str, Any],
    ) -> str:
        """
        Substitute variables in prompt using {variable} syntax.

        Args:
            prompt: Prompt with {variable} placeholders
            variables: Variable values

        Returns:
            Prompt with substituted values
        """
        result = prompt

        # Auto-map 'input' to 'context' for sequential agent flows
        if "input" in variables and "{context}" in result and "context" not in variables:
            input_val = variables["input"]
            if isinstance(input_val, dict) and "output" in input_val:
                variables["context"] = input_val["output"]
            elif isinstance(input_val, str):
                variables["context"] = input_val
            else:
                variables["context"] = str(input_val)

        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            if placeholder in result:
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, default=str)
                result = result.replace(placeholder, str(value))
        return result

    def _input_to_prompt(self, input_data: dict[str, Any]) -> str:
        """
        Convert input data to a prompt string.

        Args:
            input_data: Input data dict

        Returns:
            Prompt string
        """
        if "prompt" in input_data:
            return str(input_data["prompt"])
        if "query" in input_data:
            return str(input_data["query"])
        if "input" in input_data:
            return str(input_data["input"])
        if "text" in input_data:
            return str(input_data["text"])

        # Default: serialize to JSON
        return json.dumps(input_data, default=str)
