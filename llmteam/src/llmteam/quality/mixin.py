"""
QualityAwareLLMMixin for RFC-019.

Provides unified LLM access with quality-based model and parameter selection.
Components using this mixin get quality-aware LLM calls without hardcoding models.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    from llmteam.quality.manager import QualityManager


class QualityAwareLLMMixin:
    """
    Mixin providing quality-aware LLM access.

    Classes using this mixin must implement _get_quality_manager().

    Example:
        class MyComponent(QualityAwareLLMMixin):
            def __init__(self, team):
                self._team = team

            def _get_quality_manager(self):
                return self._team._quality_manager

            async def do_something(self):
                response = await self._quality_complete(
                    prompt="...",
                    complexity="medium",
                )
    """

    _quality_llm: Optional[Any] = None
    _quality_llm_model: Optional[str] = None

    def _get_quality_manager(self) -> "QualityManager":
        """
        Get QualityManager instance.

        Must be implemented by subclass.

        Returns:
            QualityManager instance
        """
        raise NotImplementedError(
            "Subclass must implement _get_quality_manager()"
        )

    def _get_quality_llm(
        self,
        complexity: str = "medium",
        force_refresh: bool = False,
    ) -> Any:
        """
        Get LLM provider with quality-appropriate model.

        Args:
            complexity: Task complexity ("simple", "medium", "complex")
            force_refresh: Force recreation of provider

        Returns:
            LLM provider configured with quality-based model
        """
        manager = self._get_quality_manager()
        model = manager.get_model(complexity)

        if (
            self._quality_llm is None
            or force_refresh
            or self._quality_llm_model != model
        ):
            self._quality_llm = self._create_llm_provider(model)
            self._quality_llm_model = model

        return self._quality_llm

    def _create_llm_provider(self, model: str) -> Any:
        """
        Create LLM provider for given model.

        Prefers team's runtime context if available, falls back to
        direct provider creation based on model name.

        Args:
            model: Model name

        Returns:
            LLM provider instance
        """
        # Try to use team's runtime context
        team = getattr(self, "_team", None)
        if team and hasattr(team, "_runtime") and team._runtime:
            try:
                runtime = team._runtime
                if hasattr(runtime, "resolve_llm"):
                    return runtime.resolve_llm(model)
            except Exception:
                # Try "default" as fallback
                try:
                    if hasattr(runtime, "resolve_llm"):
                        return runtime.resolve_llm("default")
                except Exception:
                    pass

        # Fallback: create provider directly
        if model.startswith("claude"):
            try:
                from llmteam.providers import AnthropicProvider
                return AnthropicProvider(model=model)
            except (ImportError, Exception):
                pass

        # Default to OpenAI
        from llmteam.providers import OpenAIProvider
        return OpenAIProvider(model=model)

    def _get_quality_params(
        self,
        override_temperature: Optional[float] = None,
        override_max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get generation parameters based on quality.

        Args:
            override_temperature: Override temperature if needed
            override_max_tokens: Override max_tokens if needed

        Returns:
            Dict with temperature, max_tokens
        """
        manager = self._get_quality_manager()
        params = manager.get_generation_params()

        if override_temperature is not None:
            params["temperature"] = override_temperature
        if override_max_tokens is not None:
            params["max_tokens"] = override_max_tokens

        return params

    async def _quality_complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        complexity: str = "medium",
        override_temperature: Optional[float] = None,
        override_max_tokens: Optional[int] = None,
    ) -> str:
        """
        Complete prompt using quality-aware LLM.

        Convenience method combining _get_quality_llm() and _get_quality_params().

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            complexity: Task complexity for model selection
            override_temperature: Override temperature
            override_max_tokens: Override max_tokens

        Returns:
            LLM response string
        """
        llm = self._get_quality_llm(complexity=complexity)
        params = self._get_quality_params(
            override_temperature=override_temperature,
            override_max_tokens=override_max_tokens,
        )

        kwargs: Dict[str, Any] = {}
        if system_prompt:
            kwargs["system_prompt"] = system_prompt
        kwargs["temperature"] = params.get("temperature", 0.5)
        kwargs["max_tokens"] = params.get("max_tokens", 1000)

        response = await llm.complete(prompt=prompt, **kwargs)

        # Extract text from response
        if isinstance(response, str):
            return response
        return getattr(response, "text", getattr(response, "content", str(response)))
