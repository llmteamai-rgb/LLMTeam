"""
LLM Agent implementation.

Text generation agent using LLM providers.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

from llmteam.agents.types import AgentType
from llmteam.agents.config import LLMAgentConfig
from llmteam.agents.result import AgentResult
from llmteam.agents.base import BaseAgent

if TYPE_CHECKING:
    from llmteam.team import LLMTeam


class LLMAgent(BaseAgent):
    """
    Text generation agent via LLM.

    Automatically:
    - Formats prompt with variables
    - Collects context from RAG/KAG
    - Calls LLM provider
    """

    agent_type = AgentType.LLM

    # Config fields
    prompt: str
    system_prompt: Optional[str]
    model: str
    temperature: float
    max_tokens: int
    use_context: bool
    output_key: str
    output_format: str

    def __init__(self, team: "LLMTeam", config: LLMAgentConfig):
        super().__init__(team, config)

        self.prompt = config.prompt
        self.system_prompt = config.system_prompt or self._default_system_prompt()
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.use_context = config.use_context
        self.output_key = config.output_key or config.role
        self.output_format = config.output_format

    def _default_system_prompt(self) -> str:
        """Generate default system prompt."""
        return f"You are {self.name}. {self.description}"

    def _format_prompt(
        self, input_data: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Format prompt with variables and context."""
        # Start with base prompt
        formatted = self.prompt

        # Format with input variables
        try:
            formatted = formatted.format(**input_data)
        except KeyError:
            # Allow missing keys
            pass

        # Add context if enabled
        if self.use_context and context:
            context_parts = []

            # RAG context
            rag_ctx = context.get("_rag_context", [])
            if rag_ctx:
                context_parts.append("## Retrieved Documents:")
                for i, doc in enumerate(rag_ctx[:5], 1):
                    text = doc.get("text", doc.get("content", str(doc)))
                    context_parts.append(f"{i}. {text[:500]}")

            # KAG context
            kag_ctx = context.get("_kag_context", [])
            if kag_ctx:
                context_parts.append("\n## Knowledge Graph:")
                for entity in kag_ctx[:5]:
                    context_parts.append(f"- {entity}")

            if context_parts:
                formatted = "\n".join(context_parts) + "\n\n" + formatted

        return formatted

    async def _execute(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> AgentResult:
        """
        INTERNAL: Generate text via LLM.

        Do NOT call directly - use team.run() instead.

        Args:
            input_data: Input data from team.run()
                Example: {"query": "What is AI?", "style": "academic"}
            context: Context from mailbox
                Example: {
                    "_rag_context": [{"text": "...", "score": 0.9}],
                    "_kag_context": {"entities": [...], "relations": [...]}
                }

        Returns:
            AgentResult:
                output: str (generated text)
                output_key: str (key for saving in results)
                tokens_used: int
                model: str
        """
        # Format prompt
        formatted_prompt = self._format_prompt(input_data, context)

        # Get LLM provider from team's runtime
        provider = self._get_provider()

        if provider is None:
            # Fallback: return formatted prompt as output (for testing)
            return AgentResult(
                output=f"[LLM would generate response for: {formatted_prompt[:200]}...]",
                output_key=self.output_key,
                success=True,
                tokens_used=0,
                model=self.model,
            )

        # Call LLM
        response = await provider.complete(
            prompt=formatted_prompt,
            system_prompt=self.system_prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return AgentResult(
            output=response.text,
            output_key=self.output_key,
            success=True,
            tokens_used=getattr(response, "tokens_used", 0),
            model=self.model,
        )

    def _get_provider(self):
        """Get LLM provider from runtime context."""
        if hasattr(self._team, "_runtime") and self._team._runtime:
            try:
                return self._team._runtime.get_llm("default")
            except Exception:
                pass
        return None
