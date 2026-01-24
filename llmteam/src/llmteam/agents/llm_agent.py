"""
LLM Agent implementation.

Text generation agent using LLM providers.
RFC-016: Tool execution loop (prompt → LLM → tool_call → execute → LLM → final).
"""

import json
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from llmteam.agents.types import AgentType
from llmteam.agents.config import LLMAgentConfig
from llmteam.agents.result import AgentResult
from llmteam.agents.base import BaseAgent

if TYPE_CHECKING:
    from llmteam.team import LLMTeam
    from llmteam.providers.base import LLMResponse


class LLMAgent(BaseAgent):
    """
    Text generation agent via LLM.

    Automatically:
    - Formats prompt with variables
    - Collects context from RAG/KAG
    - Calls LLM provider (with tool loop if tools configured)
    """

    agent_type = AgentType.LLM

    # Config fields
    prompt: str
    system_prompt: Optional[str]
    model: str
    temperature: float
    max_tokens: int
    max_tool_rounds: int
    use_context: bool
    output_key: str
    output_format: str

    # RFC-017: Optional event callback (set by team.stream())
    _event_callback: Optional[Callable] = None

    def __init__(self, team: "LLMTeam", config: LLMAgentConfig):
        super().__init__(team, config)

        self.prompt = config.prompt
        self.system_prompt = config.system_prompt or self._default_system_prompt()
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.max_tool_rounds = config.max_tool_rounds
        self.use_context = config.use_context
        self.output_key = config.output_key or config.role
        self.output_format = config.output_format
        self._event_callback = None

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
        INTERNAL: Generate text via LLM with optional tool loop.

        RFC-016: If tools are configured, runs a loop:
        1. Send prompt + tool schemas to provider
        2. If response has tool_calls → execute → append result → goto 1
        3. If response is text → return as output
        4. Stops after max_tool_rounds

        Args:
            input_data: Input data from team.run()
            context: Context from mailbox

        Returns:
            AgentResult with output text, tokens_used, tool_calls_made
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

        # Check if we have tools configured
        has_tools = self._tool_executor is not None and len(self._tool_executor.list_tools()) > 0

        if has_tools:
            return await self._execute_with_tools(formatted_prompt, provider)
        else:
            return await self._execute_simple(formatted_prompt, provider)

    async def _execute_simple(self, formatted_prompt: str, provider: Any) -> AgentResult:
        """Simple execution without tools."""
        response = await provider.complete(
            prompt=formatted_prompt,
            system_prompt=self.system_prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return AgentResult(
            output=response if isinstance(response, str) else getattr(response, "text", str(response)),
            output_key=self.output_key,
            success=True,
            tokens_used=getattr(response, "tokens_used", 0) if not isinstance(response, str) else 0,
            model=self.model,
        )

    async def _execute_with_tools(self, formatted_prompt: str, provider: Any) -> AgentResult:
        """
        RFC-016: Execute with tool calling loop.

        Loop: prompt → LLM → tool_calls → execute → LLM → ... → final text.
        """
        # Build initial messages
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": formatted_prompt},
        ]

        # Get tool schemas
        tool_schemas = self._tool_executor.get_schemas()

        total_tokens = 0
        tool_calls_made: List[Dict[str, Any]] = []

        for round_num in range(self.max_tool_rounds):
            # Call provider with tools
            llm_response: "LLMResponse" = await provider.complete_with_tools(
                messages=messages,
                tools=tool_schemas,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            total_tokens += llm_response.tokens_used

            # If no tool calls, we have the final answer
            if not llm_response.has_tool_calls:
                return AgentResult(
                    output=llm_response.content or "",
                    output_key=self.output_key,
                    success=True,
                    tokens_used=total_tokens,
                    model=llm_response.model or self.model,
                    context_payload={
                        "tool_calls_made": tool_calls_made,
                        "tool_rounds": round_num,
                    } if tool_calls_made else None,
                )

            # Process tool calls
            # Add assistant message with tool_calls to conversation
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": llm_response.content}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in llm_response.tool_calls
            ]
            messages.append(assistant_msg)

            # Execute each tool call
            for tc in llm_response.tool_calls:
                # Emit TOOL_CALL event (RFC-017)
                await self._emit_event("tool_call", {
                    "tool_name": tc.name,
                    "arguments": tc.arguments,
                    "call_id": tc.id,
                    "round": round_num,
                })

                # Execute tool
                result = await self._tool_executor.execute(tc.name, tc.arguments)

                tool_output = str(result.output) if result.success else f"Error: {result.error}"

                tool_calls_made.append({
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "output": tool_output[:500],
                    "success": result.success,
                    "round": round_num,
                })

                # Emit TOOL_RESULT event (RFC-017)
                await self._emit_event("tool_result", {
                    "tool_name": tc.name,
                    "call_id": tc.id,
                    "output": tool_output[:200],
                    "success": result.success,
                    "round": round_num,
                })

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_output,
                })

        # Max rounds exceeded — return last content or summary
        final_content = llm_response.content or f"[Tool loop ended after {self.max_tool_rounds} rounds]"
        return AgentResult(
            output=final_content,
            output_key=self.output_key,
            success=True,
            tokens_used=total_tokens,
            model=self.model,
            context_payload={
                "tool_calls_made": tool_calls_made,
                "tool_rounds": self.max_tool_rounds,
                "max_rounds_reached": True,
            },
        )

    async def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        RFC-017: Emit agent-level event via callback.

        Called during tool execution to provide real-time visibility.
        """
        if self._event_callback:
            await self._event_callback(event_type, data, self.agent_id)

    def _get_provider(self):
        """Get LLM provider from runtime context."""
        if hasattr(self._team, "_runtime") and self._team._runtime:
            runtime = self._team._runtime
            try:
                # RuntimeContext uses resolve_llm, StepContext uses get_llm
                if hasattr(runtime, "resolve_llm"):
                    return runtime.resolve_llm(self.model)
                elif hasattr(runtime, "get_llm"):
                    return runtime.get_llm(self.model)
            except Exception:
                # Try "default" as fallback
                try:
                    if hasattr(runtime, "resolve_llm"):
                        return runtime.resolve_llm("default")
                    elif hasattr(runtime, "get_llm"):
                        return runtime.get_llm("default")
                except Exception:
                    pass
        return None
