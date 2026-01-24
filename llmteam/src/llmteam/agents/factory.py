"""
Agent factory.

Creates agents from configuration dictionaries.
"""

from typing import TYPE_CHECKING, Any, Dict, Type, Union

from llmteam.agents.types import AgentType
from llmteam.agents.config import (
    AgentConfig,
    LLMAgentConfig,
    RAGAgentConfig,
    KAGAgentConfig,
)
from llmteam.agents.base import BaseAgent
from llmteam.agents.llm_agent import LLMAgent
from llmteam.agents.rag_agent import RAGAgent
from llmteam.agents.kag_agent import KAGAgent

if TYPE_CHECKING:
    from llmteam.team import LLMTeam


class AgentFactory:
    """
    Factory for creating agents.

    Used inside LLMTeam.add_agent().

    RESTRICTION: Only 3 agent types (LLM, RAG, KAG).
    Custom agents are not supported - all external logic
    should be executed outside LLMTeam.
    """

    _agent_classes: Dict[AgentType, Type[BaseAgent]] = {
        AgentType.LLM: LLMAgent,
        AgentType.RAG: RAGAgent,
        AgentType.KAG: KAGAgent,
    }

    _config_classes: Dict[AgentType, Type[AgentConfig]] = {
        AgentType.LLM: LLMAgentConfig,
        AgentType.RAG: RAGAgentConfig,
        AgentType.KAG: KAGAgentConfig,
    }

    @classmethod
    def create(
        cls,
        team: "LLMTeam",
        config: Union[Dict[str, Any], AgentConfig],
    ) -> BaseAgent:
        """
        Create agent from configuration.

        Args:
            team: Owner team
            config: Dict or AgentConfig

        Returns:
            Agent instance

        Raises:
            ValueError: If agent type is not LLM, RAG or KAG
        """
        # Parse config
        if isinstance(config, dict):
            type_str = config.get("type", "llm")
            try:
                agent_type = AgentType(type_str)
            except ValueError:
                raise ValueError(
                    f"Unknown agent type: '{type_str}'. "
                    f"Only 'llm', 'rag', 'kag' are supported. "
                    f"Custom logic should be implemented outside LLMTeam."
                )

            config_class = cls._config_classes[agent_type]

            # Get known fields from dataclass
            import dataclasses

            known_fields = {f.name for f in dataclasses.fields(config_class)}
            # Also include parent fields
            for parent in config_class.__mro__:
                if dataclasses.is_dataclass(parent):
                    known_fields.update(f.name for f in dataclasses.fields(parent))

            # Filter to known fields only
            filtered = {}
            for k, v in config.items():
                if k in known_fields:
                    # Convert type string to enum if needed
                    if k == "type" and isinstance(v, str):
                        filtered[k] = AgentType(v)
                    elif k == "mode" and isinstance(v, str):
                        from llmteam.agents.types import AgentMode

                        filtered[k] = AgentMode(v)
                    elif k == "retry_policy" and isinstance(v, dict):
                        from llmteam.agents.retry import RetryPolicy

                        filtered[k] = RetryPolicy(**v)
                    elif k == "circuit_breaker" and isinstance(v, dict):
                        from llmteam.agents.retry import CircuitBreakerPolicy

                        filtered[k] = CircuitBreakerPolicy(**v)
                    elif k == "tools" and isinstance(v, list):
                        from llmteam.tools import ToolDefinition, ToolParameter, ParamType

                        tool_defs = []
                        for tool_dict in v:
                            if isinstance(tool_dict, dict):
                                params = []
                                for p in tool_dict.get("parameters", []):
                                    params.append(ToolParameter(
                                        name=p["name"],
                                        type=ParamType(p.get("type", "string")),
                                        description=p.get("description", ""),
                                        required=p.get("required", True),
                                        default=p.get("default"),
                                    ))
                                tool_defs.append(ToolDefinition(
                                    name=tool_dict["name"],
                                    description=tool_dict.get("description", ""),
                                    parameters=params,
                                    handler=tool_dict.get("handler"),
                                ))
                            else:
                                tool_defs.append(tool_dict)
                        filtered[k] = tool_defs
                    else:
                        filtered[k] = v

            # Ensure required fields
            if "role" not in filtered:
                filtered["role"] = config.get("role", config.get("id", "agent"))

            parsed_config = config_class(**filtered)
        else:
            parsed_config = config
            agent_type = parsed_config.type

        # Get agent class
        agent_class = cls._agent_classes.get(agent_type)
        if not agent_class:
            raise ValueError(
                f"Unknown agent type: {agent_type}. " f"Only LLM, RAG, KAG are supported."
            )

        # Create agent
        return agent_class(team=team, config=parsed_config)

    @classmethod
    def get_supported_types(cls) -> list:
        """Get list of supported agent types."""
        return [t.value for t in AgentType]
