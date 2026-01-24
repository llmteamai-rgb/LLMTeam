"""
Agents package.

Provides typed agent system for LLMTeam.
"""

from llmteam.agents.types import AgentType, AgentMode, AgentStatus
from llmteam.agents.config import (
    AgentConfig,
    LLMAgentConfig,
    RAGAgentConfig,
    KAGAgentConfig,
)
from llmteam.agents.state import AgentState
from llmteam.agents.result import AgentResult, RAGResult, KAGResult
from llmteam.agents.factory import AgentFactory
from llmteam.agents.presets import (
    create_orchestrator_config,
    create_group_orchestrator_config,
    create_summarizer_config,
    create_reviewer_config,
    create_rag_config,
    create_kag_config,
)
from llmteam.agents.report import AgentReport
from llmteam.agents.retry import (
    RetryPolicy,
    CircuitBreakerPolicy,
    RetryMetrics,
    AgentRetryExecutor,
)
from llmteam.agents.orchestrator import (
    TeamOrchestrator,
    OrchestratorMode,
    OrchestratorScope,
    OrchestratorConfig,
    RoutingDecision,
    RecoveryDecision,
    RecoveryAction,
)

# BaseAgent is NOT exported - internal only
# LLMAgent, RAGAgent, KAGAgent are NOT exported - created via AgentFactory

__all__ = [
    # Types
    "AgentType",
    "AgentMode",
    "AgentStatus",
    # Config
    "AgentConfig",
    "LLMAgentConfig",
    "RAGAgentConfig",
    "KAGAgentConfig",
    # State
    "AgentState",
    # Result
    "AgentResult",
    "RAGResult",
    "KAGResult",
    # Factory
    "AgentFactory",
    # Presets
    "create_orchestrator_config",
    "create_group_orchestrator_config",
    "create_summarizer_config",
    "create_reviewer_config",
    "create_rag_config",
    "create_kag_config",
    # Report
    "AgentReport",
    # RFC-012: Retry & Circuit Breaker
    "RetryPolicy",
    "CircuitBreakerPolicy",
    "RetryMetrics",
    "AgentRetryExecutor",
    # Orchestrator
    "TeamOrchestrator",
    "OrchestratorMode",
    "OrchestratorScope",
    "OrchestratorConfig",
    "RoutingDecision",
    "RecoveryDecision",
    "RecoveryAction",
]
