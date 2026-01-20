"""
Legacy roles module for llmteam.

DEPRECATED: This module is kept for backward compatibility only.

Use the new locations instead:
    - llmteam.mining: ProcessMiningEngine, ProcessEvent, ProcessMetrics
    - llmteam.contract: TeamContract, ContractValidationResult
    - llmteam.escalation: EscalationLevel, Escalation, etc.
    - llmteam.team: LLMTeam, LLMGroup (replaces PipelineOrchestrator, GroupOrchestrator)
"""

# Process mining (deprecated, use llmteam.mining)
from llmteam.roles.process_mining import (
    ProcessEvent,
    ProcessMetrics,
    ProcessModel,
    ProcessMiningEngine,
)

# Team contract (deprecated, use llmteam.contract)
from llmteam.roles.contract import (
    TeamContract,
    ContractValidationResult,
    ContractValidationError,
)

__all__ = [
    # Process Mining (deprecated)
    "ProcessEvent",
    "ProcessMetrics",
    "ProcessModel",
    "ProcessMiningEngine",
    # Team Contract (deprecated)
    "TeamContract",
    "ContractValidationResult",
    "ContractValidationError",
]
