"""
LLMTeam base exceptions.
"""


class LLMTeamError(Exception):
    """Base exception for all LLMTeam errors."""
    pass


class NoTeamError(LLMTeamError):
    """Raised when team is not found or not configured."""
    pass


class NoGroupError(LLMTeamError):
    """Raised when group is not found or not configured."""
    pass


class NoOrchestratorError(LLMTeamError):
    """Raised when orchestrator is not found or not configured."""
    pass


# Note: ResourceNotFoundError is defined in llmteam.runtime.exceptions
# to maintain proper exception hierarchy for runtime context errors.
# Use: from llmteam.runtime import ResourceNotFoundError
