"""
Exceptions.
"""
class LLMTeamError(Exception): pass
class NoTeamError(LLMTeamError): pass
class NoGroupError(LLMTeamError): pass
class NoOrchestratorError(LLMTeamError): pass
class ResourceNotFoundError(LLMTeamError): pass # Should probably move existing one here or import
