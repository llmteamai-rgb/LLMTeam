"""
Runtime Context module for LLMTeam v2.0.0.

Provides dependency injection for enterprise resources.
Steps receive dependencies through injection, configs contain only references/IDs.
"""

from llmteam.runtime.protocols import (
    Store,
    Client,
    SecretsProvider,
    LLMProvider,
)
from llmteam.runtime.registries import (
    StoreRegistry,
    ClientRegistry,
    LLMRegistry,
)
from llmteam.runtime.context import (
    RuntimeContext,
    StepContext,
    RuntimeContextManager,
    RuntimeContextFactory,
    current_runtime,
    get_current_runtime,
)
from llmteam.runtime.exceptions import (
    ResourceNotFoundError,
    SecretAccessDeniedError,
    RuntimeContextError,
)

__all__ = [
    # Protocols
    "Store",
    "Client",
    "SecretsProvider",
    "LLMProvider",
    # Registries
    "StoreRegistry",
    "ClientRegistry",
    "LLMRegistry",
    # Context
    "RuntimeContext",
    "StepContext",
    "RuntimeContextManager",
    "RuntimeContextFactory",
    "current_runtime",
    "get_current_runtime",
    # Exceptions
    "ResourceNotFoundError",
    "SecretAccessDeniedError",
    "RuntimeContextError",
]
