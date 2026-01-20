"""
Team Contract Definition.

DEPRECATED: Import from llmteam.contract instead.
This module is kept for backward compatibility.
"""

import warnings

warnings.warn(
    "llmteam.roles.contract is deprecated. Use llmteam.contract instead.",
    DeprecationWarning,
    stacklevel=2,
)

from llmteam.contract import (
    TeamContract,
    ContractValidationResult,
    ContractValidationError,
)

__all__ = [
    "TeamContract",
    "ContractValidationResult",
    "ContractValidationError",
]
