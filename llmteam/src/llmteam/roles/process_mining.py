"""
Process mining engine for llmteam.

DEPRECATED: Import from llmteam.mining instead.
This module is kept for backward compatibility.
"""

import warnings

warnings.warn(
    "llmteam.roles.process_mining is deprecated. Use llmteam.mining instead.",
    DeprecationWarning,
    stacklevel=2,
)

from llmteam.mining import (
    ProcessEvent,
    ProcessMetrics,
    ProcessModel,
    ProcessMiningEngine,
    generate_uuid,
)

__all__ = [
    "ProcessEvent",
    "ProcessMetrics",
    "ProcessModel",
    "ProcessMiningEngine",
    "generate_uuid",
]
