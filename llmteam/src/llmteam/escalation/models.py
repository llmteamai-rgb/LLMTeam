"""
Escalation Models.
"""
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Dict, Any

class EscalationLevel(Enum):
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()
    EMERGENCY = auto()

@dataclass
class EscalationDecision:
    action: str
    target_team: Optional[str] = None
