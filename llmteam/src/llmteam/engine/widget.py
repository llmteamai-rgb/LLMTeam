"""
Widget Protocol for UI Integration.

Defines the contract for UI widgets that can be rendered by the host application (e.g., KorpOS).
"""

from typing import Any, Protocol, Dict, Optional, runtime_checkable
from dataclasses import dataclass, field
from abc import abstractmethod
from enum import Enum


class WidgetType(str, Enum):
    BUTTON = "button"
    TEXT_INPUT = "text_input"
    SELECT = "select"
    APPROVAL = "approval"
    DISPLAY = "display"


@dataclass
class WidgetIntent:
    """User intent triggered by widget interaction."""
    widget_id: str
    intent_type: str
    payload: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class WidgetComponent(Protocol):
    """Protocol for KorpOS Widget components."""
    
    widget_id: str
    type: WidgetType
    
    @abstractmethod
    def render(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render widget state from execution data.
        
        Args:
            data: Current step data/state
            
        Returns:
            JSON-serializable UI schema/props
        """
        ...
    
    @abstractmethod
    def handle_intent(self, intent: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert user intent to output data.
        
        Args:
            intent: Intent name (e.g., "click", "submit")
            payload: Intent data
            
        Returns:
            Data to be merged into workflow state, or None if no state change
        """
        ...


@dataclass
class ButtonWidget:
    """Simple clickable button."""
    widget_id: str
    label: str
    action_value: str
    style: str = "primary"
    type: WidgetType = WidgetType.BUTTON

    def render(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "id": self.widget_id,
            "props": {
                "label": self.label,
                "style": self.style,
                "disabled": data.get("disabled", False)
            }
        }

    def handle_intent(self, intent: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if intent == "click":
            return {"action": self.action_value}
        return None


@dataclass
class TextInputWidget:
    """Text input field."""
    widget_id: str
    label: str
    key: str  # State key to bind to
    placeholder: str = ""
    required: bool = False
    type: WidgetType = WidgetType.TEXT_INPUT

    def render(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "id": self.widget_id,
            "props": {
                "label": self.label,
                "value": data.get(self.key, ""),
                "placeholder": self.placeholder,
                "required": self.required
            }
        }

    def handle_intent(self, intent: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if intent == "change":
            return {self.key: payload.get("value")}
        return None


@dataclass
class ApprovalWidget:
    """Approval card with Approve/Reject buttons."""
    widget_id: str
    title: str = "Approval Required"
    type: WidgetType = WidgetType.APPROVAL

    def render(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "id": self.widget_id,
            "props": {
                "title": self.title,
                "context": data.get("context", {}),
                "options": ["approve", "reject"]
            }
        }

    def handle_intent(self, intent: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if intent in ("approve", "reject"):
            return {
                "decision": intent,
                "reason": payload.get("reason", ""),
                "approver": payload.get("user", "unknown")
            }
        return None
