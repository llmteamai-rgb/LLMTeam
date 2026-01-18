"""
Widget Protocol for KorpOS UI Integration.

Defines the interface for UI components that can be rendered
in the Canvas workflow editor and handle user interactions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable


class WidgetType(str, Enum):
    """Types of widgets supported."""

    # Input widgets
    TEXT_INPUT = "text_input"
    NUMBER_INPUT = "number_input"
    SELECT = "select"
    MULTI_SELECT = "multi_select"
    CHECKBOX = "checkbox"
    DATE_PICKER = "date_picker"
    FILE_UPLOAD = "file_upload"

    # Display widgets
    TEXT_DISPLAY = "text_display"
    JSON_VIEWER = "json_viewer"
    TABLE = "table"
    CHART = "chart"
    MARKDOWN = "markdown"
    IMAGE = "image"

    # Action widgets
    BUTTON = "button"
    BUTTON_GROUP = "button_group"

    # Layout widgets
    CARD = "card"
    ACCORDION = "accordion"
    TABS = "tabs"

    # Custom
    CUSTOM = "custom"


class IntentType(str, Enum):
    """Types of user intents from widget interactions."""

    # Data intents
    SUBMIT = "submit"
    UPDATE = "update"
    CANCEL = "cancel"
    RESET = "reset"

    # Navigation intents
    NAVIGATE = "navigate"
    EXPAND = "expand"
    COLLAPSE = "collapse"

    # Action intents
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"

    # Custom
    CUSTOM = "custom"


@dataclass
class WidgetIntent:
    """
    Represents a user intent captured from widget interaction.

    Attributes:
        intent_type: Type of the intent
        widget_id: ID of the widget that generated the intent
        payload: Data associated with the intent
        metadata: Additional context (e.g., timestamp, user_id)
    """

    intent_type: IntentType
    widget_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "intent_type": self.intent_type.value,
            "widget_id": self.widget_id,
            "payload": self.payload,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WidgetIntent":
        """Create from dictionary."""
        return cls(
            intent_type=IntentType(data["intent_type"]),
            widget_id=data["widget_id"],
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class WidgetRenderResult:
    """
    Result of rendering a widget.

    Attributes:
        widget_id: Unique identifier for this widget instance
        widget_type: Type of widget to render
        props: Properties to pass to the widget component
        children: Nested widgets (for layout widgets)
        style: CSS-like style properties
        events: Event handlers to attach
    """

    widget_id: str
    widget_type: WidgetType
    props: dict[str, Any] = field(default_factory=dict)
    children: list["WidgetRenderResult"] = field(default_factory=list)
    style: dict[str, Any] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "widget_id": self.widget_id,
            "widget_type": self.widget_type.value,
            "props": self.props,
            "children": [c.to_dict() for c in self.children],
            "style": self.style,
            "events": self.events,
        }


@dataclass
class IntentResult:
    """
    Result of handling a widget intent.

    Attributes:
        success: Whether the intent was handled successfully
        output: Output data from handling the intent
        error: Error message if handling failed
        next_render: Optional new render result to update UI
        side_effects: List of side effects to execute
    """

    success: bool
    output: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    next_render: Optional[WidgetRenderResult] = None
    side_effects: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "next_render": self.next_render.to_dict() if self.next_render else None,
            "side_effects": self.side_effects,
        }


@runtime_checkable
class WidgetComponent(Protocol):
    """
    Protocol for widget components.

    Implement this protocol to create custom widgets that can be
    rendered in the KorpOS Canvas UI and handle user interactions.

    Example:
        class ApprovalWidget:
            def __init__(self, task_id: str, title: str):
                self.task_id = task_id
                self.title = title

            def render(self, context: dict[str, Any]) -> WidgetRenderResult:
                return WidgetRenderResult(
                    widget_id=f"approval-{self.task_id}",
                    widget_type=WidgetType.CARD,
                    props={
                        "title": self.title,
                        "description": context.get("description", ""),
                    },
                    children=[
                        WidgetRenderResult(
                            widget_id=f"approve-btn-{self.task_id}",
                            widget_type=WidgetType.BUTTON,
                            props={"label": "Approve", "variant": "primary"},
                            events=["click"],
                        ),
                        WidgetRenderResult(
                            widget_id=f"reject-btn-{self.task_id}",
                            widget_type=WidgetType.BUTTON,
                            props={"label": "Reject", "variant": "danger"},
                            events=["click"],
                        ),
                    ],
                )

            async def handle_intent(
                self, intent: WidgetIntent, context: dict[str, Any]
            ) -> IntentResult:
                if intent.intent_type == IntentType.APPROVE:
                    return IntentResult(success=True, output={"approved": True})
                elif intent.intent_type == IntentType.REJECT:
                    return IntentResult(success=True, output={"approved": False})
                return IntentResult(success=False, error="Unknown intent")
    """

    def render(self, context: dict[str, Any]) -> WidgetRenderResult:
        """
        Render the widget based on current context.

        Args:
            context: Current execution context and data

        Returns:
            WidgetRenderResult describing the UI to render
        """
        ...

    async def handle_intent(
        self, intent: WidgetIntent, context: dict[str, Any]
    ) -> IntentResult:
        """
        Handle a user intent from widget interaction.

        Args:
            intent: The captured user intent
            context: Current execution context and data

        Returns:
            IntentResult with the outcome of handling the intent
        """
        ...


class BaseWidget(ABC):
    """
    Abstract base class for widgets.

    Provides common functionality and enforces the WidgetComponent protocol.
    """

    def __init__(self, widget_id: str):
        """
        Initialize widget.

        Args:
            widget_id: Unique identifier for this widget
        """
        self.widget_id = widget_id

    @abstractmethod
    def render(self, context: dict[str, Any]) -> WidgetRenderResult:
        """Render the widget."""
        pass

    @abstractmethod
    async def handle_intent(
        self, intent: WidgetIntent, context: dict[str, Any]
    ) -> IntentResult:
        """Handle a user intent."""
        pass


class TextInputWidget(BaseWidget):
    """Simple text input widget."""

    def __init__(
        self,
        widget_id: str,
        label: str = "",
        placeholder: str = "",
        required: bool = False,
        max_length: Optional[int] = None,
    ):
        super().__init__(widget_id)
        self.label = label
        self.placeholder = placeholder
        self.required = required
        self.max_length = max_length

    def render(self, context: dict[str, Any]) -> WidgetRenderResult:
        """Render text input."""
        return WidgetRenderResult(
            widget_id=self.widget_id,
            widget_type=WidgetType.TEXT_INPUT,
            props={
                "label": self.label,
                "placeholder": self.placeholder,
                "required": self.required,
                "maxLength": self.max_length,
                "value": context.get("value", ""),
            },
            events=["change", "blur"],
        )

    async def handle_intent(
        self, intent: WidgetIntent, context: dict[str, Any]
    ) -> IntentResult:
        """Handle text input intent."""
        if intent.intent_type == IntentType.UPDATE:
            value = intent.payload.get("value", "")

            # Validation
            if self.required and not value:
                return IntentResult(success=False, error="This field is required")

            if self.max_length and len(value) > self.max_length:
                return IntentResult(
                    success=False,
                    error=f"Maximum length is {self.max_length} characters",
                )

            return IntentResult(success=True, output={"value": value})

        return IntentResult(success=False, error=f"Unknown intent: {intent.intent_type}")


class ButtonWidget(BaseWidget):
    """Button widget for actions."""

    def __init__(
        self,
        widget_id: str,
        label: str,
        intent_type: IntentType = IntentType.SUBMIT,
        variant: str = "primary",
        disabled: bool = False,
    ):
        super().__init__(widget_id)
        self.label = label
        self.intent_type = intent_type
        self.variant = variant
        self.disabled = disabled

    def render(self, context: dict[str, Any]) -> WidgetRenderResult:
        """Render button."""
        return WidgetRenderResult(
            widget_id=self.widget_id,
            widget_type=WidgetType.BUTTON,
            props={
                "label": self.label,
                "variant": self.variant,
                "disabled": self.disabled or context.get("disabled", False),
                "intent_type": self.intent_type.value,
            },
            events=["click"],
        )

    async def handle_intent(
        self, intent: WidgetIntent, context: dict[str, Any]
    ) -> IntentResult:
        """Handle button click."""
        if intent.intent_type == self.intent_type:
            return IntentResult(
                success=True,
                output={"clicked": True, "intent": self.intent_type.value},
            )

        return IntentResult(success=True, output={"clicked": True})


class ApprovalWidget(BaseWidget):
    """
    Approval widget for human-in-the-loop tasks.

    Renders approve/reject buttons with optional comment field.
    """

    def __init__(
        self,
        widget_id: str,
        title: str,
        description: str = "",
        require_comment: bool = False,
    ):
        super().__init__(widget_id)
        self.title = title
        self.description = description
        self.require_comment = require_comment

    def render(self, context: dict[str, Any]) -> WidgetRenderResult:
        """Render approval widget."""
        children = []

        # Comment field if required
        if self.require_comment:
            children.append(
                WidgetRenderResult(
                    widget_id=f"{self.widget_id}-comment",
                    widget_type=WidgetType.TEXT_INPUT,
                    props={
                        "label": "Comment",
                        "placeholder": "Add a comment...",
                        "multiline": True,
                    },
                    events=["change"],
                )
            )

        # Buttons
        children.extend([
            WidgetRenderResult(
                widget_id=f"{self.widget_id}-approve",
                widget_type=WidgetType.BUTTON,
                props={"label": "Approve", "variant": "success"},
                events=["click"],
            ),
            WidgetRenderResult(
                widget_id=f"{self.widget_id}-reject",
                widget_type=WidgetType.BUTTON,
                props={"label": "Reject", "variant": "danger"},
                events=["click"],
            ),
        ])

        return WidgetRenderResult(
            widget_id=self.widget_id,
            widget_type=WidgetType.CARD,
            props={
                "title": self.title,
                "description": self.description or context.get("description", ""),
            },
            children=children,
        )

    async def handle_intent(
        self, intent: WidgetIntent, context: dict[str, Any]
    ) -> IntentResult:
        """Handle approval intent."""
        comment = intent.payload.get("comment", "")

        if self.require_comment and not comment:
            return IntentResult(success=False, error="Comment is required")

        if intent.intent_type == IntentType.APPROVE:
            return IntentResult(
                success=True,
                output={"approved": True, "comment": comment},
            )
        elif intent.intent_type == IntentType.REJECT:
            return IntentResult(
                success=True,
                output={"approved": False, "comment": comment},
            )

        return IntentResult(success=False, error=f"Unknown intent: {intent.intent_type}")


# Widget registry for dynamic widget creation
_widget_registry: dict[str, type[BaseWidget]] = {
    "text_input": TextInputWidget,
    "button": ButtonWidget,
    "approval": ApprovalWidget,
}


def register_widget(name: str, widget_class: type[BaseWidget]) -> None:
    """Register a custom widget class."""
    _widget_registry[name] = widget_class


def create_widget(name: str, widget_id: str, **kwargs) -> BaseWidget:
    """Create a widget instance by name."""
    if name not in _widget_registry:
        raise ValueError(f"Unknown widget type: {name}")

    widget_class = _widget_registry[name]
    return widget_class(widget_id=widget_id, **kwargs)


def list_widgets() -> list[str]:
    """List all registered widget types."""
    return list(_widget_registry.keys())
