"""
Action Registry.

Manages registration and lookup of action handlers.
Provides convenient methods for registering common action types.
"""

from typing import Callable, Dict, List, Optional

from .models import ActionConfig, ActionType, ActionHandler
from .handlers import WebhookActionHandler, FunctionActionHandler


class ActionRegistry:
    """Registry of available actions."""

    def __init__(self) -> None:
        self._handlers: Dict[str, ActionHandler] = {}
        self._configs: Dict[str, ActionConfig] = {}

    def register(self, config: ActionConfig, handler: ActionHandler) -> None:
        """Register an action with its handler."""
        self._configs[config.name] = config
        self._handlers[config.name] = handler

    def register_webhook(
        self,
        name: str,
        url: str,
        method: str = "POST",
        **kwargs,
    ) -> None:
        """
        Register a webhook/REST API action.

        Args:
            name: Action name
            url: Target URL
            method: HTTP method (default: POST)
            **kwargs: Additional ActionConfig parameters
        """
        config = ActionConfig(
            name=name,
            action_type=ActionType.WEBHOOK,
            url=url,
            method=method,
            **kwargs,
        )
        self.register(config, WebhookActionHandler(config))

    def register_function(
        self,
        name: str,
        func: Callable,
        **kwargs,
    ) -> None:
        """
        Register a Python function action.

        Args:
            name: Action name
            func: Function to execute
            **kwargs: Additional ActionConfig parameters
        """
        config = ActionConfig(
            name=name,
            action_type=ActionType.FUNCTION,
            **kwargs,
        )
        self.register(config, FunctionActionHandler(config, func))

    def unregister(self, name: str) -> None:
        """Unregister an action."""
        self._handlers.pop(name, None)
        self._configs.pop(name, None)

    def get_handler(self, name: str) -> Optional[ActionHandler]:
        """Get handler by action name."""
        return self._handlers.get(name)

    def get_config(self, name: str) -> Optional[ActionConfig]:
        """Get config by action name."""
        return self._configs.get(name)

    def has_action(self, name: str) -> bool:
        """Check if action is registered."""
        return name in self._handlers

    def list_actions(self) -> List[str]:
        """List all registered action names."""
        return list(self._handlers.keys())

    def clear(self) -> None:
        """Clear all registered actions."""
        self._handlers.clear()
        self._configs.clear()
