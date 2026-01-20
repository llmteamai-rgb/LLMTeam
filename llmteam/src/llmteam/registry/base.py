"""
Base registry implementation for LLMTeam.

Provides a generic, type-safe registry pattern for managing entities.
"""

from typing import Callable, Dict, Generic, Iterator, List, Optional, TypeVar

T = TypeVar("T")


class RegistryError(Exception):
    """Base exception for registry errors."""

    pass


class NotFoundError(RegistryError):
    """Raised when an item is not found in the registry."""

    pass


class AlreadyExistsError(RegistryError):
    """Raised when trying to register an item that already exists."""

    pass


class BaseRegistry(Generic[T]):
    """
    Generic type-safe registry for managing entities.

    Provides CRUD operations with optional validation and callbacks.

    Example:
        registry = BaseRegistry[Agent]()
        registry.register("agent1", my_agent)
        agent = registry.get("agent1")
        all_agents = registry.list()
    """

    def __init__(
        self,
        name: str = "registry",
        allow_overwrite: bool = False,
    ):
        """
        Initialize registry.

        Args:
            name: Name of the registry (for error messages).
            allow_overwrite: Whether to allow overwriting existing entries.
        """
        self._name = name
        self._allow_overwrite = allow_overwrite
        self._items: Dict[str, T] = {}
        self._on_register_callbacks: List[Callable[[str, T], None]] = []
        self._on_unregister_callbacks: List[Callable[[str, T], None]] = []

    @property
    def name(self) -> str:
        """Name of the registry."""
        return self._name

    def register(self, key: str, item: T) -> None:
        """
        Register an item in the registry.

        Args:
            key: Unique key for the item.
            item: Item to register.

        Raises:
            AlreadyExistsError: If key exists and overwrite not allowed.
        """
        if key in self._items and not self._allow_overwrite:
            raise AlreadyExistsError(
                f"Item '{key}' already exists in {self._name}. "
                "Set allow_overwrite=True to override."
            )

        self._items[key] = item

        for callback in self._on_register_callbacks:
            callback(key, item)

    def unregister(self, key: str) -> T:
        """
        Remove an item from the registry.

        Args:
            key: Key of item to remove.

        Returns:
            The removed item.

        Raises:
            NotFoundError: If key not found.
        """
        if key not in self._items:
            raise NotFoundError(f"Item '{key}' not found in {self._name}")

        item = self._items.pop(key)

        for callback in self._on_unregister_callbacks:
            callback(key, item)

        return item

    def get(self, key: str) -> T:
        """
        Get an item by key.

        Args:
            key: Key of item to get.

        Returns:
            The item.

        Raises:
            NotFoundError: If key not found.
        """
        if key not in self._items:
            raise NotFoundError(f"Item '{key}' not found in {self._name}")
        return self._items[key]

    def get_optional(self, key: str) -> Optional[T]:
        """
        Get an item by key, returning None if not found.

        Args:
            key: Key of item to get.

        Returns:
            The item or None.
        """
        return self._items.get(key)

    def has(self, key: str) -> bool:
        """
        Check if key exists in registry.

        Args:
            key: Key to check.

        Returns:
            True if key exists.
        """
        return key in self._items

    def list(self) -> List[T]:
        """
        List all items in the registry.

        Returns:
            List of all items.
        """
        return list(self._items.values())

    def keys(self) -> List[str]:
        """
        List all keys in the registry.

        Returns:
            List of all keys.
        """
        return list(self._items.keys())

    def items(self) -> List[tuple]:
        """
        List all key-value pairs.

        Returns:
            List of (key, item) tuples.
        """
        return list(self._items.items())

    def count(self) -> int:
        """
        Count items in registry.

        Returns:
            Number of items.
        """
        return len(self._items)

    def clear(self) -> None:
        """Remove all items from registry."""
        for key in list(self._items.keys()):
            self.unregister(key)

    def on_register(self, callback: Callable[[str, T], None]) -> None:
        """
        Add callback to be called when item is registered.

        Args:
            callback: Function taking (key, item).
        """
        self._on_register_callbacks.append(callback)

    def on_unregister(self, callback: Callable[[str, T], None]) -> None:
        """
        Add callback to be called when item is unregistered.

        Args:
            callback: Function taking (key, item).
        """
        self._on_unregister_callbacks.append(callback)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __getitem__(self, key: str) -> T:
        return self.get(key)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._name!r}, count={len(self._items)})"
