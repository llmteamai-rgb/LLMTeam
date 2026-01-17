"""
Resource registries for RuntimeContext.

Registries manage collections of resources that can be resolved by ID.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from llmteam.observability import get_logger
from llmteam.runtime.protocols import Store, Client, LLMProvider
from llmteam.runtime.exceptions import ResourceNotFoundError


logger = get_logger(__name__)


@dataclass
class StoreRegistry:
    """Registry of storage backends."""

    _stores: Dict[str, Store] = field(default_factory=dict)

    def register(self, store_id: str, store: Store) -> None:
        """Register a store."""
        self._stores[store_id] = store
        logger.debug(f"Registered store: {store_id}")

    def unregister(self, store_id: str) -> None:
        """Unregister a store."""
        if store_id in self._stores:
            self._stores.pop(store_id, None)
            logger.debug(f"Unregistered store: {store_id}")

    def get(self, store_id: str) -> Store:
        """Get store by ID."""
        if store_id not in self._stores:
            logger.warning(f"Store not found: {store_id}")
            raise ResourceNotFoundError(f"Store '{store_id}' not found")
        return self._stores[store_id]

    def has(self, store_id: str) -> bool:
        """Check if store exists."""
        return store_id in self._stores

    def list(self) -> List[str]:
        """List all store IDs."""
        return list(self._stores.keys())

    def clear(self) -> None:
        """Clear all stores."""
        count = len(self._stores)
        self._stores.clear()
        logger.debug(f"Cleared {count} stores from registry")


@dataclass
class ClientRegistry:
    """Registry of external clients (HTTP, gRPC, etc)."""

    _clients: Dict[str, Client] = field(default_factory=dict)

    def register(self, client_id: str, client: Client) -> None:
        """Register a client."""
        self._clients[client_id] = client
        logger.debug(f"Registered client: {client_id}")

    def unregister(self, client_id: str) -> None:
        """Unregister a client."""
        if client_id in self._clients:
            self._clients.pop(client_id, None)
            logger.debug(f"Unregistered client: {client_id}")

    def get(self, client_id: str) -> Client:
        """Get client by ID."""
        if client_id not in self._clients:
            logger.warning(f"Client not found: {client_id}")
            raise ResourceNotFoundError(f"Client '{client_id}' not found")
        return self._clients[client_id]

    def has(self, client_id: str) -> bool:
        """Check if client exists."""
        return client_id in self._clients

    def list(self) -> List[str]:
        """List all client IDs."""
        return list(self._clients.keys())

    def clear(self) -> None:
        """Clear all clients."""
        count = len(self._clients)
        self._clients.clear()
        logger.debug(f"Cleared {count} clients from registry")


@dataclass
class LLMRegistry:
    """Registry of LLM providers."""

    _providers: Dict[str, LLMProvider] = field(default_factory=dict)

    def register(self, llm_id: str, provider: LLMProvider) -> None:
        """Register an LLM provider."""
        self._providers[llm_id] = provider
        logger.debug(f"Registered LLM provider: {llm_id}")

    def unregister(self, llm_id: str) -> None:
        """Unregister an LLM provider."""
        if llm_id in self._providers:
            self._providers.pop(llm_id, None)
            logger.debug(f"Unregistered LLM provider: {llm_id}")

    def get(self, llm_id: str) -> LLMProvider:
        """Get LLM provider by ID."""
        if llm_id not in self._providers:
            logger.warning(f"LLM provider not found: {llm_id}")
            raise ResourceNotFoundError(f"LLM provider '{llm_id}' not found")
        return self._providers[llm_id]

    def has(self, llm_id: str) -> bool:
        """Check if LLM provider exists."""
        return llm_id in self._providers

    def list(self) -> List[str]:
        """List all LLM provider IDs."""
        return list(self._providers.keys())

    def clear(self) -> None:
        """Clear all LLM providers."""
        count = len(self._providers)
        self._providers.clear()
        logger.debug(f"Cleared {count} LLM providers from registry")
