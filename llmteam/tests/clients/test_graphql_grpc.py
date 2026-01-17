"""Tests for GraphQL and gRPC clients."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestGraphQLClientConfig:
    """Tests for GraphQLClientConfig."""

    def test_default_config(self):
        """Create config with defaults."""
        from llmteam.clients import GraphQLClientConfig

        config = GraphQLClientConfig(endpoint="https://api.example.com/graphql")

        assert config.endpoint == "https://api.example.com/graphql"
        assert config.timeout == 30.0
        assert config.retry_enabled is True
        assert config.max_retries == 3
        assert config.cache_enabled is False

    def test_custom_config(self):
        """Create config with custom values."""
        from llmteam.clients import GraphQLClientConfig

        config = GraphQLClientConfig(
            endpoint="https://api.example.com/graphql",
            headers={"Authorization": "Bearer token"},
            timeout=60.0,
            retry_enabled=False,
            cache_enabled=True,
            cache_ttl_seconds=120,
        )

        assert config.headers == {"Authorization": "Bearer token"}
        assert config.timeout == 60.0
        assert config.retry_enabled is False
        assert config.cache_enabled is True
        assert config.cache_ttl_seconds == 120


class TestGraphQLResponse:
    """Tests for GraphQLResponse."""

    def test_response_with_data(self):
        """Response with data only."""
        from llmteam.clients import GraphQLResponse

        response = GraphQLResponse(data={"user": {"name": "John"}})

        assert response.data == {"user": {"name": "John"}}
        assert response.has_errors is False

    def test_response_with_errors(self):
        """Response with errors."""
        from llmteam.clients import GraphQLResponse, GraphQLError

        response = GraphQLResponse(
            data=None,
            errors=[{"message": "Not found", "path": ["user"]}],
        )

        assert response.has_errors is True

        with pytest.raises(GraphQLError) as exc_info:
            response.raise_for_errors()

        assert "Not found" in str(exc_info.value)


class TestGraphQLClient:
    """Tests for GraphQLClient."""

    def test_initialization_simple(self):
        """Initialize with endpoint only."""
        from llmteam.clients import GraphQLClient

        client = GraphQLClient(endpoint="https://api.example.com/graphql")

        assert client.config.endpoint == "https://api.example.com/graphql"

    def test_initialization_with_headers(self):
        """Initialize with headers."""
        from llmteam.clients import GraphQLClient

        client = GraphQLClient(
            endpoint="https://api.example.com/graphql",
            headers={"Authorization": "Bearer token"},
        )

        assert client.config.headers == {"Authorization": "Bearer token"}

    def test_initialization_with_config(self):
        """Initialize with full config."""
        from llmteam.clients import GraphQLClient, GraphQLClientConfig

        config = GraphQLClientConfig(
            endpoint="https://api.example.com/graphql",
            timeout=60.0,
        )
        client = GraphQLClient(endpoint="", config=config)

        assert client.config.timeout == 60.0

    def test_clear_cache(self):
        """Clear cache returns count."""
        from llmteam.clients import GraphQLClient

        client = GraphQLClient(endpoint="https://api.example.com/graphql")
        client._cache["key1"] = MagicMock()
        client._cache["key2"] = MagicMock()

        count = client.clear_cache()

        assert count == 2
        assert len(client._cache) == 0


class TestGRPCClientConfig:
    """Tests for GRPCClientConfig."""

    def test_default_config(self):
        """Create config with defaults."""
        from llmteam.clients import GRPCClientConfig

        config = GRPCClientConfig(target="localhost:50051")

        assert config.target == "localhost:50051"
        assert config.secure is False
        assert config.default_timeout == 30.0
        assert config.retry_enabled is True
        assert config.max_retries == 3

    def test_secure_config(self):
        """Create config for secure connection."""
        from llmteam.clients import GRPCClientConfig

        config = GRPCClientConfig(
            target="api.example.com:443",
            secure=True,
            root_certificates=b"...",
        )

        assert config.secure is True
        assert config.root_certificates == b"..."


class TestGRPCClient:
    """Tests for GRPCClient."""

    def test_initialization_simple(self):
        """Initialize with target only."""
        from llmteam.clients import GRPCClient

        client = GRPCClient(target="localhost:50051")

        assert client.config.target == "localhost:50051"
        assert client.config.secure is False

    def test_initialization_secure(self):
        """Initialize with secure flag."""
        from llmteam.clients import GRPCClient

        client = GRPCClient(target="localhost:50051", secure=True)

        assert client.config.secure is True

    def test_initialization_with_config(self):
        """Initialize with full config."""
        from llmteam.clients import GRPCClient, GRPCClientConfig

        config = GRPCClientConfig(
            target="localhost:50051",
            default_timeout=60.0,
        )
        client = GRPCClient(config=config)

        assert client.config.default_timeout == 60.0

    def test_channel_state_not_connected(self):
        """Channel state when not connected."""
        from llmteam.clients import GRPCClient, ChannelState

        client = GRPCClient(target="localhost:50051")

        assert client.channel_state == ChannelState.IDLE


class TestCallOptions:
    """Tests for CallOptions."""

    def test_default_options(self):
        """Default call options."""
        from llmteam.clients import CallOptions

        options = CallOptions()

        assert options.timeout is None
        assert options.metadata is None
        assert options.wait_for_ready is False

    def test_custom_options(self):
        """Custom call options."""
        from llmteam.clients import CallOptions

        options = CallOptions(
            timeout=10.0,
            metadata={"key": "value"},
            wait_for_ready=True,
        )

        assert options.timeout == 10.0
        assert options.metadata == {"key": "value"}
        assert options.wait_for_ready is True


class TestGRPCErrors:
    """Tests for gRPC errors."""

    def test_grpc_error(self):
        """GRPCError with code and details."""
        from llmteam.clients import GRPCError

        error = GRPCError(
            message="Call failed",
            code="UNAVAILABLE",
            details="Server not responding",
        )

        assert str(error) == "Call failed"
        assert error.code == "UNAVAILABLE"
        assert error.details == "Server not responding"

    def test_grpc_client_error(self):
        """GRPCClientError for client issues."""
        from llmteam.clients import GRPCClientError

        error = GRPCClientError("Not connected")

        assert str(error) == "Not connected"


class TestClientsExports:
    """Test that clients module exports are correct."""

    def test_graphql_exports(self):
        """GraphQL classes are exported."""
        from llmteam.clients import (
            GraphQLClient,
            GraphQLClientConfig,
            GraphQLResponse,
            GraphQLError,
            GraphQLClientError,
            GraphQLSubscription,
        )

        assert GraphQLClient is not None
        assert GraphQLClientConfig is not None
        assert GraphQLResponse is not None
        assert GraphQLError is not None

    def test_grpc_exports(self):
        """gRPC classes are exported."""
        from llmteam.clients import (
            GRPCClient,
            GRPCClientConfig,
            GRPCError,
            GRPCClientError,
            CallOptions,
            ChannelState,
        )

        assert GRPCClient is not None
        assert GRPCClientConfig is not None
        assert GRPCError is not None
        assert CallOptions is not None
        assert ChannelState is not None
