"""Tests for HTTPActionHandler."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from llmteam.canvas.handlers import HTTPActionHandler
from llmteam.runtime import StepContext


@pytest.fixture
def handler():
    """Create HTTP action handler."""
    return HTTPActionHandler()


@pytest.fixture
def mock_response():
    """Create mock HTTP response."""
    response = MagicMock()
    response.status = 200
    response.json = AsyncMock(return_value={"data": "response_data"})
    return response


@pytest.fixture
def mock_ctx(mock_response):
    """Create mock step context with HTTP client."""
    ctx = MagicMock(spec=StepContext)
    ctx.step_id = "http_step"
    ctx.run_id = "run_123"

    # Mock HTTP client
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.put = AsyncMock(return_value=mock_response)
    mock_client.patch = AsyncMock(return_value=mock_response)
    mock_client.delete = AsyncMock(return_value=mock_response)
    ctx.get_client = MagicMock(return_value=mock_client)

    return ctx


class TestHTTPActionHandler:
    """Tests for HTTPActionHandler."""

    async def test_get_request(self, handler, mock_ctx):
        """GET request is executed correctly."""
        config = {
            "client_ref": "api",
            "method": "GET",
            "path": "/users",
        }
        input_data = {}

        result = await handler(mock_ctx, config, input_data)

        assert result["status"] == 200
        assert result["response"]["data"] == "response_data"
        mock_ctx.get_client.assert_called_once_with("api")
        mock_ctx.get_client.return_value.get.assert_called_once()

    async def test_post_request(self, handler, mock_ctx):
        """POST request sends body."""
        config = {
            "client_ref": "api",
            "method": "POST",
            "path": "/users",
        }
        input_data = {"body": {"name": "John"}}

        await handler(mock_ctx, config, input_data)

        mock_client = mock_ctx.get_client.return_value
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args.kwargs
        assert call_kwargs["json"] == {"name": "John"}

    async def test_put_request(self, handler, mock_ctx):
        """PUT request works."""
        config = {
            "client_ref": "api",
            "method": "PUT",
            "path": "/users/1",
        }
        input_data = {"body": {"name": "Updated"}}

        await handler(mock_ctx, config, input_data)

        mock_client = mock_ctx.get_client.return_value
        mock_client.put.assert_called_once()

    async def test_patch_request(self, handler, mock_ctx):
        """PATCH request works."""
        config = {
            "client_ref": "api",
            "method": "PATCH",
            "path": "/users/1",
        }
        input_data = {"body": {"status": "active"}}

        await handler(mock_ctx, config, input_data)

        mock_client = mock_ctx.get_client.return_value
        mock_client.patch.assert_called_once()

    async def test_delete_request(self, handler, mock_ctx):
        """DELETE request works."""
        config = {
            "client_ref": "api",
            "method": "DELETE",
            "path": "/users/1",
        }
        input_data = {}

        await handler(mock_ctx, config, input_data)

        mock_client = mock_ctx.get_client.return_value
        mock_client.delete.assert_called_once()

    async def test_custom_headers(self, handler, mock_ctx):
        """Custom headers are passed."""
        config = {
            "client_ref": "api",
            "method": "GET",
            "path": "/data",
            "headers": {"Authorization": "Bearer token123"},
        }
        input_data = {}

        await handler(mock_ctx, config, input_data)

        mock_client = mock_ctx.get_client.return_value
        call_kwargs = mock_client.get.call_args.kwargs
        assert call_kwargs["headers"]["Authorization"] == "Bearer token123"

    async def test_custom_timeout(self, handler, mock_ctx):
        """Custom timeout is passed."""
        config = {
            "client_ref": "api",
            "method": "GET",
            "path": "/slow",
            "timeout": 60,
        }
        input_data = {}

        await handler(mock_ctx, config, input_data)

        mock_client = mock_ctx.get_client.return_value
        call_kwargs = mock_client.get.call_args.kwargs
        assert call_kwargs["timeout"] == 60

    async def test_default_method_is_post(self, handler, mock_ctx):
        """Default method is POST."""
        config = {
            "client_ref": "api",
            "path": "/submit",
        }
        input_data = {"data": "value"}

        await handler(mock_ctx, config, input_data)

        mock_client = mock_ctx.get_client.return_value
        mock_client.post.assert_called_once()

    async def test_default_client_ref(self, handler, mock_ctx):
        """Default client_ref is 'default'."""
        config = {"path": "/test"}
        input_data = {}

        await handler(mock_ctx, config, input_data)

        mock_ctx.get_client.assert_called_once_with("default")

    async def test_error_returns_500(self, handler, mock_ctx):
        """Client error returns 500 status."""
        mock_client = mock_ctx.get_client.return_value
        mock_client.get = AsyncMock(side_effect=Exception("Connection failed"))

        config = {
            "client_ref": "api",
            "method": "GET",
            "path": "/fail",
        }
        input_data = {}

        result = await handler(mock_ctx, config, input_data)

        assert result["status"] == 500
        assert "error" in result["response"]
        assert "Connection failed" in result["response"]["error"]

    async def test_unsupported_method_error(self, handler, mock_ctx):
        """Unsupported method returns error."""
        config = {
            "client_ref": "api",
            "method": "OPTIONS",  # Not supported
            "path": "/test",
        }
        input_data = {}

        result = await handler(mock_ctx, config, input_data)

        assert result["status"] == 500
        assert "error" in result["response"]

    async def test_input_data_used_as_body(self, handler, mock_ctx):
        """Input data is used as body when no 'body' key."""
        config = {
            "client_ref": "api",
            "method": "POST",
            "path": "/submit",
        }
        input_data = {"name": "Alice", "age": 30}

        await handler(mock_ctx, config, input_data)

        mock_client = mock_ctx.get_client.return_value
        call_kwargs = mock_client.post.call_args.kwargs
        assert call_kwargs["json"] == input_data

    async def test_case_insensitive_method(self, handler, mock_ctx):
        """Method is case-insensitive."""
        config = {
            "client_ref": "api",
            "method": "get",  # lowercase
            "path": "/test",
        }
        input_data = {}

        await handler(mock_ctx, config, input_data)

        mock_client = mock_ctx.get_client.return_value
        mock_client.get.assert_called_once()
