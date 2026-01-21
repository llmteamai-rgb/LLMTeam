import pytest
from unittest.mock import AsyncMock, MagicMock
from llmteam.engine.handlers.rag_handler import RAGHandler
from llmteam.runtime import StepContext, RuntimeContext
from llmteam.context.provider import ContextProvider, RetrievalResult, ContextResponse


@pytest.fixture
def mock_context():
    runtime = RuntimeContext(tenant_id="test", instance_id="test_inst", run_id="run_123", segment_id="seg_456")
    return StepContext(
        step_id="rag_step",
        runtime=runtime
    )


class TestRAGHandler:
    
    @pytest.mark.asyncio
    async def test_native_mode_default(self, mock_context):
        """Test default native mode logic."""
        handler = RAGHandler()
        config = {
            "mode": "native",
            "query_template": "Query: {query}",
            "include_context": True
        }
        input_data = {"query": "Hello"}
        
        # Native provider returns mock data in our implementation
        result = await handler(mock_context, config, input_data)
        
        assert "output" in result
        assert result["output"]["query"] == "Hello"
        # Check native mock content
        assert "Native result" in result["output"]["context"][0]["content"]

    @pytest.mark.asyncio
    async def test_proxy_mode_configuration(self, mock_context):
        """Test proxy mode init and retrieval."""
        handler = RAGHandler()
        config = {
            "mode": "proxy",
            "proxy_endpoint": "http://api.korpos.local",
            "proxy_api_key": "secret",
            "context_ref": "ignored_in_factory_but_needed_logic", # handler uses params
            "query_template": "{query}",
            "include_context": True
        }
        input_data = {"query": "ProxyCheck"}
        
        # The default implementation mocks the response based on endpoint
        result = await handler(mock_context, config, input_data)
        
        assert "output" in result
        content = result["output"]["context"][0]["content"]
        assert "Proxy result" in content
        assert "http://api.korpos.local" in content

    @pytest.mark.asyncio
    async def test_proxy_requires_endpoint(self, mock_context):
        """Test validation for proxy mode."""
        handler = RAGHandler()
        config = {
            "mode": "proxy",
            # Missing proxy_endpoint
            "query_template": "{query}"
        }
        input_data = {"query": "Fail"}
        
        with pytest.raises(ValueError, match="proxy_endpoint is required"):
            await handler(mock_context, config, input_data)
