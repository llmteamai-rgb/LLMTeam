import pytest
from llmteam.engine.handlers.condition_handler import ConditionHandler
from llmteam.runtime import StepContext, RuntimeContext

@pytest.fixture
def handler():
    return ConditionHandler()

@pytest.fixture
def mock_context():
    runtime = RuntimeContext(tenant_id="test", instance_id="test_inst", run_id="run_1", segment_id="seg_1")
    return StepContext(step_id="cond_step", runtime=runtime)

class TestConditionHandler:

    @pytest.mark.asyncio
    async def test_sanitize_valid(self, handler):
        """Test valid expressions."""
        assert handler._sanitize_expression("a == 1") == "a == 1"
        assert handler._sanitize_expression("field > 10") == "field > 10"

    @pytest.mark.asyncio
    async def test_sanitize_forbidden(self, handler):
        """Test forbidden patterns."""
        with pytest.raises(ValueError, match="forbidden pattern"):
            handler._sanitize_expression("eval('print(1)')")
        
        with pytest.raises(ValueError, match="forbidden pattern"):
            handler._sanitize_expression("__import__('os')")
            
        with pytest.raises(ValueError, match="Expression too long"):
            handler._sanitize_expression("a" * 1001)

    @pytest.mark.asyncio
    async def test_evaluate_comparison(self, handler, mock_context):
        """Test basic comparison."""
        data = {"score": 85, "role": "admin"}
        
        # True cases
        res = await handler(mock_context, {"expression": "score > 80"}, data)
        assert "true" in res
        
        res = await handler(mock_context, {"expression": "role == 'admin'"}, data)
        assert "true" in res

        # False cases
        res = await handler(mock_context, {"expression": "score > 90"}, data)
        assert "false" in res

    @pytest.mark.asyncio
    async def test_evaluate_logical(self, handler, mock_context):
        """Test logical operators."""
        data = {"a": True, "b": False}
        
        res = await handler(mock_context, {"expression": "a and not b"}, data)
        assert "true" in res
        
        res = await handler(mock_context, {"expression": "a and b"}, data)
        assert "false" in res
