import pytest
import asyncio
from typing import Dict, Any
from llmteam.transport.bus import SecureBus, DataMode, BusEvent

class TestSecureBus:
    
    @pytest.mark.asyncio
    async def test_publish_subscribe(self):
        """Test basic publish/subscribe."""
        bus = SecureBus()
        received_events = []

        class Subscriber:
            async def on_event(self, event: BusEvent):
                received_events.append(event)
        
        bus.subscribe("test.event", Subscriber())
        
        event = SecureBus.create_event(
            event_type="test.event",
            trace_id="t1",
            process_run_id="pr1",
            source="test",
            payload={"foo": "bar"}
        )
        
        await bus.publish(event)
        
        assert len(received_events) == 1
        assert received_events[0].event_id == event.event_id
        assert received_events[0].payload["foo"] == "bar"

    @pytest.mark.asyncio
    async def test_audit_log(self):
        """Test that events are audited."""
        bus = SecureBus()
        event = SecureBus.create_event("audit.test", "t1", "p1", "src")
        await bus.publish(event)
        
        log = bus.get_audit_log()
        assert len(log) == 1
        assert log[0].event_type == "audit.test"

    @pytest.mark.asyncio
    async def test_control_plane(self):
        """Test control plane commands."""
        bus = SecureBus()
        
        received_args = {}
        async def handle_pause(args: Dict[str, Any]):
            received_args.update(args)
            return "paused"
            
        bus.register_control_handler("run.pause", handle_pause)
        
        result = await bus.send_control("run.pause", {"run_id": "123"})
        
        assert result == "paused"
        assert received_args["run_id"] == "123"

    @pytest.mark.asyncio
    async def test_unknown_control_command(self):
        """Test error on unknown command."""
        bus = SecureBus()
        with pytest.raises(ValueError):
            await bus.send_control("unknown", {})
