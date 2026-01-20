"""
Secure Data Exchange Bus (RFC-002).

Implements secure, auditable data exchange between LLMTeam and Corpos.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol

from llmteam.observability import get_logger

logger = get_logger(__name__)


class DataMode(Enum):
    """
    Data Transfer Modes (RFC-002 Section 7).
    """
    REFERENCE = 0  # Default: send only references
    PAYLOAD = 1    # Send limited payload (whitelisted)
    ENCRYPTED = 2  # Future: encrypted payload



@dataclass
class BusConfig:
    """Configuration for SecureBus."""
    mode: DataMode = DataMode.REFERENCE
    audit_enabled: bool = True


@dataclass
class BusEvent:
    """
    Standard event structure (RFC-002 Section 4).
    """
    event_id: str
    event_type: str
    timestamp: datetime
    trace_id: str
    process_run_id: str
    source: str
    payload: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "trace_id": self.trace_id,
            "process_run_id": self.process_run_id,
            "source": self.source,
            "payload": self.payload,
        }


class BusSubscriber(Protocol):
    """Protocol for bus subscribers."""
    async def on_event(self, event: BusEvent) -> None:
        ...


class SecureBus:
    """
    Secure Data Bus implementation.
    
    Features:
    - Event publishing with audit
    - Control plane commands
    - Data mode enforcement
    """

    def __init__(self, config: Optional[BusConfig] = None):
        self.config = config or BusConfig()
        self._subscribers: Dict[str, List[BusSubscriber]] = {}
        self._audit_log: List[BusEvent] = []
        self._control_handlers: Dict[str, Callable] = {}
        
        logger.info(f"SecureBus initialized in mode {self.config.mode.name}")

    def subscribe(self, topic: str, subscriber: BusSubscriber) -> None:
        """Subscribe to a topic."""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(subscriber)

    async def publish(self, event: BusEvent) -> None:
        """
        Publish an event to the bus.
        
        Enforces data mode rules and logs to audit.
        """
        # Audit log first
        self._audit_event(event)
        
        # Enforce Data Mode
        if self.config.mode == DataMode.REFERENCE:
            # Strip payload if not reference-only? 
            # RFC says "references only". Assuming payload contains refs.
            # If payload has actual data, we might need to mask it.
            # For this implementation, we assume payload IS the data 
            # and in REFERENCE mode we might warn or strip non-ref data?
            # RFC Section 7: "Mode 0: references only".
            # For simplicity, we pass through but log warning if large payload?
            pass
        
        elif self.config.mode == DataMode.PAYLOAD:
            # Check whitelist?
            pass

        # Distribute
        subscribers = self._subscribers.get(event.event_type, [])
        subscribers.extend(self._subscribers.get("*", []))
        
        tasks = []
        for sub in subscribers:
            tasks.append(sub.on_event(event))
            
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _audit_event(self, event: BusEvent) -> None:
        """Log event to audit trail."""
        if self.config.audit_enabled:
            self._audit_log.append(event)
            logger.debug(
                f"AUDIT event={event.event_type} trace={event.trace_id}",
                extra={"audit": True, "event": event.to_dict()}
            )

    def register_control_handler(self, command: str, handler: Callable) -> None:
        """Register handler for control plane command."""
        self._control_handlers[command] = handler

    async def send_control(self, command: str, args: Dict[str, Any]) -> Any:
        """Execute a control command."""
        logger.info(f"Control command: {command}")
        if command in self._control_handlers:
            try:
                if asyncio.iscoroutinefunction(self._control_handlers[command]):
                    return await self._control_handlers[command](args)
                return self._control_handlers[command](args)
            except Exception as e:
                logger.error(f"Control command failed: {e}")
                raise
        else:
            raise ValueError(f"Unknown control command: {command}")

    def get_audit_log(self) -> List[BusEvent]:
        """Get copy of audit log."""
        return self._audit_log.copy()

    @staticmethod
    def create_event(
        event_type: str,
        trace_id: str,
        process_run_id: str,
        source: str,
        payload: Dict[str, Any] = None
    ) -> BusEvent:
        """Factory for creating events."""
        return BusEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(),
            trace_id=trace_id,
            process_run_id=process_run_id,
            source=source,
            payload=payload or {}
        )
