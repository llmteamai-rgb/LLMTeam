"""
Ports Module.

This module provides the three-level port hierarchy for workflow steps:
- WORKFLOW: External communication (KorpOS, webhooks, APIs)
- AGENT: Internal agent-to-agent communication
- HUMAN: Human interaction (approvals, chat, escalation)
"""

from llmteam.ports.models import (
    # Enums
    PortLevel,
    PortDirection,
    PortDataType,
    # Models
    TypedPort,
    StepPorts,
    # Factories
    workflow_input,
    workflow_output,
    agent_input,
    agent_output,
    human_input,
    human_output,
    # Pre-defined port sets
    llm_agent_ports,
    human_task_ports,
    transform_ports,
    http_action_ports,
)

__all__ = [
    # Enums
    "PortLevel",
    "PortDirection",
    "PortDataType",
    # Models
    "TypedPort",
    "StepPorts",
    # Factories
    "workflow_input",
    "workflow_output",
    "agent_input",
    "agent_output",
    "human_input",
    "human_output",
    # Pre-defined port sets
    "llm_agent_ports",
    "human_task_ports",
    "transform_ports",
    "http_action_ports",
]
