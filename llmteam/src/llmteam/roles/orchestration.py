"""
Orchestration strategies for llmteam.

Provides decision-making logic for pipeline orchestrators.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypedDict

from llmteam.observability import get_logger


logger = get_logger(__name__)


class OrchestratorRole(Enum):
    """
    Roles for orchestrator.

    Attributes:
        ORCHESTRATION: Decision-making and routing
        PROCESS_MINING: Process discovery and analytics
    """

    ORCHESTRATION = "orchestration"
    PROCESS_MINING = "process_mining"


class OrchestrationDecisionDict(TypedDict):
    """Dictionary representation of OrchestrationDecision."""
    decision_type: str
    target_agents: List[str]
    reason: str
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class OrchestrationDecision:
    """
    Decision made by orchestrator.

    Attributes:
        decision_type: Type of decision (route, retry, escalate, skip, parallel, end)
        target_agents: List of agent names to execute
        reason: Reasoning behind the decision
        confidence: Confidence level (0.0-1.0)
        metadata: Additional metadata
    """

    decision_type: str
    target_agents: List[str]
    reason: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> OrchestrationDecisionDict:
        """Convert to dictionary."""
        return {
            "decision_type": self.decision_type,
            "target_agents": self.target_agents,
            "reason": self.reason,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class OrchestrationContext:
    """
    Context for orchestration decisions.

    Attributes:
        current_step: Current step in the pipeline
        available_agents: List of available agent names
        agent_states: State of each agent
        execution_history: History of execution steps
        global_state: Global state data
        step_duration: Duration of current step
        retry_count: Number of retries for current step
        error_rate: Error rate (0.0-1.0)
    """

    current_step: str
    available_agents: List[str]
    agent_states: Dict[str, dict]
    execution_history: List[dict]
    global_state: dict

    # Metrics for decision-making
    step_duration: timedelta = field(default_factory=timedelta)
    retry_count: int = 0
    error_rate: float = 0.0


class OrchestrationStrategy:
    """
    Base class for orchestration strategies.

    Subclasses must implement the decide() method.
    """

    async def decide(self, context: OrchestrationContext) -> OrchestrationDecision:
        """
        Make an orchestration decision based on context.

        Args:
            context: Current orchestration context

        Returns:
            OrchestrationDecision
        """
        raise NotImplementedError


class RuleBasedStrategy(OrchestrationStrategy):
    """
    Rule-based orchestration strategy.

    Applies a sequence of rules to determine the next action.

    Example:
        strategy = RuleBasedStrategy()

        # Add rule for high error rate
        strategy.add_rule(lambda ctx:
            OrchestrationDecision("retry", [ctx.current_step], "high_error_rate", 0.9)
            if ctx.error_rate > 0.5 else None
        )

        # Add rule for timeout
        strategy.add_rule(lambda ctx:
            OrchestrationDecision("escalate", [], "timeout", 0.8)
            if ctx.step_duration.total_seconds() > 600 else None
        )

        # Use in orchestrator
        decision = await strategy.decide(context)
    """

    def __init__(self):
        """Initialize rule-based strategy."""
        self.rules: List[Callable[[OrchestrationContext], Optional[OrchestrationDecision]]] = []

    def add_rule(self, rule: Callable[[OrchestrationContext], Optional[OrchestrationDecision]]) -> "RuleBasedStrategy":
        """
        Add a rule to the strategy.

        Args:
            rule: Function that takes OrchestrationContext and returns OrchestrationDecision or None

        Returns:
            Self for chaining
        """
        self.rules.append(rule)
        return self

    async def decide(self, context: OrchestrationContext) -> OrchestrationDecision:
        """
        Apply rules sequentially until one returns a decision.

        Args:
            context: Orchestration context

        Returns:
            OrchestrationDecision from first matching rule, or default decision
        """
        for rule in self.rules:
            decision = rule(context)
            if decision:
                return decision

        # Default: route to next agent in sequence
        if context.available_agents:
            return OrchestrationDecision(
                decision_type="route",
                target_agents=[context.available_agents[0]],
                reason="default_sequence",
                confidence=1.0,
            )
        else:
            return OrchestrationDecision(
                decision_type="end",
                target_agents=[],
                reason="no_more_agents",
                confidence=1.0,
            )


class LLMBasedStrategy(OrchestrationStrategy):
    """
    LLM-based orchestration strategy.

    Uses an LLM to make intelligent routing decisions.

    Note: This is an abstract implementation. Users must provide their own LLM.

    Example:
        # User provides their own LLM implementation
        class MyLLM:
            async def generate(self, prompt: str) -> str:
                # Call your LLM API here
                pass

        strategy = LLMBasedStrategy(llm=MyLLM())
        decision = await strategy.decide(context)
    """

    def __init__(self, llm: Any, prompt_template: Optional[str] = None):
        """
        Initialize LLM-based strategy.

        Args:
            llm: LLM instance with generate() method
            prompt_template: Custom prompt template (uses default if None)
        """
        self.llm = llm
        self.prompt_template = prompt_template or self._default_prompt()

    async def decide(self, context: OrchestrationContext) -> OrchestrationDecision:
        """
        Use LLM to make orchestration decision.

        Args:
            context: Orchestration context

        Returns:
            OrchestrationDecision parsed from LLM response
        """
        prompt = self.prompt_template.format(
            current_step=context.current_step,
            agents=context.available_agents,
            states=context.agent_states,
            history=context.execution_history[-5:],  # Last 5 steps
            error_rate=context.error_rate,
        )

        try:
            response = await self.llm.generate(prompt)
            return self._parse_response(response)
        except Exception as e:
            logger.error(f"LLM orchestration failed: {str(e)}")
            # Fallback
            return OrchestrationDecision(
                decision_type="error",
                target_agents=[],
                reason=f"llm_error: {str(e)}",
                confidence=0.0,
            )

    def _default_prompt(self) -> str:
        """
        Get default prompt template.

        Returns:
            Default prompt template string
        """
        return """
You are a pipeline orchestrator. Based on the current state, decide the next action.

Current step: {current_step}
Available agents: {agents}
Agent states: {states}
Recent history: {history}
Error rate: {error_rate}

Respond with JSON: {{"decision": "route|retry|escalate|skip|end", "targets": [...], "reason": "..."}}
""".strip()

    def _parse_response(self, response: str) -> OrchestrationDecision:
        """
        Parse LLM response into OrchestrationDecision.

        Args:
            response: LLM response string

        Returns:
            OrchestrationDecision
        """
        import json

        try:
            data = json.loads(response)
            return OrchestrationDecision(
                decision_type=data.get("decision", "route"),
                target_agents=data.get("targets", []),
                reason=data.get("reason", "llm_decision"),
                confidence=data.get("confidence", 0.8),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM response: {str(e)}")
            # Fallback to default decision
            return OrchestrationDecision(
                decision_type="route",
                target_agents=[],
                reason="llm_parse_error",
                confidence=0.5,
            )
