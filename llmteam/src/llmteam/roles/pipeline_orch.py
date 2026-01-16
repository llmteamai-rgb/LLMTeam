"""
Pipeline orchestrator for llmteam.

Combines orchestration strategy with process mining for intelligent pipeline management.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from llmteam.roles.orchestration import (
    OrchestrationStrategy,
    OrchestrationContext,
    RuleBasedStrategy,
)
from llmteam.roles.process_mining import (
    ProcessMiningEngine,
    ProcessEvent,
    ProcessMetrics,
    generate_uuid,
)


class PipelineOrchestrator:
    """
    Pipeline orchestrator with dual roles:
    1. Orchestration: Decision-making and routing
    2. Process Mining: Workflow analysis and optimization

    Example:
        # Create orchestrator with rule-based strategy
        orchestrator = PipelineOrchestrator(
            pipeline_id="loan_approval",
            strategy=RuleBasedStrategy(),
            enable_process_mining=True,
        )

        # Register agents
        orchestrator.register_agent("validator", validator_agent)
        orchestrator.register_agent("analyzer", analyzer_agent)
        orchestrator.register_agent("decider", decider_agent)

        # Execute workflow
        result = await orchestrator.orchestrate("run_123", input_data)

        # Get insights
        metrics = orchestrator.get_process_metrics()
        print(f"Avg duration: {metrics.avg_duration}")
        print(f"Bottlenecks: {metrics.bottleneck_activities}")

        # Export for analysis
        xes = orchestrator.export_process_model()
    """

    def __init__(
        self,
        pipeline_id: str,
        strategy: Optional[OrchestrationStrategy] = None,
        enable_process_mining: bool = True,
    ):
        """
        Initialize pipeline orchestrator.

        Args:
            pipeline_id: Unique identifier for this pipeline
            strategy: Orchestration strategy (defaults to RuleBasedStrategy)
            enable_process_mining: Whether to enable process mining
        """
        self.pipeline_id = pipeline_id
        self.strategy = strategy or RuleBasedStrategy()

        # Process Mining
        self.process_mining = ProcessMiningEngine() if enable_process_mining else None

        # State
        self._agents: Dict[str, Any] = {}
        self._execution_history: List[dict] = []

    def register_agent(self, name: str, agent: Any) -> None:
        """
        Register an agent with this pipeline.

        Args:
            name: Agent name
            agent: Agent instance
        """
        self._agents[name] = agent

    def unregister_agent(self, name: str) -> None:
        """
        Unregister an agent.

        Args:
            name: Agent name to remove
        """
        if name in self._agents:
            del self._agents[name]

    async def orchestrate(self, run_id: str, input_data: dict) -> dict:
        """
        Execute pipeline orchestration.

        Args:
            run_id: Unique run identifier
            input_data: Input data for the pipeline

        Returns:
            Final pipeline state
        """
        current_step = "start"
        state = input_data.copy()

        while current_step != "end":
            # Build orchestration context
            context = OrchestrationContext(
                current_step=current_step,
                available_agents=list(self._agents.keys()),
                agent_states={
                    name: self._get_agent_state(agent)
                    for name, agent in self._agents.items()
                },
                execution_history=self._execution_history,
                global_state=state,
            )

            # Make decision
            decision = await self.strategy.decide(context)

            # Record decision for process mining
            if self.process_mining:
                self.process_mining.record_event(ProcessEvent(
                    event_id=generate_uuid(),
                    timestamp=datetime.now(),
                    activity=f"decision:{decision.decision_type}",
                    resource="orchestrator",
                    case_id=run_id,
                    lifecycle="complete",
                    attributes={
                        "reason": decision.reason,
                        "confidence": decision.confidence,
                    },
                ))

            # Execute decision
            if decision.decision_type == "route":
                agents_executed = False
                for agent_name in decision.target_agents:
                    agent = self._agents.get(agent_name)
                    if not agent:
                        continue

                    agents_executed = True

                    # Record start event
                    if self.process_mining:
                        start_time = datetime.now()
                        self.process_mining.record_event(ProcessEvent(
                            event_id=generate_uuid(),
                            timestamp=start_time,
                            activity=agent_name,
                            resource=agent_name,
                            case_id=run_id,
                            lifecycle="start",
                        ))

                    # Execute agent
                    result = await agent.process(state)
                    state.update(result)

                    # Record complete event
                    if self.process_mining:
                        end_time = datetime.now()
                        duration_ms = int((end_time - start_time).total_seconds() * 1000)
                        self.process_mining.record_event(ProcessEvent(
                            event_id=generate_uuid(),
                            timestamp=end_time,
                            activity=agent_name,
                            resource=agent_name,
                            case_id=run_id,
                            lifecycle="complete",
                            duration_ms=duration_ms,
                        ))

                    current_step = agent_name

                # If no agents were executed (all missing), end the pipeline
                if not agents_executed:
                    current_step = "end"

            elif decision.decision_type == "end" or not decision.target_agents:
                current_step = "end"

            elif decision.decision_type == "retry":
                # Retry current step (simplified)
                if decision.target_agents:
                    current_step = decision.target_agents[0]
                else:
                    current_step = "end"

            else:
                # Unknown decision type, end execution
                current_step = "end"

            # Save to history
            self._execution_history.append({
                "step": current_step,
                "decision": {
                    "type": decision.decision_type,
                    "targets": decision.target_agents,
                    "reason": decision.reason,
                },
                "timestamp": datetime.now().isoformat(),
            })

        return state

    def get_process_metrics(self) -> Optional[ProcessMetrics]:
        """
        Get process mining metrics.

        Returns:
            ProcessMetrics if process mining is enabled, None otherwise
        """
        if self.process_mining:
            return self.process_mining.calculate_metrics()
        return None

    def export_process_model(self) -> Optional[str]:
        """
        Export process model to XES format.

        Returns:
            XES XML string if process mining is enabled, None otherwise
        """
        if self.process_mining:
            return self.process_mining.export_xes()
        return None

    def get_execution_history(self) -> List[dict]:
        """
        Get execution history.

        Returns:
            List of execution steps
        """
        return self._execution_history.copy()

    def _get_agent_state(self, agent: Any) -> dict:
        """
        Get agent state.

        Args:
            agent: Agent instance

        Returns:
            Dictionary with agent state
        """
        if hasattr(agent, 'get_state'):
            return agent.get_state()
        return {"status": "available"}
