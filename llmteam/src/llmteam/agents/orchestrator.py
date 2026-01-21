"""
TeamOrchestrator - Supervisor for agent teams.

NOT an agent. Separate entity that oversees agent execution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Flag, Enum, auto
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import json

from llmteam.agents.report import AgentReport

if TYPE_CHECKING:
    from llmteam.team import LLMTeam
    from llmteam.orchestration.models import GroupContext


class OrchestratorMode(Flag):
    """Operating modes for orchestrator."""

    # === Design-time ===
    CONFIGURATOR = auto()  # Help configure team via LLM (RFC-005)

    # === Run-time ===
    SUPERVISOR = auto()  # Observe, receive reports
    REPORTER = auto()  # Generate execution reports
    RECOVERY = auto()  # Decide on error recovery
    ROUTER = auto()  # Control agent selection

    # Presets
    PASSIVE = SUPERVISOR | REPORTER  # Default
    ACTIVE = SUPERVISOR | REPORTER | ROUTER  # Full routing control
    FULL = SUPERVISOR | REPORTER | ROUTER | RECOVERY  # Everything
    ASSISTED = CONFIGURATOR | PASSIVE  # Design-time + passive runtime (RFC-005)


class OrchestratorScope(Enum):
    """Scope of orchestrator authority."""

    TEAM = "team"  # Manages agents within team
    GROUP = "group"  # Also coordinates other teams


class RecoveryAction(Enum):
    """Actions for error recovery."""

    RETRY = "retry"  # Retry the failed agent
    SKIP = "skip"  # Skip the failed agent
    FALLBACK = "fallback"  # Use fallback agent
    ESCALATE = "escalate"  # Escalate to group/human
    ABORT = "abort"  # Stop execution


@dataclass
class OrchestratorConfig:
    """Configuration for TeamOrchestrator."""

    mode: OrchestratorMode = OrchestratorMode.PASSIVE
    model: str = "gpt-4o-mini"

    # Recovery settings
    auto_retry: bool = True
    max_retries: int = 2
    escalate_on_failure: bool = True

    # Report settings
    generate_report: bool = True
    report_format: str = "markdown"  # "markdown" | "json" | "text"
    include_agent_outputs: bool = True
    include_timing: bool = True


@dataclass
class RoutingDecision:
    """Decision from orchestrator about next agent."""

    next_agent: str
    reason: str
    confidence: float = 1.0
    fallback_agent: Optional[str] = None


@dataclass
class RecoveryDecision:
    """Decision from orchestrator about error recovery."""

    action: RecoveryAction
    reason: str
    retry_with_changes: Optional[Dict] = None
    fallback_agent: Optional[str] = None


class TeamOrchestrator:
    """
    Supervisor for agent team execution.

    NOT an agent. Does not inherit BaseAgent.
    Contains an internal LLM for decision-making.

    Responsibilities:
    - SUPERVISOR: Receive reports from agents
    - REPORTER: Generate execution summary/report
    - RECOVERY: Decide how to handle errors
    - ROUTER: Choose which agent runs next
    """

    def __init__(
        self,
        team: "LLMTeam",
        config: Optional[OrchestratorConfig] = None,
    ):
        """
        Initialize orchestrator.

        Args:
            team: Owner team
            config: Orchestrator configuration
        """
        self._team = team
        self._config = config or OrchestratorConfig()
        self._scope = OrchestratorScope.TEAM

        # Run state
        self._current_run_id: Optional[str] = None
        self._reports: List[AgentReport] = []
        self._started_at: Optional[datetime] = None

        # LLM provider (lazy init)
        self._llm = None

        # RFC-009: Group context
        self._group_context: Optional["GroupContext"] = None

    # Properties

    @property
    def mode(self) -> OrchestratorMode:
        """Current operating mode."""
        return self._config.mode

    @property
    def scope(self) -> OrchestratorScope:
        """Current scope."""
        return self._scope

    @property
    def is_router(self) -> bool:
        """Check if routing mode is enabled."""
        return OrchestratorMode.ROUTER in self._config.mode

    @property
    def reports(self) -> List[AgentReport]:
        """Get collected reports."""
        return self._reports.copy()

    # Lifecycle

    def start_run(self, run_id: str) -> None:
        """
        Start tracking a new run.

        Args:
            run_id: Run identifier
        """
        self._current_run_id = run_id
        self._reports = []
        self._started_at = datetime.utcnow()

    def end_run(self) -> None:
        """End current run tracking."""
        self._current_run_id = None

    # Reporting

    def receive_report(self, report: AgentReport) -> None:
        """
        Receive report from agent.

        Args:
            report: AgentReport from completed agent
        """
        if OrchestratorMode.SUPERVISOR not in self._config.mode:
            return

        self._reports.append(report)

    def generate_report(self) -> str:
        """
        Generate execution report.

        Returns:
            Report string in configured format
        """
        if OrchestratorMode.REPORTER not in self._config.mode:
            return ""

        if not self._reports:
            return "No agent reports collected."

        if self._config.report_format == "json":
            return self._generate_json_report()
        elif self._config.report_format == "text":
            return self._generate_text_report()
        else:  # markdown
            return self._generate_markdown_report()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get structured execution summary.

        Returns:
            Dictionary with execution summary
        """
        total_duration = sum(r.duration_ms for r in self._reports)
        total_tokens = sum(r.tokens_used for r in self._reports)
        success_count = sum(1 for r in self._reports if r.success)
        fail_count = len(self._reports) - success_count

        return {
            "run_id": self._current_run_id,
            "agents_executed": len(self._reports),
            "agents_succeeded": success_count,
            "agents_failed": fail_count,
            "total_duration_ms": total_duration,
            "total_tokens_used": total_tokens,
            "execution_order": [r.agent_id for r in self._reports],
            "errors": [
                {"agent_id": r.agent_id, "error": r.error}
                for r in self._reports
                if not r.success
            ],
        }

    # Routing (ROUTER mode)

    async def decide_next_agent(
        self,
        current_state: Dict[str, Any],
        available_agents: List[str],
    ) -> RoutingDecision:
        """
        Decide which agent should run next.

        Args:
            current_state: Current execution state
            available_agents: List of available agent IDs

        Returns:
            RoutingDecision with next agent
        """
        if OrchestratorMode.ROUTER not in self._config.mode:
            # Not in router mode - return first available
            return RoutingDecision(
                next_agent=available_agents[0] if available_agents else "",
                reason="Router mode not enabled, using sequential",
            )

        # Use LLM to decide
        llm = self._get_llm()
        if not llm:
            return RoutingDecision(
                next_agent=available_agents[0] if available_agents else "",
                reason="No LLM available, using first agent",
            )

        # Build prompt
        from llmteam.agents.prompts import build_routing_prompt

        prompt = build_routing_prompt(
            team=self._team,
            current_state=current_state,
            available_agents=available_agents,
            execution_history=self._reports,
        )

        # Call LLM
        response = await llm.complete(
            prompt=prompt,
            system_prompt="You are a team orchestrator. Choose the most appropriate agent for the current task.",
            temperature=0.1,
            max_tokens=200,
        )

        # Parse response
        return self._parse_routing_response(response, available_agents)

    # Recovery (RECOVERY mode)

    async def decide_recovery(
        self,
        error: Exception,
        failed_agent: str,
        context: Dict[str, Any],
    ) -> RecoveryDecision:
        """
        Decide how to recover from error.

        Args:
            error: The exception that occurred
            failed_agent: ID of agent that failed
            context: Execution context

        Returns:
            RecoveryDecision with action
        """
        if OrchestratorMode.RECOVERY not in self._config.mode:
            # Default behavior
            if self._config.auto_retry:
                return RecoveryDecision(
                    action=RecoveryAction.RETRY,
                    reason="Auto-retry enabled",
                )
            return RecoveryDecision(
                action=RecoveryAction.ABORT,
                reason="Recovery mode not enabled",
            )

        # Use LLM to decide
        llm = self._get_llm()
        if not llm:
            return RecoveryDecision(
                action=RecoveryAction.ABORT,
                reason="No LLM available for recovery decision",
            )

        # Build prompt
        from llmteam.agents.prompts import build_recovery_prompt

        prompt = build_recovery_prompt(
            error=error,
            failed_agent=failed_agent,
            context=context,
            team=self._team,
        )

        # Call LLM
        response = await llm.complete(
            prompt=prompt,
            system_prompt="You are a team orchestrator handling an error. Decide the best recovery action.",
            temperature=0.1,
            max_tokens=200,
        )

        # Parse response
        return self._parse_recovery_response(response)

    # Group scope

    def promote_to_group(self, group_id: str, teams: List["LLMTeam"]) -> None:
        """
        Promote orchestrator to GROUP scope.

        Args:
            group_id: Group identifier
            teams: Other teams in the group
        """
        self._scope = OrchestratorScope.GROUP
        # Store reference to other teams for coordination
        self._group_teams = teams

    # RFC-009: Group Context Integration

    def _set_group_context(self, context: "GroupContext") -> None:
        """
        INTERNAL: Set group context.

        Called from LLMTeam._join_group().

        Args:
            context: GroupContext from GroupOrchestrator
        """
        self._group_context = context
        self._scope = OrchestratorScope.GROUP

    def _clear_group_context(self) -> None:
        """
        INTERNAL: Clear group context.

        Called from LLMTeam._leave_group().
        """
        self._group_context = None
        self._scope = OrchestratorScope.TEAM

    @property
    def is_in_group(self) -> bool:
        """Is team in a group? (RFC-009)"""
        return self._group_context is not None

    @property
    def group_id(self) -> Optional[str]:
        """Group ID (if in group). (RFC-009)"""
        return self._group_context.group_id if self._group_context else None

    # Internal methods

    def _get_llm(self):
        """Get or create LLM provider."""
        if self._llm is not None:
            return self._llm

        try:
            from llmteam.providers import OpenAIProvider

            self._llm = OpenAIProvider(model=self._config.model)
            return self._llm
        except ImportError:
            return None

    def _generate_markdown_report(self) -> str:
        """Generate markdown format report."""
        lines = [
            f"# Execution Report",
            f"",
            f"**Run ID:** {self._current_run_id}",
            f"**Team:** {self._team.team_id}",
            f"**Agents Executed:** {len(self._reports)}",
            f"",
            f"## Agent Execution Log",
            f"",
        ]

        for i, report in enumerate(self._reports, 1):
            status = "[OK]" if report.success else "[FAIL]"
            lines.append(f"### {i}. {report.agent_id} ({report.agent_role})")
            lines.append(f"- **Type:** {report.agent_type}")
            lines.append(f"- **Status:** {status} {'Success' if report.success else 'Failed'}")
            lines.append(f"- **Duration:** {report.duration_ms}ms")

            if self._config.include_timing:
                lines.append(f"- **Started:** {report.started_at}")
                lines.append(f"- **Completed:** {report.completed_at}")

            if report.tokens_used > 0:
                lines.append(f"- **Tokens:** {report.tokens_used}")

            if self._config.include_agent_outputs:
                lines.append(f"- **Output:** {report.output_summary}")

            if not report.success and report.error:
                lines.append(f"- **Error:** {report.error}")

            lines.append("")

        # Summary
        summary = self.get_summary()
        lines.extend([
            f"## Summary",
            f"",
            f"- **Total Duration:** {summary['total_duration_ms']}ms",
            f"- **Total Tokens:** {summary['total_tokens_used']}",
            f"- **Success Rate:** {summary['agents_succeeded']}/{summary['agents_executed']}",
        ])

        return "\n".join(lines)

    def _generate_json_report(self) -> str:
        """Generate JSON format report."""
        data = {
            "run_id": self._current_run_id,
            "team_id": self._team.team_id,
            "reports": [r.to_dict() for r in self._reports],
            "summary": self.get_summary(),
        }
        return json.dumps(data, indent=2, default=str)

    def _generate_text_report(self) -> str:
        """Generate plain text report."""
        lines = [
            f"Execution Report - {self._team.team_id}",
            f"Run ID: {self._current_run_id}",
            f"=" * 50,
            "",
        ]

        for report in self._reports:
            lines.append(report.to_log_line())

        summary = self.get_summary()
        lines.extend([
            "",
            f"Total: {summary['agents_executed']} agents, "
            f"{summary['total_duration_ms']}ms, "
            f"{summary['total_tokens_used']} tokens",
        ])

        return "\n".join(lines)

    def _parse_routing_response(
        self,
        response: str,
        available_agents: List[str],
    ) -> RoutingDecision:
        """Parse LLM routing response."""
        # Try to parse as JSON
        try:
            data = json.loads(response)
            next_agent = data.get("next_agent", "")
            reason = data.get("reason", "LLM decision")
            confidence = data.get("confidence", 1.0)

            if next_agent in available_agents:
                return RoutingDecision(
                    next_agent=next_agent,
                    reason=reason,
                    confidence=confidence,
                )
        except json.JSONDecodeError:
            pass

        # Fallback: look for agent name in response
        for agent in available_agents:
            if agent.lower() in response.lower():
                return RoutingDecision(
                    next_agent=agent,
                    reason=f"Found '{agent}' in LLM response",
                )

        # Default to first agent
        return RoutingDecision(
            next_agent=available_agents[0] if available_agents else "",
            reason="Could not parse LLM response, using first agent",
        )

    def _parse_recovery_response(self, response: str) -> RecoveryDecision:
        """Parse LLM recovery response."""
        # Try to parse as JSON
        try:
            data = json.loads(response)
            action_str = data.get("action", "abort").upper()
            action = RecoveryAction[action_str]
            reason = data.get("reason", "LLM decision")

            return RecoveryDecision(
                action=action,
                reason=reason,
                retry_with_changes=data.get("changes"),
                fallback_agent=data.get("fallback"),
            )
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: look for action in response
        response_lower = response.lower()
        if "retry" in response_lower:
            return RecoveryDecision(action=RecoveryAction.RETRY, reason=response)
        if "skip" in response_lower:
            return RecoveryDecision(action=RecoveryAction.SKIP, reason=response)
        if "escalate" in response_lower:
            return RecoveryDecision(action=RecoveryAction.ESCALATE, reason=response)

        return RecoveryDecision(
            action=RecoveryAction.ABORT,
            reason="Could not parse LLM response",
        )

    def __repr__(self) -> str:
        return (
            f"<TeamOrchestrator team='{self._team.team_id}' "
            f"mode={self._config.mode.name} scope={self._scope.value}>"
        )
