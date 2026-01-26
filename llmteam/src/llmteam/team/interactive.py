"""
Interactive Session for task-solving with Q&A.

RFC-022/RFC-023: Provides conversational interface for task clarification
before team creation and execution.

States:
    IDLE → GATHERING_INFO → ROUTING_CONFIG → PROPOSING → READY → EXECUTING → COMPLETED
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from llmteam.team.team import LLMTeam
    from llmteam.team.result import RunResult
    from llmteam.builder import TeamBlueprint
    from llmteam.configuration import ConfiguratorOutput, TaskAnalysis


class SessionState(str, Enum):
    """Interactive session states."""

    IDLE = "idle"
    GATHERING_INFO = "gathering_info"
    ROUTING_CONFIG = "routing_config"  # RFC-023: Routing preferences
    PROPOSING = "proposing"
    READY = "ready"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Question:
    """Question from the orchestrator."""

    text: str
    """Question text."""

    options: Optional[List[str]] = None
    """Optional multiple choice options."""

    required: bool = True
    """Is answer required?"""

    field_name: str = ""
    """Internal field name for the answer."""


@dataclass
class TeamProposal:
    """Proposed team configuration."""

    agents: List[Dict[str, Any]] = field(default_factory=list)
    """Proposed agents."""

    flow: str = ""
    """Proposed flow."""

    decision_points: List[Dict[str, Any]] = field(default_factory=list)
    """Decision points for adaptive routing."""

    estimated_cost_min: float = 0.0
    """Minimum estimated cost."""

    estimated_cost_max: float = 0.0
    """Maximum estimated cost."""

    adaptive_cost: float = 0.0
    """Cost of routing decisions."""


class InteractiveSession:
    """
    Interactive session for task clarification and team creation.

    RFC-022/RFC-023: Conversational flow for task-solving with routing config.

    Example:
        session = await LLMTeam.start("Create marketing campaign")

        while session.state == SessionState.GATHERING_INFO:
            print(session.question)
            await session.answer(input("> "))

        # If routing config needed (quality >= 60 with decision points)
        if session.state == SessionState.ROUTING_CONFIG:
            print(session.question)
            await session.answer("auto")

        print(session.proposal)
        await session.adjust("Add SEO specialist")

        result = await session.execute()
    """

    def __init__(
        self,
        task: str,
        quality: int = 50,
        routing_mode: str = "hybrid",
        session_id: Optional[str] = None,
    ):
        """
        Initialize session.

        Args:
            task: Initial task description
            quality: Quality level 0-100
            routing_mode: Routing mode for execution
            session_id: Optional session identifier
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.task = task
        self.quality = quality
        self.routing_mode = routing_mode

        # RFC-023: Routing preference
        self.routing_preference: str = "auto"  # auto | rules_only | ask_each_time | full_llm

        self._state = SessionState.IDLE
        self._messages: List[Dict[str, str]] = []
        self._answers: Dict[str, Any] = {}
        self._current_question: Optional[Question] = None
        self._proposal: Optional[TeamProposal] = None
        self._team: Optional["LLMTeam"] = None
        self._blueprint: Optional["TeamBlueprint"] = None
        self._config_output: Optional["ConfiguratorOutput"] = None
        self._analysis: Optional["TaskAnalysis"] = None
        self._result: Optional["RunResult"] = None

    @property
    def state(self) -> SessionState:
        """Current session state."""
        return self._state

    @property
    def question(self) -> Optional[str]:
        """Current question text (if in GATHERING_INFO or ROUTING_CONFIG state)."""
        if self._current_question:
            return self._current_question.text
        return None

    @property
    def question_options(self) -> Optional[List[str]]:
        """Current question options (if multiple choice)."""
        if self._current_question:
            return self._current_question.options
        return None

    @property
    def proposal(self) -> Optional[TeamProposal]:
        """Team proposal (if in PROPOSING or later state)."""
        return self._proposal

    @property
    def team(self) -> Optional["LLMTeam"]:
        """Built team (if in READY or later state)."""
        return self._team

    @property
    def plan(self) -> Optional[str]:
        """Human-readable plan description."""
        if not self._proposal:
            return None

        lines = ["Proposed Team:"]
        for agent in self._proposal.agents:
            lines.append(f"  - {agent.get('role', 'agent')}: {agent.get('purpose', '')}")

        lines.append(f"\nFlow: {self._proposal.flow}")

        if self._proposal.decision_points:
            lines.append("\nDecision Points:")
            for dp in self._proposal.decision_points:
                dp_id = dp.get('decision_id', 'unknown')
                has_llm = dp.get('llm_fallback') is not None
                mode = "rules + LLM" if has_llm else "rules only"
                lines.append(f"  - {dp_id}: {mode}")

        cost_str = f"${self._proposal.estimated_cost_min:.2f} - ${self._proposal.estimated_cost_max:.2f}"
        if self._proposal.adaptive_cost > 0:
            cost_str += f" (includes ~${self._proposal.adaptive_cost:.2f} for routing)"
        lines.append(f"\nEstimated cost: {cost_str}")

        return "\n".join(lines)

    @property
    def result(self) -> Optional["RunResult"]:
        """Execution result (if COMPLETED)."""
        return self._result

    @property
    def messages(self) -> List[Dict[str, str]]:
        """Chat message history."""
        return self._messages.copy()

    async def _start(self) -> None:
        """Internal: Start the session by analyzing the task."""
        from llmteam.configuration import Configurator

        self._state = SessionState.GATHERING_INFO

        # Add initial message
        self._messages.append({
            "role": "user",
            "content": self.task,
        })

        # Use Configurator for analysis
        configurator = Configurator(routing_mode=self.routing_mode, quality=self.quality)

        try:
            # Analyze task to understand requirements
            self._analysis = await configurator._analyze_task(
                task=self.task,
                quality=self.quality,
                constraints={},
                context=None,
            )

            # Check if task needs clarification based on complexity
            if self._analysis.complexity == "complex" and len(self._analysis.sub_tasks) > 3:
                # Complex task - ask clarifying questions
                questions = self._generate_clarifying_questions()
                if questions:
                    self._current_question = Question(
                        text=questions[0],
                        field_name="clarification_0",
                    )
                    self._messages.append({
                        "role": "assistant",
                        "content": questions[0],
                    })
                    return

            # Task is clear enough, check for routing config
            if self.quality >= 60 and self._analysis.decision_points:
                self._state = SessionState.ROUTING_CONFIG
                self._current_question = self._generate_routing_question()
                self._messages.append({
                    "role": "assistant",
                    "content": self._current_question.text,
                })
            else:
                # Simple task or low quality - go to proposing
                await self._generate_proposal()

        except Exception as e:
            # Fallback to simple proposal
            self._messages.append({
                "role": "assistant",
                "content": f"Analysis note: {e}. Generating simple proposal...",
            })
            await self._generate_proposal()

    def _generate_clarifying_questions(self) -> List[str]:
        """Generate clarifying questions based on analysis."""
        questions = []

        if self._analysis:
            # Ask about unclear aspects
            if not self._analysis.domain:
                questions.append("What domain or industry is this for?")

            if len(self._analysis.sub_tasks) > 5:
                questions.append(
                    f"This seems complex with {len(self._analysis.sub_tasks)} subtasks. "
                    "Could you prioritize the most important ones?"
                )

        return questions

    def _generate_routing_question(self) -> Question:
        """Generate question about routing preferences (RFC-023)."""
        dp_descriptions = []
        for dp in self._analysis.decision_points:
            dp_descriptions.append(f"  - After {dp.after_step}: {dp.description}")

        text = f"""Based on my analysis, I've identified these decision points:

{chr(10).join(dp_descriptions)}

How would you like to handle routing at these points?

Options:
1. **auto** (recommended) - Use rules where possible, LLM only when rules can't decide
2. **rules_only** - Only use deterministic rules, fail if can't decide
3. **ask_each_time** - Pause and ask user at each decision point
4. **full_llm** - Always use LLM for routing (more flexible but more expensive)

For quality={self.quality}, I recommend: {'rules_only' if self.quality >= 80 else 'auto'}"""

        return Question(
            text=text,
            options=["auto", "rules_only", "ask_each_time", "full_llm"],
            field_name="routing_preference",
        )

    async def answer(self, response: str) -> None:
        """
        Answer the current question.

        Args:
            response: User's answer
        """
        if self._state == SessionState.ROUTING_CONFIG:
            # Handle routing preference answer
            await self._handle_routing_answer(response)
            return

        if self._state != SessionState.GATHERING_INFO:
            raise RuntimeError(
                f"Cannot answer in state {self._state.value}. "
                f"Expected GATHERING_INFO or ROUTING_CONFIG."
            )

        if not self._current_question:
            raise RuntimeError("No question to answer")

        # Store answer
        self._answers[self._current_question.field_name] = response

        # Add to messages
        self._messages.append({
            "role": "user",
            "content": response,
        })

        # Check if we need routing config (for quality >= 60 with decision points)
        if self.quality >= 60 and self._analysis and self._analysis.decision_points:
            self._state = SessionState.ROUTING_CONFIG
            self._current_question = self._generate_routing_question()
            self._messages.append({
                "role": "assistant",
                "content": self._current_question.text,
            })
        else:
            # Ready to propose
            self._current_question = None
            await self._generate_proposal()

    async def _handle_routing_answer(self, response: str) -> None:
        """Handle routing preference answer."""
        # Parse response
        response_lower = response.lower().strip()
        if "auto" in response_lower:
            self.routing_preference = "auto"
        elif "rules_only" in response_lower or "rules only" in response_lower:
            self.routing_preference = "rules_only"
        elif "ask" in response_lower:
            self.routing_preference = "ask_each_time"
        elif "full" in response_lower or "llm" in response_lower:
            self.routing_preference = "full_llm"
        else:
            self.routing_preference = "auto"

        # Store in answers
        self._answers["routing_preference"] = self.routing_preference

        # Add to messages
        self._messages.append({
            "role": "user",
            "content": response,
        })

        self._messages.append({
            "role": "assistant",
            "content": f"Got it. Using '{self.routing_preference}' routing mode.",
        })

        # Move to proposing
        self._current_question = None
        await self._generate_proposal()

    async def _generate_proposal(self) -> None:
        """Generate team proposal from task and answers."""
        from llmteam.configuration import Configurator

        self._state = SessionState.PROPOSING

        # Build full task description with answers
        full_task = self.task
        if self._answers:
            full_task += "\n\nAdditional requirements:\n"
            for key, value in self._answers.items():
                if key != "routing_preference":
                    full_task += f"- {value}\n"

        # Use Configurator for full configuration
        configurator = Configurator(routing_mode=self.routing_mode, quality=self.quality)

        try:
            self._config_output = await configurator.configure(
                task=full_task,
                quality=self.quality,
                constraints={},
                routing_mode=self.routing_mode,
            )

            # Apply routing preference to decision points
            decision_points = self._apply_routing_preference(
                self._config_output.decision_points
            )

            # Convert to proposal
            self._proposal = TeamProposal(
                agents=self._config_output.agents,
                flow=self._config_output.flow,
                decision_points=[dp.to_dict() for dp in decision_points],
                estimated_cost_min=self._config_output.estimated_cost.min_cost if self._config_output.estimated_cost else 0.05,
                estimated_cost_max=self._config_output.estimated_cost.max_cost if self._config_output.estimated_cost else 0.20,
                adaptive_cost=self._config_output.estimated_cost.adaptive_cost if self._config_output.estimated_cost else 0.0,
            )

        except Exception as e:
            # Fallback to simple proposal
            self._proposal = TeamProposal(
                agents=[
                    {
                        "role": "worker",
                        "type": "llm",
                        "purpose": "Process the task",
                        "prompt": f"Complete this task: {self.task}\n\nInput: {{input}}",
                        "model": "gpt-4o-mini",
                    }
                ],
                flow="worker",
                decision_points=[],
                estimated_cost_min=0.01,
                estimated_cost_max=0.05,
            )

        # Add proposal message
        proposal_text = f"I propose the following team:\n\n{self.plan}\n\nWould you like to proceed or make adjustments?"
        self._messages.append({
            "role": "assistant",
            "content": proposal_text,
        })

    def _apply_routing_preference(self, decision_points: List) -> List:
        """Apply routing preference to decision points."""
        from llmteam.configuration.models import DecisionPointConfig

        updated = []
        for dp in decision_points:
            if self.routing_preference == "rules_only":
                # Remove LLM fallback
                dp.llm_fallback = None
            elif self.routing_preference == "full_llm":
                # Clear rules, keep only LLM
                dp.rules = []
            # "auto" and "ask_each_time" keep configuration as-is
            updated.append(dp)

        return updated

    async def adjust(self, adjustment: str) -> None:
        """
        Request adjustment to the proposal.

        Args:
            adjustment: Description of desired changes
        """
        if self._state not in (SessionState.PROPOSING, SessionState.READY):
            raise RuntimeError(
                f"Cannot adjust in state {self._state.value}. "
                f"Expected PROPOSING or READY."
            )

        if not self._proposal:
            raise RuntimeError("No proposal to adjust")

        # Add adjustment to messages
        self._messages.append({
            "role": "user",
            "content": adjustment,
        })

        # Check if adjustment is about routing
        adjustment_lower = adjustment.lower()
        if "routing" in adjustment_lower or "decision" in adjustment_lower or "llm fallback" in adjustment_lower:
            await self._adjust_routing(adjustment)
        else:
            await self._adjust_agents(adjustment)

        # Add updated proposal message
        proposal_text = f"Updated team:\n\n{self.plan}\n\nWould you like to proceed?"
        self._messages.append({
            "role": "assistant",
            "content": proposal_text,
        })

    async def _adjust_routing(self, adjustment: str) -> None:
        """Adjust routing configuration based on user request."""
        adjustment_lower = adjustment.lower()

        # Parse common adjustments
        if "remove llm" in adjustment_lower or "no llm" in adjustment_lower:
            # Remove LLM fallback from all decision points
            for dp in self._proposal.decision_points:
                dp["llm_fallback"] = None
            self._proposal.adaptive_cost = 0.0

        elif "rules only" in adjustment_lower:
            self.routing_preference = "rules_only"
            for dp in self._proposal.decision_points:
                dp["llm_fallback"] = None
            self._proposal.adaptive_cost = 0.0

        # Update state
        self._state = SessionState.PROPOSING

    async def _adjust_agents(self, adjustment: str) -> None:
        """Adjust agent configuration based on user request."""
        from llmteam.configuration import Configurator

        # Use LLM to understand and apply adjustment
        configurator = Configurator(routing_mode=self.routing_mode, quality=self.quality)

        # Re-configure with adjustment as additional context
        full_task = f"{self.task}\n\nAdjustment: {adjustment}"
        if self._answers:
            full_task += "\n\nPrevious context:\n"
            for key, value in self._answers.items():
                if key != "routing_preference":
                    full_task += f"- {value}\n"

        try:
            self._config_output = await configurator.configure(
                task=full_task,
                quality=self.quality,
                constraints={"adjustment": adjustment},
                routing_mode=self.routing_mode,
            )

            # Update proposal
            self._proposal = TeamProposal(
                agents=self._config_output.agents,
                flow=self._config_output.flow,
                decision_points=[dp.to_dict() for dp in self._config_output.decision_points],
                estimated_cost_min=self._config_output.estimated_cost.min_cost if self._config_output.estimated_cost else 0.05,
                estimated_cost_max=self._config_output.estimated_cost.max_cost if self._config_output.estimated_cost else 0.25,
                adaptive_cost=self._config_output.estimated_cost.adaptive_cost if self._config_output.estimated_cost else 0.0,
            )

        except Exception:
            # Keep existing proposal if adjustment fails
            pass

        self._state = SessionState.PROPOSING

    async def confirm(self) -> None:
        """
        Confirm the proposal and build the team.

        After confirmation, team is ready to execute.
        """
        if self._state != SessionState.PROPOSING:
            raise RuntimeError(
                f"Cannot confirm in state {self._state.value}. "
                f"Expected PROPOSING."
            )

        if not self._proposal:
            raise RuntimeError("No proposal to confirm")

        from llmteam.team import LLMTeam

        # Build team from proposal
        self._team = LLMTeam(
            team_id=f"session_{self.session_id[:8]}",
            orchestration=True,  # Enable ROUTER mode
        )

        for agent in self._proposal.agents:
            self._team.add_agent(agent)

        if self._proposal.flow:
            self._team.set_flow(self._proposal.flow)

        self._state = SessionState.READY

        self._messages.append({
            "role": "assistant",
            "content": "Team confirmed and ready to execute.",
        })

    async def execute(self) -> "RunResult":
        """
        Execute the confirmed team.

        Returns:
            RunResult with output and metadata

        Raises:
            RuntimeError: If team not confirmed
        """
        if self._state == SessionState.PROPOSING:
            # Auto-confirm if executing from PROPOSING state
            await self.confirm()

        if self._state != SessionState.READY:
            raise RuntimeError(
                f"Cannot execute in state {self._state.value}. "
                f"Expected READY."
            )

        if not self._team:
            raise RuntimeError("No team to execute")

        self._state = SessionState.EXECUTING

        self._messages.append({
            "role": "assistant",
            "content": "Executing team...",
        })

        # Build input from task and answers
        input_data = {
            "task": self.task,
        }
        for key, value in self._answers.items():
            if key != "routing_preference":
                input_data[key] = value

        try:
            self._result = await self._team.run(input_data)
            self._state = SessionState.COMPLETED

            self._messages.append({
                "role": "assistant",
                "content": f"Execution completed. Success: {self._result.success}",
            })

            return self._result

        except Exception as e:
            self._state = SessionState.FAILED

            self._messages.append({
                "role": "assistant",
                "content": f"Execution failed: {e}",
            })

            raise
