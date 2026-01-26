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


class SessionState(str, Enum):
    """Interactive session states."""

    IDLE = "idle"
    GATHERING_INFO = "gathering_info"
    ROUTING_CONFIG = "routing_config"
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


class InteractiveSession:
    """
    Interactive session for task clarification and team creation.

    RFC-022/RFC-023: Conversational flow for task-solving.

    Example:
        session = await LLMTeam.start("Create marketing campaign")

        while session.state == SessionState.GATHERING_INFO:
            print(session.question)
            session.answer(input("> "))

        print(session.proposal)
        session.adjust("Add SEO specialist")

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

        self._state = SessionState.IDLE
        self._messages: List[Dict[str, str]] = []
        self._answers: Dict[str, Any] = {}
        self._current_question: Optional[Question] = None
        self._proposal: Optional[TeamProposal] = None
        self._team: Optional["LLMTeam"] = None
        self._blueprint: Optional["TeamBlueprint"] = None
        self._result: Optional["RunResult"] = None

    @property
    def state(self) -> SessionState:
        """Current session state."""
        return self._state

    @property
    def question(self) -> Optional[str]:
        """Current question text (if in GATHERING_INFO state)."""
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
                lines.append(f"  - {dp.get('decision_id', 'unknown')}")

        lines.append(
            f"\nEstimated cost: ${self._proposal.estimated_cost_min:.2f} - "
            f"${self._proposal.estimated_cost_max:.2f}"
        )

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
        from llmteam.builder import DynamicTeamBuilder

        self._state = SessionState.GATHERING_INFO

        # Add initial message
        self._messages.append({
            "role": "user",
            "content": self.task,
        })

        # Use DynamicTeamBuilder for analysis
        builder = DynamicTeamBuilder(verbose=False)

        # Check if task needs clarification
        questions = await builder.ask_clarifying_questions(self.task)

        if questions:
            # Need to gather more info
            self._current_question = Question(
                text=questions[0],
                field_name="clarification_0",
            )
            self._messages.append({
                "role": "assistant",
                "content": questions[0],
            })
        else:
            # Task is clear, move to proposing
            await self._generate_proposal()

    async def answer(self, response: str) -> None:
        """
        Answer the current question.

        Args:
            response: User's answer
        """
        if self._state != SessionState.GATHERING_INFO:
            raise RuntimeError(
                f"Cannot answer in state {self._state.value}. "
                f"Expected GATHERING_INFO."
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

        # Check if more questions needed
        from llmteam.builder import DynamicTeamBuilder

        builder = DynamicTeamBuilder(verbose=False)

        # Build context from task + answers
        context = f"{self.task}\n\nAdditional context:\n"
        for key, value in self._answers.items():
            context += f"- {value}\n"

        questions = await builder.ask_clarifying_questions(context)

        if questions:
            # More questions needed
            q_index = len(self._answers)
            self._current_question = Question(
                text=questions[0],
                field_name=f"clarification_{q_index}",
            )
            self._messages.append({
                "role": "assistant",
                "content": questions[0],
            })
        else:
            # Ready to propose
            self._current_question = None
            await self._generate_proposal()

    async def _generate_proposal(self) -> None:
        """Generate team proposal from task and answers."""
        from llmteam.builder import DynamicTeamBuilder

        self._state = SessionState.PROPOSING

        builder = DynamicTeamBuilder(verbose=False)

        # Build full task description with answers
        full_task = self.task
        if self._answers:
            full_task += "\n\nAdditional requirements:\n"
            for value in self._answers.values():
                full_task += f"- {value}\n"

        # Generate blueprint
        self._blueprint = await builder.analyze_task(full_task)

        # Convert to proposal
        self._proposal = TeamProposal(
            agents=[
                {
                    "role": agent.role,
                    "purpose": agent.purpose,
                    "model": agent.model,
                    "tools": agent.tools,
                }
                for agent in self._blueprint.agents
            ],
            flow=" → ".join(a.role for a in self._blueprint.agents),
            decision_points=[],  # TODO: Add when blueprint supports decision points
            estimated_cost_min=0.05,  # TODO: Calculate from blueprint
            estimated_cost_max=0.20,
        )

        # Add proposal message
        proposal_text = f"I propose the following team:\n\n{self.plan}\n\nWould you like to proceed?"
        self._messages.append({
            "role": "assistant",
            "content": proposal_text,
        })

        self._state = SessionState.PROPOSING

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

        if not self._blueprint:
            raise RuntimeError("No proposal to adjust")

        from llmteam.builder import DynamicTeamBuilder

        # Add adjustment to messages
        self._messages.append({
            "role": "user",
            "content": adjustment,
        })

        builder = DynamicTeamBuilder(verbose=False)

        # Refine blueprint
        self._blueprint = await builder.refine_blueprint(self._blueprint, adjustment)

        # Update proposal
        self._proposal = TeamProposal(
            agents=[
                {
                    "role": agent.role,
                    "purpose": agent.purpose,
                    "model": agent.model,
                    "tools": agent.tools,
                }
                for agent in self._blueprint.agents
            ],
            flow=" → ".join(a.role for a in self._blueprint.agents),
            decision_points=self._proposal.decision_points if self._proposal else [],
            estimated_cost_min=0.05,
            estimated_cost_max=0.25,
        )

        # Add updated proposal message
        proposal_text = f"Updated team:\n\n{self.plan}\n\nWould you like to proceed?"
        self._messages.append({
            "role": "assistant",
            "content": proposal_text,
        })

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

        if not self._blueprint:
            raise RuntimeError("No proposal to confirm")

        from llmteam.builder import DynamicTeamBuilder

        builder = DynamicTeamBuilder(verbose=False)

        # Build team from blueprint
        self._team = builder.build_team(self._blueprint)

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
            **self._answers,
        }

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
