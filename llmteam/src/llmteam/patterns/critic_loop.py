"""
Critic Loop Pattern.

This module provides the CriticLoop pattern for recursive improvement
through a Generator-Critic feedback loop.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Protocol, Union
import asyncio


class CriticVerdict(Enum):
    """Verdict from the critic agent."""

    APPROVED = "approved"
    NEEDS_REVISION = "needs_revision"
    REJECTED = "rejected"


@dataclass
class CriticLoopConfig:
    """
    Configuration for the critic loop pattern.

    Attributes:
        max_iterations: Maximum number of generate-critique cycles
        quality_threshold: Score threshold for approval (0.0-1.0)
        timeout_per_iteration: Timeout per iteration in seconds
        stop_on_rejection: Whether to stop immediately on rejection
        improvement_threshold: Minimum score improvement to continue
    """

    max_iterations: int = 5
    quality_threshold: float = 0.85
    timeout_per_iteration: float = 60.0
    stop_on_rejection: bool = True
    improvement_threshold: float = 0.05


@dataclass
class CriticFeedback:
    """
    Feedback from the critic agent.

    Attributes:
        verdict: The critic's verdict (approved, needs_revision, rejected)
        score: Quality score from 0.0 to 1.0
        feedback: Detailed feedback text
        suggestions: List of specific suggestions for improvement
    """

    verdict: CriticVerdict
    score: float
    feedback: str
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "verdict": self.verdict.value,
            "score": self.score,
            "feedback": self.feedback,
            "suggestions": self.suggestions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CriticFeedback":
        """Create from dictionary."""
        return cls(
            verdict=CriticVerdict(data["verdict"]),
            score=data["score"],
            feedback=data["feedback"],
            suggestions=data.get("suggestions", []),
        )


@dataclass
class IterationRecord:
    """Record of a single iteration in the critic loop."""

    iteration: int
    output: Any
    feedback: CriticFeedback
    duration_ms: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "iteration": self.iteration,
            "output": self.output,
            "feedback": self.feedback.to_dict(),
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CriticLoopResult:
    """
    Result of critic loop execution.

    Attributes:
        final_output: The final generated output
        iterations: Number of iterations performed
        final_score: Final quality score achieved
        history: List of iteration records
        converged: Whether the loop converged (met quality threshold)
        reason: Reason for termination
    """

    final_output: Any
    iterations: int
    final_score: float
    history: list[IterationRecord]
    converged: bool
    reason: str  # "quality_threshold", "max_iterations", "rejected", "no_improvement"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "final_output": self.final_output,
            "iterations": self.iterations,
            "final_score": self.final_score,
            "history": [r.to_dict() for r in self.history],
            "converged": self.converged,
            "reason": self.reason,
        }


class GeneratorProtocol(Protocol):
    """Protocol for generator agents/callables."""

    async def __call__(
        self,
        context: Any,
        input_data: dict[str, Any],
        history: list[IterationRecord],
    ) -> Any:
        """Generate output based on input and history."""
        ...


class CriticProtocol(Protocol):
    """Protocol for critic agents/callables."""

    async def __call__(
        self,
        context: Any,
        output: Any,
        history: list[IterationRecord],
    ) -> CriticFeedback:
        """Critique the output and provide feedback."""
        ...


class CriticLoop:
    """
    Recursive improvement through Generator-Critic pattern.

    The CriticLoop orchestrates a feedback loop where:
    1. A generator produces output based on input and previous feedback
    2. A critic evaluates the output and provides feedback
    3. The loop continues until quality threshold is met or limits are reached

    Example:
        async def writer(ctx, input_data, history):
            # Use LLM to generate content
            if history:
                # Incorporate feedback from previous iteration
                feedback = history[-1].feedback
                prompt = f"Revise based on: {feedback.feedback}"
            else:
                prompt = input_data["task"]
            return await llm.generate(prompt)

        async def reviewer(ctx, output, history):
            # Use LLM to review content
            review = await llm.review(output)
            return CriticFeedback(
                verdict=CriticVerdict.APPROVED if review.score > 0.9 else CriticVerdict.NEEDS_REVISION,
                score=review.score,
                feedback=review.feedback,
                suggestions=review.suggestions,
            )

        loop = CriticLoop(
            generator=writer,
            critic=reviewer,
            config=CriticLoopConfig(
                max_iterations=5,
                quality_threshold=0.85,
            ),
        )

        result = await loop.run(ctx, {"task": "Write article about AI"})
        print(f"Final output: {result.final_output}")
        print(f"Converged: {result.converged} after {result.iterations} iterations")
    """

    def __init__(
        self,
        generator: Union[GeneratorProtocol, Callable],
        critic: Union[CriticProtocol, Callable],
        config: Optional[CriticLoopConfig] = None,
    ) -> None:
        """
        Initialize CriticLoop.

        Args:
            generator: Agent or callable that generates output
            critic: Agent or callable that critiques output
            config: Loop configuration
        """
        self.generator = generator
        self.critic = critic
        self.config = config or CriticLoopConfig()

    async def run(
        self,
        context: Any,
        input_data: dict[str, Any],
    ) -> CriticLoopResult:
        """
        Execute the critic loop until convergence or max iterations.

        Args:
            context: Runtime context for agents
            input_data: Initial input data for the generator

        Returns:
            CriticLoopResult with final output, history, and convergence info
        """
        history: list[IterationRecord] = []
        current_input = input_data
        previous_score = 0.0

        for iteration in range(self.config.max_iterations):
            iteration_start = datetime.now()

            # Generate output
            try:
                if self.config.timeout_per_iteration > 0:
                    output = await asyncio.wait_for(
                        self._generate(context, current_input, history),
                        timeout=self.config.timeout_per_iteration,
                    )
                else:
                    output = await self._generate(context, current_input, history)
            except asyncio.TimeoutError:
                # Timeout during generation
                return CriticLoopResult(
                    final_output=history[-1].output if history else None,
                    iterations=iteration,
                    final_score=previous_score,
                    history=history,
                    converged=False,
                    reason="timeout",
                )

            # Critique output
            try:
                if self.config.timeout_per_iteration > 0:
                    feedback = await asyncio.wait_for(
                        self._critique(context, output, history),
                        timeout=self.config.timeout_per_iteration,
                    )
                else:
                    feedback = await self._critique(context, output, history)
            except asyncio.TimeoutError:
                # Timeout during critique - use last output
                return CriticLoopResult(
                    final_output=output,
                    iterations=iteration + 1,
                    final_score=previous_score,
                    history=history,
                    converged=False,
                    reason="timeout",
                )

            # Record iteration
            duration_ms = int((datetime.now() - iteration_start).total_seconds() * 1000)
            record = IterationRecord(
                iteration=iteration + 1,
                output=output,
                feedback=feedback,
                duration_ms=duration_ms,
            )
            history.append(record)

            # Check for approval
            if feedback.verdict == CriticVerdict.APPROVED:
                return CriticLoopResult(
                    final_output=output,
                    iterations=iteration + 1,
                    final_score=feedback.score,
                    history=history,
                    converged=True,
                    reason="quality_threshold",
                )

            # Check for rejection
            if feedback.verdict == CriticVerdict.REJECTED and self.config.stop_on_rejection:
                return CriticLoopResult(
                    final_output=output,
                    iterations=iteration + 1,
                    final_score=feedback.score,
                    history=history,
                    converged=False,
                    reason="rejected",
                )

            # Check quality threshold
            if feedback.score >= self.config.quality_threshold:
                return CriticLoopResult(
                    final_output=output,
                    iterations=iteration + 1,
                    final_score=feedback.score,
                    history=history,
                    converged=True,
                    reason="quality_threshold",
                )

            # Check improvement
            improvement = feedback.score - previous_score
            if iteration > 0 and improvement < self.config.improvement_threshold:
                return CriticLoopResult(
                    final_output=output,
                    iterations=iteration + 1,
                    final_score=feedback.score,
                    history=history,
                    converged=False,
                    reason="no_improvement",
                )

            previous_score = feedback.score

            # Prepare input for next iteration
            current_input = self._prepare_revision_input(input_data, output, feedback)

        # Max iterations reached
        last_record = history[-1] if history else None
        return CriticLoopResult(
            final_output=last_record.output if last_record else None,
            iterations=self.config.max_iterations,
            final_score=last_record.feedback.score if last_record else 0.0,
            history=history,
            converged=False,
            reason="max_iterations",
        )

    async def _generate(
        self,
        context: Any,
        input_data: dict[str, Any],
        history: list[IterationRecord],
    ) -> Any:
        """Generate output using the generator."""
        return await self.generator(context, input_data, history)

    async def _critique(
        self,
        context: Any,
        output: Any,
        history: list[IterationRecord],
    ) -> CriticFeedback:
        """Critique output using the critic."""
        return await self.critic(context, output, history)

    def _prepare_revision_input(
        self,
        original_input: dict[str, Any],
        output: Any,
        feedback: CriticFeedback,
    ) -> dict[str, Any]:
        """
        Prepare input for the next iteration.

        By default, adds the previous output and feedback to the input.
        Override this method for custom revision input preparation.
        """
        return {
            **original_input,
            "_previous_output": output,
            "_feedback": feedback.feedback,
            "_suggestions": feedback.suggestions,
            "_score": feedback.score,
        }
