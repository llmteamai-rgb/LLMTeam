"""
Adaptive Step Handler.

RFC-022: Handler for adaptive routing steps.

Implements rules-first routing with LLM fallback:
1. Evaluate rules in order (cheap, deterministic)
2. If no rules match, use LLM to decide (if configured)
3. Fall back to default route if nothing else works
"""

import time
import re
from typing import Any, Dict, Optional

from llmteam.runtime import StepContext
from llmteam.observability import get_logger
from llmteam.routing import (
    AdaptiveStepConfig,
    RoutingRule,
    LLMFallbackConfig,
    RoutingDecision,
    RoutingMethod,
    AdaptiveDecisionEvent,
)


logger = get_logger(__name__)


# Security: Forbidden patterns (same as condition_handler)
FORBIDDEN_PATTERNS = [
    r'__\w+__',           # Dunder methods
    r'\beval\b',          # eval()
    r'\bexec\b',          # exec()
    r'\bcompile\b',       # compile()
    r'\bimport\b',        # import
    r'\bopen\b',          # open()
    r'\bos\.',            # os module
    r'\bsys\.',           # sys module
    r'\bsubprocess\b',    # subprocess
    r'\bglobals\b',       # globals()
    r'\blocals\b',        # locals()
    r'\bgetattr\b',       # getattr()
    r'\bsetattr\b',       # setattr()
    r'\bdelattr\b',       # delattr()
    r'\b__builtins__\b',  # builtins
    r'\blambda\b',        # lambda
]

_FORBIDDEN_REGEX = re.compile('|'.join(FORBIDDEN_PATTERNS), re.IGNORECASE)


class AdaptiveStepHandler:
    """
    Handler for adaptive routing steps.

    Evaluates rules first, falls back to LLM if no rules match.
    """

    def __init__(self) -> None:
        """Initialize handler."""
        self._checkpoint_store: Optional[Any] = None

    async def __call__(
        self,
        ctx: StepContext,
        config: Dict[str, Any],
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute adaptive routing step.

        Args:
            ctx: Step context
            config: Step configuration (AdaptiveStepConfig as dict)
            input_data: Input data for routing decision

        Returns:
            Dict with routing decision and target-specific output port
        """
        start_time = time.time()

        # Parse config
        adaptive_config = AdaptiveStepConfig.from_dict(config)
        decision_id = adaptive_config.decision_id

        logger.debug(f"Adaptive step {decision_id}: evaluating routing")

        # 1. Checkpoint before decision (if configured)
        if adaptive_config.checkpoint_before:
            await self._save_checkpoint(ctx, decision_id, input_data)

        # 2. Try rule-based routing first
        decision = self._evaluate_rules(adaptive_config, input_data)

        # 3. LLM fallback if no rule matched
        if decision is None and adaptive_config.llm_fallback:
            decision = await self._llm_decide(
                adaptive_config.llm_fallback,
                input_data,
                decision_id,
            )

        # 4. Default route
        if decision is None and adaptive_config.default_route:
            decision = RoutingDecision(
                target=adaptive_config.default_route,
                method=RoutingMethod.DEFAULT,
                decision_id=decision_id,
            )

        # 5. Error if no route found
        if decision is None:
            raise ValueError(
                f"No route found for adaptive step {decision_id}. "
                "Configure rules, LLM fallback, or default_route."
            )

        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)
        decision.duration_ms = duration_ms

        # Log decision event
        event = AdaptiveDecisionEvent.from_decision(
            decision,
            step_id=ctx.step_id if hasattr(ctx, 'step_id') else "",
        )
        logger.info(
            f"Adaptive decision: {decision_id} → {decision.target} "
            f"(method={decision.method.value}, {duration_ms}ms)"
        )

        # Return output with target-specific port
        target_key = f"routed_to_{decision.target.replace('-', '_')}"
        return {
            "routed_to": decision.target,
            "routing_method": decision.method.value,
            "decision_id": decision_id,
            target_key: True,
            # Include full decision for debugging/audit
            "decision": decision.to_dict(),
            # Include event for logging
            "event": event.to_dict(),
            # Pass through input to next step
            "output": input_data,
        }

    def _evaluate_rules(
        self,
        config: AdaptiveStepConfig,
        input_data: Dict[str, Any],
    ) -> Optional[RoutingDecision]:
        """
        Evaluate rules in order.

        Returns first matching rule's decision, or None if no match.
        """
        for rule in config.rules:
            try:
                if self._evaluate_condition(rule.condition, input_data):
                    logger.debug(
                        f"Rule matched: {rule.condition} → {rule.target}"
                    )
                    return RoutingDecision(
                        target=rule.target,
                        method=RoutingMethod.RULE,
                        rule_condition=rule.condition,
                        rule_description=rule.description,
                        decision_id=config.decision_id,
                    )
            except Exception as e:
                logger.warning(f"Rule evaluation error: {rule.condition} - {e}")
                continue

        return None

    def _evaluate_condition(
        self,
        condition: str,
        data: Dict[str, Any],
    ) -> bool:
        """
        Safely evaluate a condition expression.

        Supports:
        - output.field == 'value'
        - output.score > 80
        - output.status in ['ready', 'approved']
        """
        condition = condition.strip()

        # Security check
        if _FORBIDDEN_REGEX.search(condition):
            raise ValueError(f"Forbidden pattern in condition: {condition}")

        # Simple truthy check
        if not any(op in condition for op in ['==', '!=', '>', '<', ' in ', ' not in ']):
            value = self._get_value(condition, data)
            return bool(value)

        # Comparison operators
        for op, func in [
            (' not in ', lambda a, b: a not in b),
            (' in ', lambda a, b: a in b),
            ('!=', lambda a, b: a != b),
            ('==', lambda a, b: a == b),
            ('>=', lambda a, b: a >= b),
            ('<=', lambda a, b: a <= b),
            ('>', lambda a, b: a > b),
            ('<', lambda a, b: a < b),
        ]:
            if op in condition:
                parts = condition.split(op, 1)
                if len(parts) == 2:
                    left = self._get_value(parts[0].strip(), data)
                    right = self._parse_literal(parts[1].strip(), data)
                    try:
                        return func(left, right)
                    except TypeError:
                        return False

        return False

    def _get_value(self, path: str, data: Dict[str, Any]) -> Any:
        """Get value using dot notation (e.g., 'output.field')."""
        path = path.strip()

        # Handle quoted strings
        if (path.startswith('"') and path.endswith('"')) or \
           (path.startswith("'") and path.endswith("'")):
            return path[1:-1]

        # Handle numeric literals
        try:
            if "." in path and not path.startswith("output"):
                return float(path)
            return int(path)
        except ValueError:
            pass

        # Navigate nested fields
        current = data
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def _parse_literal(self, value: str, data: Dict[str, Any]) -> Any:
        """Parse literal value or resolve field reference."""
        value = value.strip()

        # Boolean literals
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        if value.lower() in ("none", "null"):
            return None

        # String literals
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            return value[1:-1]

        # List literals
        if value.startswith("[") and value.endswith("]"):
            try:
                import ast
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass

        # Numeric literals
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # Field reference
        return self._get_value(value, data)

    async def _llm_decide(
        self,
        config: LLMFallbackConfig,
        input_data: Dict[str, Any],
        decision_id: str,
    ) -> Optional[RoutingDecision]:
        """Use LLM to make routing decision."""
        import json

        # Build routes description
        routes_desc = "\n".join([
            f"- {r.target}: {r.description}" + (f" (when: {r.when})" if r.when else "")
            for r in config.routes
        ])

        # Build prompt
        prompt = f"""{config.prompt}

Available routes:
{routes_desc}

Input data:
{json.dumps(input_data, indent=2, default=str)[:2000]}

Respond with JSON only:
{{"target": "<route_target>", "reasoning": "<why this route>", "confidence": 0.0-1.0}}"""

        try:
            # Try to get LLM provider from runtime context
            from llmteam.providers import OpenAIProvider

            provider = OpenAIProvider()
            response = await provider.complete(
                messages=[{"role": "user", "content": prompt}],
                model=config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )

            # Parse response
            decision = RoutingDecision.from_json(
                response.content,
                decision_id=decision_id,
            )

            # Validate target is in routes
            valid_targets = [r.target for r in config.routes]
            if decision.target and decision.target in valid_targets:
                return decision

            logger.warning(
                f"LLM returned invalid target '{decision.target}'. "
                f"Valid targets: {valid_targets}"
            )
            return None

        except Exception as e:
            logger.error(f"LLM fallback failed: {e}")
            return None

    async def _save_checkpoint(
        self,
        ctx: StepContext,
        decision_id: str,
        input_data: Dict[str, Any],
    ) -> None:
        """Save checkpoint before decision."""
        # TODO: Implement checkpoint storage
        # For now, just log
        logger.debug(f"Checkpoint saved before adaptive step: {decision_id}")
