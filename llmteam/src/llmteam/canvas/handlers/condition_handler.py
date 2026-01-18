"""
Condition Handler.

Evaluates conditions and routes to true/false output ports.
Includes input sanitization to prevent unsafe eval patterns.
"""

from typing import Any
import operator
import re

from llmteam.runtime import StepContext
from llmteam.observability import get_logger


logger = get_logger(__name__)

# Security: Maximum expression length
MAX_EXPRESSION_LENGTH = 1000

# Security: Forbidden patterns that could be used for injection
FORBIDDEN_PATTERNS = [
    r'__\w+__',           # Dunder methods (__class__, __import__, etc.)
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
    r'\bfor\b.*\bin\b',   # for loops (except 'in' operator)
    r'\bwhile\b',         # while loops
    r'\bclass\b',         # class definition
    r'\bdef\b',           # function definition
]

# Compiled forbidden patterns for performance
_FORBIDDEN_REGEX = re.compile('|'.join(FORBIDDEN_PATTERNS), re.IGNORECASE)


class ConditionHandler:
    """
    Handler for condition step type.

    Evaluates expressions and outputs to 'true' or 'false' ports.
    """

    # Supported comparison operators
    OPERATORS = {
        "==": operator.eq,
        "!=": operator.ne,
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "in": lambda a, b: a in b,
        "not in": lambda a, b: a not in b,
        "contains": lambda a, b: b in a,
        "startswith": lambda a, b: str(a).startswith(str(b)),
        "endswith": lambda a, b: str(a).endswith(str(b)),
    }

    def __init__(self, max_expression_length: int = MAX_EXPRESSION_LENGTH) -> None:
        """
        Initialize handler.

        Args:
            max_expression_length: Maximum allowed expression length
        """
        self.max_expression_length = max_expression_length

    def _sanitize_expression(self, expression: str) -> str:
        """
        Sanitize expression to prevent injection attacks.

        Args:
            expression: Raw expression string

        Returns:
            Sanitized expression

        Raises:
            ValueError: If expression contains forbidden patterns
        """
        # Check length
        if len(expression) > self.max_expression_length:
            raise ValueError(
                f"Expression too long: {len(expression)} > {self.max_expression_length}"
            )

        # Check for forbidden patterns
        match = _FORBIDDEN_REGEX.search(expression)
        if match:
            raise ValueError(
                f"Expression contains forbidden pattern: '{match.group()}'"
            )

        return expression

    async def __call__(
        self,
        ctx: StepContext,
        config: dict[str, Any],
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Evaluate condition and route to output port.

        Args:
            ctx: Step context
            config: Step configuration:
                - expression: Condition expression (required)
                  Formats:
                  - "field == value"
                  - "field > 10"
                  - "field in ['a', 'b']"
                  - "field" (truthy check)
            input_data: Input data for evaluation

        Returns:
            Dict with either 'true' or 'false' key containing input data
        """
        expression = config.get("expression", "")

        logger.debug(f"Condition: evaluating '{expression}'")

        try:
            # Sanitize expression before evaluation
            expression = self._sanitize_expression(expression)
            result = self._evaluate(expression, input_data)
            logger.debug(f"Condition result: {result}")

            if result:
                return {"true": input_data}
            else:
                return {"false": input_data}

        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            # Default to false on error
            return {"false": input_data}

    def _evaluate(
        self,
        expression: str,
        data: dict[str, Any],
    ) -> bool:
        """
        Evaluate a condition expression.

        Args:
            expression: Condition expression
            data: Data to evaluate against

        Returns:
            Boolean result
        """
        expression = expression.strip()

        # Boolean literals
        if expression.lower() == "true":
            return True
        if expression.lower() == "false":
            return False

        # Logical operators (checked first - lower precedence)
        if " and " in expression.lower():
            parts = re.split(r'\s+and\s+', expression, flags=re.IGNORECASE)
            return all(self._evaluate(p.strip(), data) for p in parts)

        if " or " in expression.lower():
            parts = re.split(r'\s+or\s+', expression, flags=re.IGNORECASE)
            return any(self._evaluate(p.strip(), data) for p in parts)

        # Negation
        if expression.lower().startswith("not "):
            return not self._evaluate(expression[4:].strip(), data)

        # Try to parse comparison expression
        for op_str in sorted(self.OPERATORS.keys(), key=len, reverse=True):
            if f" {op_str} " in expression:
                parts = expression.split(f" {op_str} ", 1)
                if len(parts) == 2:
                    left = self._get_value(parts[0].strip(), data)
                    right = self._parse_literal(parts[1].strip(), data)
                    return self.OPERATORS[op_str](left, right)

        # Simple truthy check on field
        value = self._get_value(expression, data)
        return bool(value)

    def _get_value(
        self,
        path: str,
        data: dict[str, Any],
    ) -> Any:
        """
        Get value from data using dot notation.

        Args:
            path: Field path like "field" or "field.subfield"
            data: Data dict

        Returns:
            Value at path
        """
        path = path.strip()

        # Handle quoted strings as literals
        if (path.startswith('"') and path.endswith('"')) or \
           (path.startswith("'") and path.endswith("'")):
            return path[1:-1]

        # Handle numeric literals
        try:
            if "." in path:
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

    def _parse_literal(
        self,
        value: str,
        data: dict[str, Any],
    ) -> Any:
        """
        Parse a literal value or resolve field reference.

        Args:
            value: Literal or field reference
            data: Data for field resolution

        Returns:
            Parsed value
        """
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
