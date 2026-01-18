# Contributing to LLMTeam

Thank you for your interest in contributing to LLMTeam! This document provides guidelines and instructions for contributing.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/llmteamai-rgb/LLMTeam.git
   cd LLMTeam/llmteam
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Verify installation:
   ```bash
   # Bash
   PYTHONPATH=src python -c "import llmteam; print(f'v{llmteam.__version__}')"

   # PowerShell
   $env:PYTHONPATH="src"; python -c "import llmteam; print(f'v{llmteam.__version__}')"
   ```

## Running Tests

Tests require sequential or limited parallel execution to prevent memory exhaustion.

```bash
# Recommended: use test runner
python run_tests.py                    # Sequential (safest)
python run_tests.py --parallel 2       # Limited parallelism
python run_tests.py --module canvas    # Single module
python run_tests.py --coverage         # With coverage

# Single test
PYTHONPATH=src pytest tests/canvas/test_runner.py::TestSegmentRunner::test_simple_run -vv
```

**Avoid:** `pytest tests/ -n auto` — causes memory issues.

## Code Quality

Before submitting a PR, ensure your code passes all quality checks:

```bash
# Type checking
mypy src/llmteam/

# Formatting
black src/ tests/

# Linting
ruff check src/ tests/
```

Or use the Makefile:
```bash
make lint    # Ruff + mypy
make format  # Black
make test    # Run all tests
```

## Pull Request Process

1. **Create a feature branch** from `master`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards below.

3. **Add tests** for new functionality.

4. **Update documentation** if needed (CLAUDE.md, docstrings).

5. **Run all checks**:
   ```bash
   make lint
   python run_tests.py
   ```

6. **Submit a PR** with a clear description of changes.

## Coding Standards

### Python Style

- Follow PEP 8 with 100 character line length
- Use type hints for all function signatures
- Use `async`/`await` consistently for async code
- Use `asyncio.Lock()` for thread-safety

### Docstrings

Use Google-style docstrings:

```python
def function_name(arg1: str, arg2: int) -> dict[str, Any]:
    """
    Brief description of function.

    Longer description if needed.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When something is wrong
    """
```

### Module Structure

When creating a new module:

1. Create module directory with `__init__.py` containing exports
2. Add imports to `llmteam/__init__.py` (or use lazy import for optional deps)
3. Create tests in `tests/{module}/test_{module}.py`
4. Add module to `TEST_MODULES` in `run_tests.py`

### Handler Protocol

Step handlers should follow this pattern:

```python
from typing import Any
from llmteam.runtime import StepContext

class MyHandler:
    """
    Handler description.

    Step Type: "my_handler"

    Config:
        option1: Description
        option2: Description

    Input:
        field1: Description

    Output:
        result: Description
    """

    STEP_TYPE = "my_handler"
    DISPLAY_NAME = "My Handler"
    DESCRIPTION = "What this handler does"
    CATEGORY = "category_name"

    async def __call__(
        self,
        ctx: StepContext,
        config: dict[str, Any],
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        # Implementation
        return {"result": "value"}
```

## Security

- Never commit secrets or API keys
- Use `SealedData` for sensitive information
- Follow the security principles:
  - Horizontal Isolation: Agents never see each other's contexts
  - Vertical Visibility: Orchestrators see only their child agents
  - Tenant Isolation: Complete data separation between tenants

## License

By contributing to LLMTeam, you agree that your contributions will be licensed under the Apache 2.0 License.

## Questions?

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- Tag issues appropriately (bug, enhancement, documentation)
