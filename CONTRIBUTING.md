# Contributing to llmteam

Thank you for your interest in contributing to llmteam! We welcome contributions from the community to help make this the best enterprise AI workflow runtime.

## Development Setup

1. **Clone and Install**
   ```bash
   git clone https://github.com/llmteamai-rgb/LLMTeam.git
   cd LLMTeam/llmteam
   pip install -e ".[dev]"
   ```

2. **Verify Installation**
   ```bash
   python -c "import llmteam; print(llmteam.__version__)"
   ```

## Development Guidelines

### Code Style
- We use `black` for formatting and `ruff` for linting.
- All code must include type hints (`mypy` strict mode).
- Docstrings should follow Google style.

```bash
make format   # Run black
make lint     # Run ruff and mypy
```

### Testing
- Run tests sequentially to avoid memory issues.
- Write unit tests for all new logic.
- Use `pytest` fixtures for resources.

```bash
# Run all tests
python run_tests.py

# Run specific module
python run_tests.py --module canvas
```

## Pull Request Process

1. Create a new branch: `git checkout -b feature/my-feature`
2. Implement your changes.
3. Add tests and documentation.
4. Ensure all checks pass: `make lint && make test`
5. Submit a PR describing your changes.

## Release Process (Maintainers)

1. Update version in `pyproject.toml` and `src/llmteam/__init__.py`.
2. Update `CHANGELOG.md`.
3. Create a release tag `vX.Y.Z`.
4. Build and publish to PyPI.
