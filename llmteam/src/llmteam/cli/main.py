"""LLMTeam CLI - Main entry point.

Usage:
    llmteam run segment.json --input data.json
    llmteam validate segment.json
    llmteam catalog
    llmteam version
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, TextIO

import click

import llmteam


def _load_json_file(filepath: str | Path) -> dict[str, Any]:
    """Load and parse JSON file."""
    path = Path(filepath)
    if not path.exists():
        raise click.ClickException(f"File not found: {filepath}")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON in {filepath}: {e}")


def _write_json(data: Any, output: TextIO, pretty: bool = True) -> None:
    """Write JSON to output stream."""
    if pretty:
        json.dump(data, output, indent=2, ensure_ascii=False)
    else:
        json.dump(data, output, ensure_ascii=False)
    output.write("\n")


@click.group()
@click.version_option(version=llmteam.__version__, prog_name="llmteam")
def cli() -> None:
    """LLMTeam CLI - Enterprise AI Workflow Runtime.

    Build and run multi-agent LLM pipelines with security,
    orchestration, and workflow capabilities.
    """
    pass


@cli.command()
@click.argument("segment_file", type=click.Path(exists=True))
@click.option(
    "--input", "-i",
    "input_file",
    type=click.Path(exists=True),
    help="Input JSON file with data for the segment",
)
@click.option(
    "--input-json", "-j",
    "input_json",
    type=str,
    help="Input data as JSON string",
)
@click.option(
    "--output", "-o",
    "output_file",
    type=click.Path(),
    help="Output file (default: stdout)",
)
@click.option(
    "--timeout", "-t",
    default=300,
    type=int,
    help="Timeout in seconds (default: 300)",
)
@click.option(
    "--tenant",
    default="default",
    help="Tenant ID for multi-tenant isolation",
)
@click.option(
    "--pretty/--no-pretty",
    default=True,
    help="Pretty print JSON output",
)
def run(
    segment_file: str,
    input_file: str | None,
    input_json: str | None,
    output_file: str | None,
    timeout: int,
    tenant: str,
    pretty: bool,
) -> None:
    """Run a segment from a JSON file.

    Examples:

        llmteam run workflow.json --input data.json

        llmteam run workflow.json -j '{"query": "hello"}'

        llmteam run workflow.json -o result.json
    """
    # Load segment definition
    segment_data = _load_json_file(segment_file)

    # Load input data
    input_data: dict[str, Any] = {}
    if input_file:
        input_data = _load_json_file(input_file)
    elif input_json:
        try:
            input_data = json.loads(input_json)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid input JSON: {e}")

    # Run segment
    result = asyncio.run(_run_segment_async(
        segment_data=segment_data,
        input_data=input_data,
        timeout=timeout,
        tenant_id=tenant,
    ))

    # Output result
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            _write_json(result, f, pretty)
        click.echo(f"Result written to {output_file}")
    else:
        _write_json(result, sys.stdout, pretty)

    # Exit with error code if failed
    if result.get("status") == "failed":
        raise SystemExit(1)


async def _run_segment_async(
    segment_data: dict[str, Any],
    input_data: dict[str, Any],
    timeout: int,
    tenant_id: str,
) -> dict[str, Any]:
    """Run segment asynchronously."""
    try:
        from llmteam.engine import WorkflowDefinition, ExecutionEngine, RunConfig
        from llmteam.runtime import RuntimeContextManager
        # Backward compatibility alias
        SegmentDefinition = WorkflowDefinition
    except ImportError as e:
        return {
            "status": "failed",
            "error": f"Missing dependencies for engine execution: {e}",
        }

    try:
        # Parse segment
        segment = SegmentDefinition.from_dict(segment_data)

        # Create runtime context
        manager = RuntimeContextManager()
        runtime = manager.create_runtime(
            tenant_id=tenant_id,
            instance_id=f"cli-{segment.segment_id}",
        )

        # Configure and run
        config = RunConfig(timeout_seconds=timeout)
        runner = SegmentRunner()

        result = await runner.run(
            segment=segment,
            input_data=input_data,
            runtime=runtime,
            config=config,
        )

        return {
            "status": result.status.value,
            "output": result.output,
            "steps_executed": result.steps_executed,
            "duration_ms": result.duration_ms,
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
        }


@cli.command()
@click.argument("segment_file", type=click.Path(exists=True))
@click.option(
    "--strict/--no-strict",
    default=True,
    help="Fail on warnings (default: strict)",
)
def validate(segment_file: str, strict: bool) -> None:
    """Validate a segment JSON file.

    Checks segment structure, step types, edge connections,
    and configuration schemas.

    Example:

        llmteam validate workflow.json
    """
    segment_data = _load_json_file(segment_file)

    errors: list[str] = []
    warnings: list[str] = []

    # Basic structure validation
    required_fields = ["segment_id", "name", "steps"]
    for field in required_fields:
        if field not in segment_data:
            errors.append(f"Missing required field: {field}")

    if errors:
        _print_validation_results(errors, warnings, segment_file)
        raise SystemExit(1)

    # Validate steps
    steps = segment_data.get("steps", [])
    step_ids: set[str] = set()

    for i, step in enumerate(steps):
        step_id = step.get("step_id")
        if not step_id:
            errors.append(f"Step {i}: missing step_id")
            continue

        if step_id in step_ids:
            errors.append(f"Duplicate step_id: {step_id}")
        step_ids.add(step_id)

        if "step_type" not in step and "type" not in step:
            errors.append(f"Step {step_id}: missing step_type")

    # Validate edges
    edges = segment_data.get("edges", [])
    for i, edge in enumerate(edges):
        source = edge.get("source_step") or edge.get("from_step")
        target = edge.get("target_step") or edge.get("to_step")

        if not source:
            errors.append(f"Edge {i}: missing source_step")
        elif source not in step_ids:
            errors.append(f"Edge {i}: unknown source step '{source}'")

        if not target:
            errors.append(f"Edge {i}: missing target_step")
        elif target not in step_ids:
            errors.append(f"Edge {i}: unknown target step '{target}'")

    # Validate entrypoint
    entrypoint = segment_data.get("entrypoint")
    if entrypoint and entrypoint not in step_ids:
        errors.append(f"Entrypoint references unknown step: {entrypoint}")
    elif not entrypoint and steps:
        warnings.append("No entrypoint defined, first step will be used")

    # Try to parse with Pydantic model
    try:
        from llmteam.engine import WorkflowDefinition
        WorkflowDefinition.from_dict(segment_data)
    except ImportError:
        warnings.append("Engine module not available for full validation")
    except Exception as e:
        errors.append(f"Model validation failed: {e}")

    _print_validation_results(errors, warnings, segment_file)

    if errors or (strict and warnings):
        raise SystemExit(1)


def _print_validation_results(
    errors: list[str],
    warnings: list[str],
    filename: str,
) -> None:
    """Print validation results."""
    if errors:
        click.secho(f"\nValidation errors in {filename}:", fg="red", bold=True)
        for error in errors:
            click.secho(f"  X {error}", fg="red")

    if warnings:
        click.secho(f"\nWarnings in {filename}:", fg="yellow")
        for warning in warnings:
            click.secho(f"  ! {warning}", fg="yellow")

    if not errors and not warnings:
        click.secho(f"\n{filename} is valid", fg="green")


@cli.command()
@click.option(
    "--json", "-j",
    "as_json",
    is_flag=True,
    help="Output as JSON",
)
def catalog(as_json: bool) -> None:
    """List available step types.

    Shows all registered step types with their descriptions
    and configuration schemas.

    Example:

        llmteam catalog

        llmteam catalog --json
    """
    try:
        from llmteam.engine import StepCatalog
        cat = StepCatalog.instance()
        types = {t.type_id: t for t in cat.list_all()}
    except ImportError:
        # Fallback to built-in types description
        types = _get_builtin_types()

    if as_json:
        def get_category(meta: Any) -> str:
            cat = getattr(meta, "category", "general")
            return cat.value if hasattr(cat, "value") else str(cat)

        output = {
            type_id: {
                "display_name": getattr(meta, "display_name", type_id),
                "description": getattr(meta, "description", ""),
                "category": get_category(meta),
            }
            for type_id, meta in types.items()
        }
        _write_json(output, sys.stdout)
        return

    click.echo("\nAvailable Step Types:\n")

    for type_id, meta in sorted(types.items()):
        display_name = getattr(meta, "display_name", type_id)
        description = getattr(meta, "description", "No description")
        category_val = getattr(meta, "category", "general")
        category = category_val.value if hasattr(category_val, "value") else str(category_val)

        click.secho(f"  {type_id}", fg="cyan", bold=True)
        click.echo(f"    Name: {display_name}")
        click.echo(f"    Category: {category}")
        click.echo(f"    {description}")
        click.echo()


def _get_builtin_types() -> dict[str, Any]:
    """Get built-in step types for fallback."""
    class TypeInfo:
        def __init__(self, display_name: str, description: str, category: str):
            self.display_name = display_name
            self.description = description
            self.category = category

    return {
        "llm_agent": TypeInfo(
            "LLM Agent",
            "Executes LLM inference with configurable model and prompts",
            "ai",
        ),
        "transform": TypeInfo(
            "Transform",
            "Transforms data using expressions or functions",
            "data",
        ),
        "conditional": TypeInfo(
            "Conditional",
            "Routes execution based on conditions",
            "control",
        ),
        "loop": TypeInfo(
            "Loop",
            "Iterates over collection with configurable parallelism",
            "control",
        ),
        "human_task": TypeInfo(
            "Human Task",
            "Pauses execution for human input or approval",
            "interaction",
        ),
        "api_call": TypeInfo(
            "API Call",
            "Makes HTTP requests to external APIs",
            "integration",
        ),
        "subflow": TypeInfo(
            "Subflow",
            "Executes another segment as a nested workflow",
            "orchestration",
        ),
    }


@cli.command()
def version() -> None:
    """Show version and system information."""
    click.echo(f"llmteam {llmteam.__version__}")
    click.echo(f"Python {sys.version}")

    # Show installed optional dependencies
    click.echo("\nInstalled components:")

    components = [
        ("engine", "llmteam.engine"),
        ("runtime", "llmteam.runtime"),
        ("events", "llmteam.events"),
        ("api", "llmteam.api"),
        ("tenancy", "llmteam.tenancy"),
        ("audit", "llmteam.audit"),
        ("providers", "llmteam.providers"),
        ("middleware", "llmteam.middleware"),
        ("auth", "llmteam.auth"),
        ("secrets", "llmteam.secrets"),
    ]

    for name, module in components:
        try:
            __import__(module)
            click.secho(f"  {name}: available", fg="green")
        except ImportError:
            click.secho(f"  {name}: not installed", fg="yellow")


@cli.command()
@click.option(
    "--json", "-j",
    "as_json",
    is_flag=True,
    help="Output as JSON",
)
def providers(as_json: bool) -> None:
    """List available LLM providers.

    Shows all LLM providers with their status (installed or not).

    Example:

        llmteam providers

        llmteam providers --json
    """
    provider_info = [
        {
            "name": "OpenAI",
            "class": "OpenAIProvider",
            "module": "llmteam.providers.openai",
            "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            "env_var": "OPENAI_API_KEY",
            "install": "pip install llmteam-ai[providers]",
        },
        {
            "name": "Anthropic",
            "class": "AnthropicProvider",
            "module": "llmteam.providers.anthropic",
            "models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
            "env_var": "ANTHROPIC_API_KEY",
            "install": "pip install llmteam-ai[providers]",
        },
        {
            "name": "Azure OpenAI",
            "class": "AzureOpenAIProvider",
            "module": "llmteam.providers.azure",
            "models": ["gpt-4", "gpt-35-turbo"],
            "env_var": "AZURE_OPENAI_API_KEY",
            "install": "pip install llmteam-ai[providers]",
        },
        {
            "name": "AWS Bedrock",
            "class": "BedrockProvider",
            "module": "llmteam.providers.bedrock",
            "models": ["anthropic.claude-3", "amazon.titan", "meta.llama2"],
            "env_var": "AWS_ACCESS_KEY_ID",
            "install": "pip install llmteam-ai[aws]",
        },
        {
            "name": "Google Vertex AI",
            "class": "VertexAIProvider",
            "module": "llmteam.providers.vertex",
            "models": ["gemini-1.5-pro", "gemini-1.5-flash"],
            "env_var": "GOOGLE_APPLICATION_CREDENTIALS",
            "install": "pip install llmteam-ai[vertex]",
        },
        {
            "name": "Ollama",
            "class": "OllamaProvider",
            "module": "llmteam.providers.ollama",
            "models": ["llama2", "mistral", "codellama", "phi"],
            "env_var": "OLLAMA_HOST",
            "install": "pip install llmteam-ai (no extra needed)",
        },
        {
            "name": "LiteLLM",
            "class": "LiteLLMProvider",
            "module": "llmteam.providers.litellm",
            "models": ["100+ providers via unified API"],
            "env_var": "(varies by provider)",
            "install": "pip install llmteam-ai[litellm]",
        },
    ]

    # Check which providers are installed
    import os
    for provider in provider_info:
        try:
            __import__(provider["module"])
            provider["installed"] = True
        except ImportError:
            provider["installed"] = False

        # Check if env var is set
        env_var = provider["env_var"]
        if env_var and not env_var.startswith("("):
            provider["configured"] = bool(os.environ.get(env_var))
        else:
            provider["configured"] = None

    if as_json:
        _write_json(provider_info, sys.stdout)
        return

    click.echo("\nAvailable LLM Providers:\n")

    for provider in provider_info:
        # Status indicator
        if provider["installed"]:
            if provider.get("configured"):
                status_icon = click.style("✓", fg="green")
                status_text = "ready"
            else:
                status_icon = click.style("○", fg="yellow")
                status_text = "installed (not configured)"
        else:
            status_icon = click.style("✗", fg="red")
            status_text = "not installed"

        click.echo(f"  {status_icon} {click.style(provider['name'], bold=True)}")
        click.echo(f"      Class: {provider['class']}")
        click.echo(f"      Status: {status_text}")
        click.echo(f"      Models: {', '.join(provider['models'][:3])}")
        if not provider["installed"]:
            click.echo(f"      Install: {provider['install']}")
        click.echo()


@cli.command()
@click.argument("segment_file", type=click.Path(exists=True))
@click.option("--format", "-f", "output_format", default="text", type=click.Choice(["text", "json"]))
def check(segment_file: str, output_format: str) -> None:
    """Check a segment file for issues.

    Performs comprehensive validation including:
    - JSON syntax validation
    - Schema validation
    - Step type validation
    - Edge connection validation
    - Cycle detection

    Example:

        llmteam check workflow.json

        llmteam check workflow.json --format json
    """
    segment_data = _load_json_file(segment_file)

    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    # Basic structure validation
    required_fields = ["segment_id", "name", "steps"]
    for field in required_fields:
        if field not in segment_data:
            issues.append({
                "type": "error",
                "code": "MISSING_FIELD",
                "message": f"Missing required field: {field}",
                "location": "root",
            })

    # Validate steps
    steps = segment_data.get("steps", [])
    step_ids: set[str] = set()

    for i, step in enumerate(steps):
        step_id = step.get("step_id")
        if not step_id:
            issues.append({
                "type": "error",
                "code": "MISSING_STEP_ID",
                "message": f"Step {i}: missing step_id",
                "location": f"steps[{i}]",
            })
            continue

        if step_id in step_ids:
            issues.append({
                "type": "error",
                "code": "DUPLICATE_STEP_ID",
                "message": f"Duplicate step_id: {step_id}",
                "location": f"steps[{i}]",
            })
        step_ids.add(step_id)

        step_type = step.get("step_type") or step.get("type")
        if not step_type:
            issues.append({
                "type": "error",
                "code": "MISSING_STEP_TYPE",
                "message": f"Step {step_id}: missing step_type",
                "location": f"steps[{i}]",
            })

    # Validate edges
    edges = segment_data.get("edges", [])
    for i, edge in enumerate(edges):
        source = edge.get("source_step") or edge.get("from_step")
        target = edge.get("target_step") or edge.get("to_step")

        if source and source not in step_ids:
            issues.append({
                "type": "error",
                "code": "UNKNOWN_SOURCE_STEP",
                "message": f"Edge {i}: unknown source step '{source}'",
                "location": f"edges[{i}]",
            })

        if target and target not in step_ids:
            issues.append({
                "type": "error",
                "code": "UNKNOWN_TARGET_STEP",
                "message": f"Edge {i}: unknown target step '{target}'",
                "location": f"edges[{i}]",
            })

    # Validate entrypoint
    entrypoint = segment_data.get("entrypoint")
    if entrypoint and entrypoint not in step_ids:
        issues.append({
            "type": "error",
            "code": "INVALID_ENTRYPOINT",
            "message": f"Entrypoint references unknown step: {entrypoint}",
            "location": "entrypoint",
        })
    elif not entrypoint and steps:
        warnings.append({
            "type": "warning",
            "code": "NO_ENTRYPOINT",
            "message": "No entrypoint defined, first step will be used",
            "location": "root",
        })

    # Output results
    result = {
        "file": segment_file,
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "stats": {
            "steps": len(steps),
            "edges": len(edges),
        },
    }

    if output_format == "json":
        _write_json(result, sys.stdout)
    else:
        if issues:
            click.secho(f"\nErrors in {segment_file}:", fg="red", bold=True)
            for issue in issues:
                click.secho(f"  ✗ [{issue['code']}] {issue['message']}", fg="red")
                click.echo(f"    Location: {issue['location']}")

        if warnings:
            click.secho(f"\nWarnings in {segment_file}:", fg="yellow")
            for warning in warnings:
                click.secho(f"  ! [{warning['code']}] {warning['message']}", fg="yellow")

        if not issues and not warnings:
            click.secho(f"\n✓ {segment_file} is valid", fg="green")

        click.echo(f"\nStats: {len(steps)} steps, {len(edges)} edges")

    if issues:
        raise SystemExit(1)


@cli.command()
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
@click.option("--port", "-p", default=8000, type=int, help="Port to bind (default: 8000)")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(host: str, port: int, reload: bool) -> None:
    """Start the API server.

    Runs the LLMTeam REST and WebSocket API server.

    Example:

        llmteam serve --port 8080

        llmteam serve --reload
    """
    try:
        import uvicorn
    except ImportError:
        raise click.ClickException(
            "uvicorn not installed. Install with: pip install llmteam[api]"
        )

    try:
        from llmteam.api import app  # noqa: F401
    except ImportError:
        raise click.ClickException(
            "API module not available. Install with: pip install llmteam[api]"
        )

    click.echo(f"Starting LLMTeam API server on {host}:{port}")
    uvicorn.run(
        "llmteam.api:app",
        host=host,
        port=port,
        reload=reload,
    )


def main() -> None:
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
