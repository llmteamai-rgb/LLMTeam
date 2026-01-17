"""
OpenTelemetry Tracing Integration.

Provides distributed tracing for LLMTeam workflows using OpenTelemetry.

Usage:
    from llmteam.observability.tracing import TracingConfig, setup_tracing, TracingMiddleware

    # Configure tracing
    setup_tracing(TracingConfig(
        service_name="my-workflow",
        exporter="otlp",
        endpoint="http://localhost:4317",
    ))

    # Use tracing middleware
    stack.use(TracingMiddleware())
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from contextlib import contextmanager
from datetime import datetime
import os

from llmteam.runtime import StepContext


@dataclass
class TracingConfig:
    """Configuration for OpenTelemetry tracing."""

    service_name: str = "llmteam"
    service_version: str = ""

    # Exporter configuration
    exporter: str = "console"  # "console", "otlp", "jaeger", "zipkin"
    endpoint: str = ""

    # Sampling
    sample_rate: float = 1.0  # 1.0 = sample all

    # Additional attributes
    resource_attributes: dict[str, str] = field(default_factory=dict)

    # Enable/disable
    enabled: bool = True

    def __post_init__(self) -> None:
        # Load from environment if not set
        if not self.endpoint:
            self.endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        if not self.service_version:
            try:
                import llmteam
                self.service_version = llmteam.__version__
            except ImportError:
                self.service_version = "unknown"


# Global tracer instance
_tracer: Optional[Any] = None
_config: Optional[TracingConfig] = None


def setup_tracing(config: TracingConfig) -> None:
    """
    Initialize OpenTelemetry tracing.

    Args:
        config: Tracing configuration

    Note:
        Requires optional dependency: pip install llmteam-ai[tracing]
        or: pip install opentelemetry-api opentelemetry-sdk
    """
    global _tracer, _config

    if not config.enabled:
        _config = config
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.resources import Resource

        # Create resource
        resource_attrs = {
            "service.name": config.service_name,
            "service.version": config.service_version,
            **config.resource_attributes,
        }
        resource = Resource.create(resource_attrs)

        # Create provider
        provider = TracerProvider(resource=resource)

        # Configure exporter
        _configure_exporter(provider, config)

        # Set global provider
        trace.set_tracer_provider(provider)

        # Get tracer
        _tracer = trace.get_tracer(
            config.service_name,
            config.service_version,
        )
        _config = config

    except ImportError:
        # OpenTelemetry not installed, use no-op tracer
        _tracer = NoOpTracer()
        _config = config


def _configure_exporter(provider: Any, config: TracingConfig) -> None:
    """Configure the span exporter based on config."""
    try:
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        if config.exporter == "console":
            processor = BatchSpanProcessor(ConsoleSpanExporter())
            provider.add_span_processor(processor)

        elif config.exporter == "otlp":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=config.endpoint)
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)

        elif config.exporter == "jaeger":
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter
            exporter = JaegerExporter(
                collector_endpoint=config.endpoint or "http://localhost:14268/api/traces",
            )
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)

        elif config.exporter == "zipkin":
            from opentelemetry.exporter.zipkin.json import ZipkinExporter
            exporter = ZipkinExporter(
                endpoint=config.endpoint or "http://localhost:9411/api/v2/spans",
            )
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)

    except ImportError as e:
        # Specific exporter not available, fall back to console
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)


def get_tracer() -> Any:
    """Get the configured tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = NoOpTracer()
    return _tracer


class NoOpSpan:
    """No-op span for when tracing is disabled."""

    def __init__(self, name: str = "") -> None:
        self.name = name

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[dict] = None) -> None:
        pass

    def __enter__(self) -> "NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def end(self) -> None:
        pass


class NoOpTracer:
    """No-op tracer for when OpenTelemetry is not installed."""

    def start_span(self, name: str, **kwargs: Any) -> NoOpSpan:
        return NoOpSpan(name)

    @contextmanager
    def start_as_current_span(self, name: str, **kwargs: Any):
        yield NoOpSpan(name)


@dataclass
class SpanAttributes:
    """Common span attribute names for LLMTeam."""

    # Segment attributes
    SEGMENT_ID = "llmteam.segment.id"
    SEGMENT_NAME = "llmteam.segment.name"
    RUN_ID = "llmteam.run.id"

    # Step attributes
    STEP_ID = "llmteam.step.id"
    STEP_TYPE = "llmteam.step.type"
    STEP_NAME = "llmteam.step.name"

    # Tenant attributes
    TENANT_ID = "llmteam.tenant.id"

    # LLM attributes
    LLM_MODEL = "llmteam.llm.model"
    LLM_TOKENS_INPUT = "llmteam.llm.tokens.input"
    LLM_TOKENS_OUTPUT = "llmteam.llm.tokens.output"
    LLM_PROVIDER = "llmteam.llm.provider"

    # HTTP attributes
    HTTP_METHOD = "http.method"
    HTTP_URL = "http.url"
    HTTP_STATUS_CODE = "http.status_code"

    # Error attributes
    ERROR_TYPE = "llmteam.error.type"
    ERROR_MESSAGE = "llmteam.error.message"


class TracingMiddleware:
    """
    Middleware for adding OpenTelemetry traces to step execution.

    Creates spans for each step with relevant attributes.
    """

    name = "tracing"
    priority = 5
    enabled = True

    def __init__(
        self,
        record_input: bool = False,
        record_output: bool = False,
        custom_attributes: Optional[Callable[[StepContext, dict], dict]] = None,
    ) -> None:
        """
        Initialize tracing middleware.

        Args:
            record_input: Whether to record input data as span attribute
            record_output: Whether to record output data as span attribute
            custom_attributes: Function to generate custom attributes
        """
        self.record_input = record_input
        self.record_output = record_output
        self.custom_attributes = custom_attributes

    def should_run(
        self,
        step_type: str,
        step_id: str,
        middleware_ctx: Any,
    ) -> bool:
        """Check if tracing should run."""
        return self.enabled and _config is not None and _config.enabled

    async def __call__(
        self,
        ctx: StepContext,
        config: dict,
        input_data: dict,
        next_handler: Callable,
        middleware_ctx: Any,
    ) -> Any:
        tracer = get_tracer()

        span_name = f"{middleware_ctx.step_type}:{middleware_ctx.step_id}"

        with tracer.start_as_current_span(span_name) as span:
            # Set common attributes
            span.set_attribute(SpanAttributes.SEGMENT_ID, middleware_ctx.segment_id)
            span.set_attribute(SpanAttributes.RUN_ID, middleware_ctx.run_id)
            span.set_attribute(SpanAttributes.STEP_ID, middleware_ctx.step_id)
            span.set_attribute(SpanAttributes.STEP_TYPE, middleware_ctx.step_type)
            span.set_attribute(SpanAttributes.TENANT_ID, middleware_ctx.tenant_id)

            # Record input if configured
            if self.record_input:
                import json
                span.set_attribute("llmteam.input", json.dumps(input_data, default=str)[:1000])

            # Custom attributes
            if self.custom_attributes:
                for key, value in self.custom_attributes(ctx, config).items():
                    span.set_attribute(key, value)

            try:
                result = await next_handler(ctx, config, input_data)

                # Record output if configured
                if self.record_output and result:
                    import json
                    span.set_attribute("llmteam.output", json.dumps(result, default=str)[:1000])

                # Add success event
                span.add_event("step.completed", {
                    "duration_ms": middleware_ctx.middleware_data.get("timing", {}).get("duration_ms", 0),
                })

                return result

            except Exception as e:
                # Record exception
                span.record_exception(e)
                span.set_attribute(SpanAttributes.ERROR_TYPE, type(e).__name__)
                span.set_attribute(SpanAttributes.ERROR_MESSAGE, str(e))

                try:
                    from opentelemetry.trace import StatusCode
                    span.set_status(StatusCode.ERROR, str(e))
                except ImportError:
                    pass

                raise


def trace_segment(segment_id: str, run_id: str, tenant_id: str):
    """
    Context manager for tracing a segment execution.

    Usage:
        with trace_segment("my-segment", "run-123", "tenant-1") as span:
            # Run segment
            span.add_event("custom_event")
    """
    tracer = get_tracer()

    span_name = f"segment:{segment_id}"
    span = tracer.start_span(span_name)
    span.set_attribute(SpanAttributes.SEGMENT_ID, segment_id)
    span.set_attribute(SpanAttributes.RUN_ID, run_id)
    span.set_attribute(SpanAttributes.TENANT_ID, tenant_id)

    @contextmanager
    def _context():
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_attribute(SpanAttributes.ERROR_TYPE, type(e).__name__)
            raise
        finally:
            span.end()

    return _context()


def trace_llm_call(
    model: str,
    provider: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
):
    """
    Context manager for tracing an LLM call.

    Usage:
        with trace_llm_call("gpt-4", "openai", input_tokens=100, output_tokens=50) as span:
            response = await llm.complete(prompt)
    """
    tracer = get_tracer()

    span_name = f"llm:{provider}:{model}"
    span = tracer.start_span(span_name)
    span.set_attribute(SpanAttributes.LLM_MODEL, model)
    span.set_attribute(SpanAttributes.LLM_PROVIDER, provider)

    @contextmanager
    def _context():
        try:
            yield span
            span.set_attribute(SpanAttributes.LLM_TOKENS_INPUT, input_tokens)
            span.set_attribute(SpanAttributes.LLM_TOKENS_OUTPUT, output_tokens)
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            span.end()

    return _context()
