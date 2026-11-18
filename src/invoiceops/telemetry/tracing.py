"""OpenTelemetry tracing for requests, ADK agent steps, LLM usage, and tool calls (Section 6)."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Span, Status, StatusCode

from invoiceops.config import AppConfig


class JsonlSpanExporter(SpanExporter):
    """Exports OpenTelemetry spans to a local JSONL file and optional in-memory buffer."""

    def __init__(self, output_path: Path, keep_in_memory: bool = True) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.keep_in_memory = keep_in_memory
        self.spans: list[dict[str, Any]] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        rows: list[str] = []
        for sp in spans:
            ctx = sp.get_span_context()
            parent = sp.parent
            start_ns = sp.start_time or 0
            end_ns = sp.end_time or start_ns
            duration_ms = round((end_ns - start_ns) / 1_000_000.0, 3)
            payload: dict[str, Any] = {
                "trace_id": f"{ctx.trace_id:032x}",
                "span_id": f"{ctx.span_id:016x}",
                "parent_span_id": f"{parent.span_id:016x}" if parent else None,
                "name": sp.name,
                "status": sp.status.status_code.name,
                "start_time_ns": start_ns,
                "end_time_ns": end_ns,
                "duration_ms": duration_ms,
                "attributes": dict(sp.attributes or {}),
            }
            if self.keep_in_memory:
                self.spans.append(payload)
            rows.append(json.dumps(payload, default=str))

        if rows:
            with self.output_path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(rows) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


class InvoiceOpsTelemetry:
    """Isolated OpenTelemetry TracerProvider wrapper for local JSONL, OTLP, and Cloud Trace."""

    def __init__(
        self,
        config: AppConfig,
        jsonl_trace_path: Path,
        keep_in_memory: bool = True,
    ) -> None:
        resource = Resource.create(
            {
                "service.name": config.project_name,
                "service.version": config.config_version,
                "cloud.region": config.platform.region,
                "deployment.environment": config.platform.mode,
            }
        )
        self.provider = TracerProvider(resource=resource)
        self.jsonl_exporter = JsonlSpanExporter(
            output_path=jsonl_trace_path,
            keep_in_memory=keep_in_memory,
        )
        self.provider.add_span_processor(SimpleSpanProcessor(self.jsonl_exporter))

        # Optional OTLP HTTP collector export when explicitly enabled
        if os.getenv("INVOICEOPS_ENABLE_OTLP", "").lower() in {"1", "true", "yes"}:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
            otlp_exp = OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
            self.provider.add_span_processor(BatchSpanProcessor(otlp_exp))

        # Automatic Google Cloud Trace export when running in Phase P4 GCP Enterprise mode
        if config.platform.mode == "gcp_enterprise":
            try:
                from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

                gcp_exp = CloudTraceSpanExporter(project_id=config.platform.gcp_project_id)
                self.provider.add_span_processor(BatchSpanProcessor(gcp_exp))
            except ImportError:
                pass

        self.tracer = self.provider.get_tracer("invoiceops.tracer", config.config_version)

    @contextmanager
    def request_span(
        self,
        case_id: str,
        vendor_id: str,
        version: str,
    ) -> Iterator[Span]:
        """Root span covering end-to-end resolution of one invoice exception request."""
        with self.tracer.start_as_current_span("invoiceops.resolve_invoice") as span:
            span.set_attribute("invoiceops.case_id", case_id)
            span.set_attribute("invoiceops.vendor_id", vendor_id)
            span.set_attribute("invoiceops.version", version)
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    @contextmanager
    def agent_span(self, agent_name: str, model_id: str) -> Iterator[Span]:
        """Child span for one specialist ADK agent invocation and its LLM token/cost usage."""
        with self.tracer.start_as_current_span(f"agent.{agent_name}") as span:
            span.set_attribute("agent.name", agent_name)
            span.set_attribute("gen_ai.request.model", model_id)
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    @contextmanager
    def tool_span(self, tool_name: str) -> Iterator[Span]:
        """Child span for an MCP ERP lookup, vector search, or deterministic matching tool."""
        with self.tracer.start_as_current_span(f"tool.{tool_name}") as span:
            span.set_attribute("tool.name", tool_name)
            span.set_attribute("tool.retries", 0)
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    def record_llm_metrics_on_span(
        self,
        span: Span,
        resolved_model: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float,
        llm_calls: int = 1,
    ) -> None:
        """Attach standardized Section 6 LLM telemetry attributes to an active span."""
        span.set_attribute("gen_ai.response.model", resolved_model)
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        span.set_attribute("invoiceops.estimated_cost_usd", round(estimated_cost_usd, 8))
        span.set_attribute("invoiceops.llm_calls", llm_calls)


def get_current_trace_id_hex() -> str | None:
    """Return the active 32-hex OpenTelemetry trace ID if recording."""
    ctx = trace.get_current_span().get_span_context()
    if ctx and ctx.is_valid:
        return f"{ctx.trace_id:032x}"
    return None
