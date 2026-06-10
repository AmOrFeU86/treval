"""
OpenTelemetry bridge for treval.

Exports native treval spans to any OTLP-compatible backend
(Langfuse, Grafana, Phoenix, Datadog, etc.).

Usage:
    # Export all spans to an OTLP endpoint
    treval export otlp --endpoint http://localhost:4317

    # In code:
    from treval.otel import OtelExporter
    exporter = OtelExporter(endpoint="http://localhost:4317")
    count = exporter.export_all()
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from treval.db import SpanStore, DB_PATH


# ─── treval → OTEL type maps ───

SPAN_TYPE_KIND = {
    "AGENT": "INTERNAL",
    "OPERATION": "INTERNAL",
    "TOOL": "INTERNAL",
    "LLM": "INTERNAL",
    "WORKFLOW": "INTERNAL",
    "TASK": "INTERNAL",
}

SPAN_TYPE_ATTR = {
    "AGENT": "treval.agent",
    "OPERATION": "treval.operation",
    "TOOL": "treval.tool",
    "LLM": "treval.llm",
    "WORKFLOW": "treval.workflow",
    "TASK": "treval.task",
}


class OtelExporter:
    """Exports treval spans to an OTLP endpoint.

    Supports both gRPC (port 4317) and HTTP/protobuf (port 4318).

    Args:
        endpoint: URL of the OTLP collector. If None, reads from
                  TREVAL_OTLP_ENDPOINT or send to console.
        service_name: Service name for OTEL (default: "treval")
        use_console: If True, also print spans to console
    """

    _instance = None

    def __init__(
        self,
        endpoint: str | None = None,
        service_name: str = "treval",
        use_console: bool = False,
    ):
        self.endpoint = endpoint or os.environ.get("TREVAL_OTLP_ENDPOINT")
        self.service_name = service_name
        self.use_console = use_console or (not self.endpoint)
        self._provider = None
        self._tracer = None

    @classmethod
    def start_streaming(cls, endpoint: str | None = None,
                        service_name: str = "treval") -> "OtelExporter":
        """Creates an exporter and registers it for real-time streaming.

        Spans saved in SpanStore are automatically exported.
        """
        from treval.db import add_post_save_hook
        exporter = cls(endpoint=endpoint, service_name=service_name)
        cls._instance = exporter
        add_post_save_hook(exporter.export_streaming)
        return exporter

    @property
    def tracer(self):
        if self._tracer is None:
            self._setup()
        return self._tracer

    def _setup(self):
        """Configures the OTEL TracerProvider."""
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )

        provider = TracerProvider(
            resource=self._resource(),
        )

        if self.endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
                exporter = OTLPSpanExporter(endpoint=self.endpoint, insecure=True)
                provider.add_span_processor(BatchSpanProcessor(exporter))
            except Exception:
                # Fallback to HTTP/protobuf
                try:
                    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                        OTLPSpanExporter,
                    )
                    exporter = OTLPSpanExporter(endpoint=self.endpoint.replace("4317", "4318"))
                    provider.add_span_processor(BatchSpanProcessor(exporter))
                except Exception as e:
                    import warnings
                    warnings.warn(f"Could not connect to OTLP endpoint {self.endpoint}: {e}")

        if self.use_console:
            provider.add_span_processor(
                SimpleSpanProcessor(ConsoleSpanExporter())
            )

        self._provider = provider
        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer("treval", "0.1.0")

    def _resource(self):
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.semconv.resource import ResourceAttributes
        return Resource.create({
            ResourceAttributes.SERVICE_NAME: self.service_name,
            ResourceAttributes.SERVICE_VERSION: "0.1.0",
        })

    def _span_to_otel(self, span: dict) -> None:
        """Converts a treval span to an OTEL span and emits it."""
        from opentelemetry import trace
        from opentelemetry.trace import SpanKind

        span_type = span.get("type", "TOOL")
        span_name = span.get("name", "unknown")
        status_ok = span.get("status") == "ok"

        attrs = {
            "treval.type": span_type,
            "treval.span_id": span["id"],
            "treval.name": span_name,
        }

        if span.get("parent_id"):
            attrs["treval.parent_id"] = span["parent_id"]

        # LLM semantic attributes (OpenInference)
        if span_type == "LLM":
            if span.get("input"):
                attrs["gen_ai.prompt"] = span["input"]
            if span.get("output"):
                attrs["gen_ai.completion"] = span["output"]
            if "deepseek" in span_name.lower() or "gpt" in span_name.lower():
                attrs["gen_ai.model"] = span_name
                attrs["gen_ai.request.model"] = span_name

        # Generic input/output
        if span.get("input"):
            attrs["treval.input"] = span["input"]
        if span.get("output"):
            attrs["treval.output"] = span["output"]

        # Duration
        duration_ns = 0
        if span.get("duration_ms") is not None:
            duration_ns = int(span["duration_ms"] * 1_000_000)

        # Timestamp (created_at)
        import datetime

        try:
            created = datetime.datetime.fromisoformat(span["created_at"])
            timestamp_ns = int(created.timestamp() * 1_000_000_000)
        except (ValueError, TypeError):
            timestamp_ns = None

        kind_name = SPAN_TYPE_KIND.get(span_type, "INTERNAL")
        kind = getattr(SpanKind, kind_name, SpanKind.INTERNAL)

        otel_span = self.tracer.start_span(
            name=span_name,
            kind=kind,
            attributes=attrs,
            start_time=timestamp_ns,
        )
        if status_ok:
            otel_span.set_status(trace.Status(trace.StatusCode.OK))
        else:
            otel_span.set_status(trace.Status(
                trace.StatusCode.ERROR,
                span.get("output", ""),
            ))
        if timestamp_ns and duration_ns:
            otel_span.end(end_time=timestamp_ns + duration_ns)
        else:
            otel_span.end()

    def export_all(self, limit: int = 1000) -> int:
        """Exports all spans from SQLite to OTEL.

        Returns:
            Number of exported spans.
        """
        store = SpanStore()
        spans = store.list_spans(limit=limit)
        for span in spans:
            self._span_to_otel(span)
        return len(spans)

    def export_one(self, span_id: int) -> bool:
        """Exports a specific span to OTEL."""
        store = SpanStore()
        span = store.get(span_id)
        if span:
            self._span_to_otel(span)
            return True
        return False

    def export_streaming(self, span: dict) -> None:
        """Exports a span in real time (called from SpanStore.save)."""
        try:
            self._span_to_otel(span)
        except Exception:
            pass  # Don't break the main app if OTEL fails