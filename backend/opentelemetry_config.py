from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor


def setup_otel(app):
    # Identify the service for Grafana Tempo
    resource = Resource.create({
        "service.name": "rag_api"
    })

    # Create trace provider
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    # Export spans to OTEL Collector
    exporter = OTLPSpanExporter(
        endpoint="http://rag_otel:4318/v1/traces"   # correct endpoint
    )

    # Add span processor
    span_processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(span_processor)

    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)

    # Instrument outgoing calls (OpenAI, requests, etc.)
    RequestsInstrumentor().instrument()
