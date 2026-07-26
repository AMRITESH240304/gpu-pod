"""
GPU Pod — OpenTelemetry integration
=====================================
Shared helper that all three components (server, worker, client) use to send
traces, logs, and metrics to SigNoz (localhost:4317 gRPC).

Import and call `init_otel()` at startup, then use the tracer anywhere.

Usage:
    from gpu_pod_otel import init_otel, tracer, logger, meter

    init_otel("gpu-pod-server", "server")

    with tracer.start_as_current_span("do-work"):
        logger.info("doing work")
        counter = meter.create_counter("work.count")
        counter.add(1)
"""

import os
import logging

# OTel SDK
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION, DEPLOYMENT_ENVIRONMENT
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

# OTel logging
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

# Instrumentations
from opentelemetry.instrumentation.logging import LoggingInstrumentor

SIGNOZ_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

tracer = trace.NoOpTracer()
logger = logging.getLogger("gpu-pod")
meter = metrics.noop_meter()

_INITIALIZED = False


def init_otel(
    service_name: str,
    version: str = "0.1.0",
    env: str = "production",
    otlp_endpoint: str | None = None,
):
    """
    Call once at startup to initialise OTel tracing, logging, and metrics.

    Args:
        service_name: e.g. "gpu-pod-server", "gpu-pod-worker", "gpu-pod-client"
        version:  service version
        env:      e.g. "production", "development"
        otlp_endpoint: OTLP gRPC endpoint (default: localhost:4317)
    """
    global tracer, logger, meter, _INITIALIZED

    if _INITIALIZED:
        return
    _INITIALIZED = True

    endpoint = otlp_endpoint or SIGNOZ_ENDPOINT

    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: version,
        DEPLOYMENT_ENVIRONMENT: env,
    })

    # ── Tracing ──
    trace_provider = TracerProvider(resource=resource)
    span_processor = BatchSpanProcessor(
        OTLPSpanExporter(endpoint=endpoint, insecure=True)
    )
    trace_provider.add_span_processor(span_processor)
    trace.set_tracer_provider(trace_provider)
    tracer = trace.get_tracer(service_name, version)

    # ── Logging ──
    logger_provider = LoggerProvider(resource=resource)
    log_processor = BatchLogRecordProcessor(
        OTLPLogExporter(endpoint=endpoint, insecure=True)
    )
    logger_provider.add_log_record_processor(log_processor)
    set_logger_provider(logger_provider)

    # OTel logging handler — captures stdlib logging and sends to SigNoz
    handler = LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)
    logging.getLogger().addHandler(handler)
    LoggingInstrumentor().instrument(set_logging_format=True)

    logger = logging.getLogger(service_name)
    logger.setLevel(logging.INFO)

    # ── Metrics ──
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=endpoint, insecure=True),
        export_interval_millis=30_000,
    )
    metric_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(metric_provider)
    meter = metrics.get_meter(service_name, version)

    logger.info("OTel initialised", extra={"service": service_name, "endpoint": endpoint})
    print(f"[OTEL] {service_name} → SigNoz @ {endpoint}")


# ── Convenience: shut down ──

def shutdown():
    """Flush and shut down all OTel providers."""
    trace.get_tracer_provider().shutdown()
    metrics.get_meter_provider().shutdown()
    logger_provider = getattr(logging.getLogger(), "logger_provider", None)
    if logger_provider:
        logger_provider.shutdown()
