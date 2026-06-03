"""Prometheus metrics."""

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "asset_analyze_requests_total",
    "Total asset analyze requests",
    ["status"],
)

REQUEST_LATENCY = Histogram(
    "asset_analyze_latency_seconds",
    "Asset analyze request latency",
    buckets=(0.5, 1, 2, 5, 10, 15, 30, 60),
)

ANALYSIS_METHOD = Counter(
    "asset_analysis_method_total",
    "Analysis method used",
    ["method"],
)
