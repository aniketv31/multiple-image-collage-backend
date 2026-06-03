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

STITCH_METHOD = Counter(
    "asset_unified_view_method_total",
    "Unified view generation method used",
    ["method"],
)
