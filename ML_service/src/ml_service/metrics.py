from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

INGESTED = Counter(
    "ml_ingested_total",
    "Accepted market messages",
    labelnames=("kind",),
)
DUPLICATES = Counter(
    "ml_duplicate_total",
    "Repeated market states",
    labelnames=("kind",),
)
DROPPED_PAIR_WORK = Counter(
    "ml_pair_work_dropped_total",
    "Pair evaluations coalesced or dropped by latest-wins scheduling",
)
MODEL_INFERENCE = Histogram(
    "ml_model_inference_seconds",
    "Model inference latency",
    labelnames=("model", "device"),
    buckets=(0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.05, 0.1),
)
MODEL_ERRORS = Counter(
    "ml_model_errors_total",
    "Model inference errors",
    labelnames=("model",),
)
PENDING_PAIRS = Gauge(
    "ml_pending_pairs",
    "Unique pair evaluations waiting in the latest-wins scheduler",
)

