from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from ml_service.predictors.base import PredictionContext
from ml_service.predictors.transformer.predictor import TransformerPredictor


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "max_ms": float(array.max()),
        "stdev_ms": float(statistics.pstdev(values)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark pure and end-to-end Transformer serving latency."
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("/models/transformer"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--max-p95-ms", type=float, default=100.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    predictor = TransformerPredictor(
        "transformer", args.bundle_dir, args.device
    )
    predictor.load()

    l2_columns = list(predictor.contract["l2_feature_columns"])
    ohlcv_columns = list(predictor.contract["ohlcv_feature_columns"])
    feature_columns = list(predictor.contract["feature_columns"])
    local_steps = int(predictor.contract["local_history_steps"])
    long_tokens = int(predictor.contract["long_history_tokens"])
    stride = int(predictor.contract["long_history_stride_steps"])
    needed = max(local_steps, (long_tokens - 1) * stride + 1)

    tensor_ready = {
        "local": np.zeros((local_steps, len(l2_columns)), dtype=np.float32),
        "long": np.zeros((long_tokens, len(l2_columns)), dtype=np.float32),
        "ohlcv_state": np.zeros(2 * len(ohlcv_columns), dtype=np.float32),
        "position_state": np.zeros(6, dtype=np.float32),
        "entry_snapshot": np.zeros(6, dtype=np.float32),
        "pair_type_id": 0,
        "direction_id": 0,
        "leg1_exchange_id": 0,
        "leg2_exchange_id": 0,
        "pair_hash_id": 0,
    }
    tensor_context = PredictionContext(
        features={}, transformer_input=tensor_ready
    )

    history_row = {column: 0.0 for column in feature_columns}
    history_row.update(
        {
            "pair_id": "benchmark-pair",
            "pair_type": "perp_perp_cross_exchange",
            "direction_code": 0,
            "leg1_exchange": "bitget",
            "leg2_exchange": "gate",
            "current_entry_executable": 1.0,
            "current_entry_fill_share": 1.0,
        }
    )
    history = [history_row] * needed

    def history_context(offset: int) -> PredictionContext:
        return PredictionContext(
            features=history_row,
            history=history,
            history_timestamps=list(range(offset, offset + needed)),
            position_state=[0.0] * 6,
            pair_id="benchmark-pair",
            pair_type="perp_perp_cross_exchange",
            direction_code=0,
        )

    predictor.warmup()
    cold_history_ms = predictor.predict(history_context(0)).latency_ms
    for offset in range(1, args.warmup + 1):
        predictor.predict(tensor_context)
        predictor.predict(history_context(offset))

    tensor_latencies = [
        predictor.predict(tensor_context).latency_ms
        for _ in range(args.iterations)
    ]
    history_latencies = []
    for offset in range(
        args.warmup + 1,
        args.warmup + 1 + args.iterations,
    ):
        history_latencies.append(
            predictor.predict(history_context(offset)).latency_ms
        )
    report = {
        "device": predictor.device,
        "version": predictor.version,
        "iterations": args.iterations,
        "history_steps": needed,
        "local_steps": local_steps,
        "long_tokens": long_tokens,
        "tensor_ready": summarize(tensor_latencies),
        "cold_full_history_ms": cold_history_ms,
        "full_history": summarize(history_latencies),
        "target_p95_ms": args.max_p95_ms,
    }
    report["passes_target"] = (
        report["device"] == "cuda"
        and report["full_history"]["p95_ms"] <= args.max_p95_ms
    )

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    predictor.close()
    return 0 if report["passes_target"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
