"""Сбор компактной телеметрии для operator dashboard."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ml_service.context import AppContext


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return float(ordered[index])


@lru_cache(maxsize=8)
def _transformer_history_requirement(contract_path: str) -> int:
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    return max(
        int(contract["local_history_steps"]),
        (int(contract["long_history_tokens"]) - 1)
        * int(contract["long_history_stride_steps"])
        + 1,
    )


def _required_history(context: AppContext) -> int:
    requirements = [1]
    for model in context.settings.models.values():
        if not model.enabled or model.kind != "transformer":
            continue
        manifest_path = model.bundle_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contract_path = model.bundle_dir / manifest.get(
            "dataset_contract", "dataset_contract.json"
        )
        if contract_path.exists():
            requirements.append(
                _transformer_history_requirement(str(contract_path.resolve()))
            )
    return max(requirements)


def _decision_view(row: dict[str, Any]) -> dict[str, Any]:
    predictions = _json_object(row.get("predictions"))
    compact_predictions: dict[str, dict[str, Any]] = {}
    for model, values in predictions.items():
        if not isinstance(values, dict):
            continue
        compact_predictions[str(model)] = {
            key: value
            for key, value in values.items()
            if key
            in {
                "watch_q35_bps",
                "enter_probability",
                "entry_executable_probability",
                "enter_now_q35_bps",
                "wait_best_q35_bps",
                "enter_advantage_q35_bps",
                "exit_probability",
                "exit_advantage_bps",
                "action",
                "gate_active",
                "frozen_q35_bps",
                "gate_age_sec",
                "state_replayed",
                "forced_safety_exit",
                "_latency_ms",
                "_device",
                "_error",
            }
        }
    return {
        "id": int(row.get("id", 0)),
        "ts": int(row["decision_ts"]),
        "strategy": str(row["strategy"]),
        "pair_id": str(row["pair_id"]),
        "state": str(row["state"]),
        "action": str(row["action"]),
        "reason": str(row["reason"]),
        "predictions": compact_predictions,
        "features": _json_object(row.get("features")),
    }


def _rl_gate_threshold(context: AppContext) -> float:
    for model in context.settings.models.values():
        if not model.enabled or model.kind != "rl":
            continue
        manifest_path = model.bundle_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return float(manifest.get("q35_gate_bps", 30.0))
    return 30.0


def _rl_runtime_view(
    *,
    strategy_active: bool,
    model_ready: bool,
    q35_bps: float | None,
    gate_threshold_bps: float,
    execution_valid: bool,
    outputs: dict[str, Any],
) -> dict[str, Any]:
    """Преобразовать сырые выходы RL в понятный оператору статус."""

    error = str(outputs["_error"]) if outputs.get("_error") else None
    action = _finite(outputs.get("action"))
    gate_active = bool(_finite(outputs.get("gate_active")) or 0.0)
    gate_margin = (
        q35_bps - gate_threshold_bps if q35_bps is not None else None
    )

    if not strategy_active:
        status, detail = "PAUSED", "RL paper strategy is paused"
    elif not model_ready:
        status, detail = "OFFLINE", "RL model is not ready"
    elif error:
        status, detail = "ERROR", error
    elif q35_bps is None:
        status, detail = "NO Q35", "Waiting for q35 prediction"
    elif not gate_active and q35_bps < gate_threshold_bps:
        status = "WAIT Q35"
        detail = (
            f"q35 {q35_bps:.1f} < gate {gate_threshold_bps:.1f} bps"
        )
    elif action is None:
        status, detail = "STARTING", "Gate passed; waiting for RL inference"
    else:
        status = {0: "WAIT", 1: "ENTER", 2: "HOLD", 3: "EXIT"}.get(
            int(action), "UNKNOWN"
        )
        frozen_gate = _finite(outputs.get("frozen_q35_bps"))
        detail = f"Gate active at {frozen_gate or q35_bps:.1f} bps"
        if not execution_valid and status in {"WAIT", "ENTER"}:
            status = "BLOCKED"
            detail = "Gate active, but current L2 execution is not valid"

    return {
        "rl_action": action,
        "rl_status": status,
        "rl_detail": detail,
        "rl_error": error,
        "rl_gate_active": gate_active,
        "rl_gate_threshold_bps": gate_threshold_bps,
        "rl_gate_margin_bps": gate_margin,
        "rl_frozen_q35_bps": _finite(outputs.get("frozen_q35_bps")),
        "rl_gate_age_sec": _finite(outputs.get("gate_age_sec")),
        "rl_state_replayed": bool(
            _finite(outputs.get("state_replayed")) or 0.0
        ),
    }


def monitor_overview(context: AppContext) -> dict[str, Any]:
    """Собрать модели, пары, стратегии и сделки одним компактным ответом."""

    decision_rows = context.repository.decisions(limit=1_000)
    decisions = [_decision_view(row) for row in decision_rows]
    latest_decisions: dict[tuple[str, str], dict[str, Any]] = {}
    latency_samples: dict[str, list[float]] = {}
    model_errors: dict[str, int] = {}
    seen_inferences: set[tuple[int, str, str]] = set()
    for decision in decisions:
        latest_decisions.setdefault(
            (decision["pair_id"], decision["strategy"]),
            decision,
        )
        for model, values in decision["predictions"].items():
            key = (decision["ts"], decision["pair_id"], model)
            if key in seen_inferences:
                continue
            seen_inferences.add(key)
            latency = _finite(values.get("_latency_ms"))
            if latency is not None:
                latency_samples.setdefault(model, []).append(latency)
            if values.get("_error"):
                model_errors[model] = model_errors.get(model, 0) + 1

    model_statuses = context.registry.statuses()
    model_ready = {
        str(status["name"]): status["state"] == "ready"
        for status in model_statuses
    }
    models = []
    for status in model_statuses:
        samples = latency_samples.get(str(status["name"]), [])
        models.append(
            {
                **status,
                "latency_mean_ms": fmean(samples) if samples else None,
                "latency_p50_ms": _percentile(samples, 0.50),
                "latency_p95_ms": _percentile(samples, 0.95),
                "recent_inferences": len(samples),
                "recent_errors": model_errors.get(str(status["name"]), 0),
            }
        )

    strategy_stats = {
        str(item["strategy"]): item for item in context.repository.stats()
    }
    strategies = []
    for strategy in context.paper.strategies():
        name = str(strategy["name"])
        strategies.append(
            {
                **strategy,
                "models": sorted(context.paper.required_models(name)),
                "stats": strategy_stats.get(
                    name,
                    {
                        "strategy": name,
                        "trades": 0,
                        "total_net_pnl_bps": 0.0,
                        "mean_trade_pnl_bps": 0.0,
                        "median_trade_pnl_bps": 0.0,
                        "win_rate": 0.0,
                        "max_drawdown_bps": 0.0,
                        "mean_hold_seconds": 0.0,
                    },
                ),
            }
        )

    required_history = _required_history(context)
    rl_gate_threshold = _rl_gate_threshold(context)
    pairs = []
    for pair in context.store.list_pairs():
        history = context.features.history(pair.pair_id)
        current = history[-1] if history else None
        pair_strategies = []
        merged_predictions: dict[str, dict[str, Any]] = {}
        for strategy in strategies:
            name = str(strategy["name"])
            decision = latest_decisions.get((pair.pair_id, name))
            position = context.paper.position(name, pair.pair_id)
            unrealized = None
            if position is not None and current is not None:
                state = context.paper.position_state(name, current)
                unrealized = float(state[1]) * 100.0
            if decision is not None:
                for model, values in decision["predictions"].items():
                    merged_predictions.setdefault(model, values)
            pair_strategies.append(
                {
                    "name": name,
                    "active": bool(strategy["active"]),
                    "state": (
                        "open"
                        if position is not None
                        else (
                            str(decision["state"])
                            if decision is not None
                            else "waiting"
                        )
                    ),
                    "action": (
                        str(decision["action"]) if decision is not None else "-"
                    ),
                    "reason": (
                        str(decision["reason"]) if decision is not None else ""
                    ),
                    "unrealized_pnl_bps": unrealized,
                    "opened_ts": (
                        int(position.opened_ts) if position is not None else None
                    ),
                }
            )

        q35 = merged_predictions.get("q35_perp", {})
        transformer = merged_predictions.get("transformer", {})
        rl = merged_predictions.get("rl_agent", {})
        q35_watch = _finite(q35.get("watch_q35_bps"))
        execution_valid = bool(
            current is not None
            and (_finite(current.features.get("execution_valid_numeric")) or 0.0)
            > 0.5
        )
        rl_strategy_active = any(
            bool(strategy["active"])
            and "rl_agent" in strategy["models"]
            and (
                strategy.get("pair_types") is None
                or pair.pair_type in strategy["pair_types"]
            )
            for strategy in strategies
        )
        rl_view = _rl_runtime_view(
            strategy_active=rl_strategy_active,
            model_ready=model_ready.get("rl_agent", False),
            q35_bps=q35_watch,
            gate_threshold_bps=rl_gate_threshold,
            execution_valid=execution_valid,
            outputs=rl,
        )
        history_steps = len(history)
        pairs.append(
            {
                "pair_id": pair.pair_id,
                "base_ticker": pair.base_ticker,
                "pair_type": pair.pair_type,
                "direction_code": pair.direction_code,
                "direction": (
                    f"LONG {pair.leg1.exchange} / SHORT {pair.leg2.exchange}"
                    if pair.direction_code == 0
                    else f"LONG {pair.leg2.exchange} / SHORT {pair.leg1.exchange}"
                ),
                "leg1_exchange": pair.leg1.exchange,
                "leg2_exchange": pair.leg2.exchange,
                "decision_ts": (
                    int(current.decision_ts) if current is not None else None
                ),
                "edge_bps": (
                    _finite(current.features.get("entry_edge_top1_bps"))
                    if current is not None
                    else None
                ),
                "fill_share": (
                    _finite(current.features.get("current_entry_fill_share"))
                    if current is not None
                    else None
                ),
                "pair_skew_ms": (
                    _finite(current.features.get("pair_skew_sec")) * 1_000.0
                    if current is not None
                    and _finite(current.features.get("pair_skew_sec"))
                    is not None
                    else None
                ),
                "history_steps": history_steps,
                "history_required": required_history,
                "warmup_progress": min(
                    1.0, history_steps / max(1, required_history)
                ),
                "q35_watch_bps": q35_watch,
                "transformer_enter_probability": _finite(
                    transformer.get("enter_probability")
                ),
                "transformer_exit_probability": _finite(
                    transformer.get("exit_probability")
                ),
                **rl_view,
                "strategies": pair_strategies,
            }
        )
    pairs.sort(
        key=lambda item: (
            str(item["base_ticker"]).casefold(),
            min(item["leg1_exchange"], item["leg2_exchange"]).casefold(),
            max(item["leg1_exchange"], item["leg2_exchange"]).casefold(),
            int(item["direction_code"]),
            str(item["pair_id"]).casefold(),
        )
    )

    events = [
        _decision_view(row) for row in context.repository.events(limit=80)
    ]
    trades = context.repository.trades(limit=100)
    repository_status = context.repository.status()
    total_pnl = sum(
        float(item["stats"]["total_net_pnl_bps"]) for item in strategies
    )
    return {
        "server_ts": max(
            (
                int(pair["decision_ts"])
                for pair in pairs
                if pair["decision_ts"] is not None
            ),
            default=0,
        ),
        "summary": {
            "monitored_pairs": len(pairs),
            "active_strategies": sum(
                bool(item["active"]) for item in strategies
            ),
            "open_positions": len(context.paper.positions()),
            "trades_in_memory": int(repository_status["trades"]),
            "total_net_pnl_bps": total_pnl,
        },
        "models": models,
        "strategies": strategies,
        "pairs": pairs,
        "events": events,
        "trades": trades[:30],
        "paper_store": repository_status,
    }


def monitor_pair_series(
    context: AppContext,
    pair_id: str,
    limit: int,
) -> dict[str, Any] | None:
    """Подготовить временной ряд edge и маркеры сделок выбранной пары."""

    pair = context.store.get_pair(pair_id)
    if pair is None:
        return None
    history = context.features.history(pair_id)[-int(limit) :]
    points = [
        {
            "ts": int(item.grid_ts),
            "edge_bps": _finite(
                item.features.get("entry_edge_top1_bps")
            ),
        }
        for item in history
    ]
    markers = []
    for trade in context.repository.trades(limit=200, pair_id=pair_id):
        markers.extend(
            (
                {
                    "ts": int(trade["entry_fill_ts"]),
                    "kind": "entry",
                    "strategy": str(trade["strategy"]),
                    "pnl_bps": None,
                },
                {
                    "ts": int(trade["closed_ts"]),
                    "kind": "exit",
                    "strategy": str(trade["strategy"]),
                    "pnl_bps": float(trade["net_pnl_bps"]),
                },
            )
        )
    for position in context.paper.positions():
        if position["pair_id"] != pair_id:
            continue
        markers.append(
            {
                "ts": int(position["entry_fill_ts"]),
                "kind": "entry_open",
                "strategy": str(position["strategy"]),
                "pnl_bps": None,
            }
        )
    markers.sort(key=lambda item: item["ts"])
    return {
        "pair_id": pair_id,
        "base_ticker": pair.base_ticker,
        "direction_code": pair.direction_code,
        "points": points,
        "markers": markers,
    }
