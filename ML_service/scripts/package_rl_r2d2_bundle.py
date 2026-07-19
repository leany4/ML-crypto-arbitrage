"""Экспорт минимального recurrent Double-DQN bundle для инференса."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import torch


Q35_RUNTIME_FEATURES = {
    "q35_causal_bps",
    "q35_gate_margin_bps",
    "time_from_gate_sec",
    "agent_active",
}


parser = argparse.ArgumentParser(
    description="Упаковать RL checkpoint без optimizer и replay buffer"
)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--scaler", type=Path, required=True)
parser.add_argument("--evaluation-summary", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--version", required=True)
parser.add_argument("--decision-ms", type=int, default=100)
parser.add_argument("--burn-in-steps", type=int, default=1200)
parser.add_argument("--min-hold-steps", type=int, default=10)
parser.add_argument("--cooldown-steps", type=int, default=100)
parser.add_argument("--max-session-steps", type=int, default=18_000)
parser.add_argument("--max-state-gap-ms", type=int, default=2_000)
args = parser.parse_args()

if not args.checkpoint.exists():
    raise FileNotFoundError(args.checkpoint)
if not args.scaler.exists():
    raise FileNotFoundError(args.scaler)
if args.decision_ms != 100:
    raise ValueError("The current recurrent agent was trained at 100 ms")
runtime_contract = (
    (args.burn_in_steps, 1200, "burn-in-steps"),
    (args.min_hold_steps, 10, "min-hold-steps"),
    (args.cooldown_steps, 100, "cooldown-steps"),
    (args.max_session_steps, 18_000, "max-session-steps"),
)
mismatched = [
    f"--{name}={actual}, expected={expected}"
    for actual, expected, name in runtime_contract
    if actual != expected
]
if mismatched:
    raise ValueError(
        "Arguments differ from the trained contract: " + "; ".join(mismatched)
    )

checkpoint = torch.load(
    args.checkpoint, map_location="cpu", weights_only=False
)
required = {
    "online_state_dict",
    "feature_cols",
    "obs_dim",
    "hidden_size",
    "q35_gate_bps",
}
missing = sorted(required - checkpoint.keys())
if missing:
    raise RuntimeError(f"Checkpoint misses: {', '.join(missing)}")

feature_columns = [str(value) for value in checkpoint["feature_cols"]]
if int(checkpoint["obs_dim"]) != len(feature_columns) + 8:
    raise RuntimeError("Checkpoint obs_dim is not feature_count + 8")
missing_runtime = sorted(Q35_RUNTIME_FEATURES - set(feature_columns))
if missing_runtime:
    raise RuntimeError(
        f"Checkpoint misses q35 runtime features: {', '.join(missing_runtime)}"
    )
scaler = joblib.load(args.scaler)
if [str(value) for value in scaler["feature_cols"]] != feature_columns:
    raise RuntimeError("Checkpoint and scaler feature order differ")
mean = np.asarray(scaler["mean"], dtype=np.float32)
std = np.asarray(scaler["std"], dtype=np.float32)
expected_shape = (len(feature_columns),)
if (
    mean.shape != expected_shape
    or std.shape != expected_shape
    or not np.isfinite(mean).all()
    or not np.isfinite(std).all()
    or (std <= 0).any()
):
    raise RuntimeError("Invalid feature scaler")

args.output.mkdir(parents=True, exist_ok=True)
deployment_checkpoint = {
    "online_state_dict": checkpoint["online_state_dict"],
    "feature_cols": feature_columns,
    "obs_dim": int(checkpoint["obs_dim"]),
    "hidden_size": int(checkpoint["hidden_size"]),
    "q35_gate_bps": float(checkpoint["q35_gate_bps"]),
    "source_episode": int(checkpoint.get("episode", 0)),
    "best_validation_score": float(
        checkpoint.get("best_validation_score", float("nan"))
    ),
}
torch.save(deployment_checkpoint, args.output / "lstm_r2d2.pt")
shutil.copy2(args.scaler, args.output / "feature_scaler.joblib")
if args.evaluation_summary is not None:
    if not args.evaluation_summary.exists():
        raise FileNotFoundError(args.evaluation_summary)
    shutil.copy2(
        args.evaluation_summary,
        args.output / "evaluation_summary.json",
    )

manifest = {
    "name": "rl_agent",
    "kind": "rl",
    "version": args.version,
    "loader": "torch_r2d2",
    "algorithm": "recurrent_double_dqn",
    "artifact": "lstm_r2d2.pt",
    "scaler": "feature_scaler.joblib",
    "pair_types": [
        "perp_perp_cross_exchange",
        "spot_perp_same_exchange",
    ],
    "decision_ms": args.decision_ms,
    "execution_delay_ms": 100,
    "burn_in_steps": args.burn_in_steps,
    "min_hold_steps": args.min_hold_steps,
    "cooldown_steps": args.cooldown_steps,
    "max_session_steps": args.max_session_steps,
    "max_state_gap_ms": args.max_state_gap_ms,
    "q35_gate_bps": float(checkpoint["q35_gate_bps"]),
    "actions": {
        "0": "WAIT",
        "1": "ENTER",
        "2": "HOLD",
        "3": "EXIT",
    },
    "manual_state": [
        "position_open",
        "current_net_pnl_div_100",
        "mfe_div_100",
        "mae_div_100",
        "hold_steps_div_3000",
        "session_realized_bps_div_100",
        "cooldown_remaining_div_100_steps",
        "session_trade_count_div_10",
    ],
}
(args.output / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("READY:", args.output)
