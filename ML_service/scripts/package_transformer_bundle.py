"""Проверка checkpoint, scaler и контракта Transformer перед serving."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch


parser = argparse.ArgumentParser(
    description="Упаковать проверенный Transformer bundle"
)
parser.add_argument("--source-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--version", required=True)
parser.add_argument(
    "--checkpoint", default="multiscale_transformer_best.pt"
)
args = parser.parse_args()

required = [
    args.checkpoint,
    "feature_scaler.npz",
    "category_config.json",
    "model_config.json",
]
missing = [name for name in required if not (args.source_dir / name).exists()]
if missing:
    raise FileNotFoundError(f"Missing Transformer artifacts: {missing}")

checkpoint_path = args.source_dir / args.checkpoint
checkpoint = torch.load(
    checkpoint_path, map_location="cpu", weights_only=False
)
contract = checkpoint.get("contract")
if not isinstance(contract, dict):
    raise RuntimeError("Transformer checkpoint has no embedded dataset contract")

feature_columns = [str(value) for value in checkpoint.get("feature_columns", [])]
contract_columns = [
    str(value) for value in contract.get("feature_columns", [])
]
if not feature_columns or feature_columns != contract_columns:
    raise RuntimeError("Checkpoint and dataset contract feature order differ")

external_contract_path = args.source_dir / "dataset_contract.json"
if external_contract_path.exists():
    external_contract = json.loads(external_contract_path.read_text())
    if external_contract != contract:
        raise RuntimeError("External and embedded dataset contracts differ")

category_config = json.loads(
    (args.source_dir / "category_config.json").read_text()
)
if checkpoint.get("category_config") != category_config:
    raise RuntimeError("Checkpoint and category config differ")

model_config = json.loads(
    (args.source_dir / "model_config.json").read_text()
)
if int(model_config["input_dim"]) != len(feature_columns):
    raise RuntimeError("model_config input_dim differs from feature contract")
expected_feature_count = (
    int(model_config["l2_temporal_feature_count"])
    + int(model_config["ohlcv_feature_count"])
)
if expected_feature_count != len(feature_columns):
    raise RuntimeError("L2 + OHLCV feature counts differ from input_dim")

with np.load(args.source_dir / "feature_scaler.npz") as scaler:
    mean = np.asarray(scaler["mean"])
    std = np.asarray(scaler["std"])
expected_shape = (len(feature_columns),)
if (
    mean.shape != expected_shape
    or std.shape != expected_shape
    or not np.isfinite(mean).all()
    or not np.isfinite(std).all()
    or (std <= 0).any()
):
    raise RuntimeError("Invalid Transformer feature scaler")

args.output.mkdir(parents=True, exist_ok=True)
for name in required:
    shutil.copy2(args.source_dir / name, args.output / name)
(args.output / "dataset_contract.json").write_text(
    json.dumps(contract, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

manifest = {
    "name": "transformer",
    "kind": "transformer",
    "version": args.version,
    "artifact": args.checkpoint,
    "scaler": "feature_scaler.npz",
    "category_config": "category_config.json",
    "model_config": "model_config.json",
    "dataset_contract": "dataset_contract.json",
    "pair_types": contract.get(
        "pair_types", ["perp_perp_cross_exchange"]
    ),
    "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
    "feature_count": len(feature_columns),
    "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
}
(args.output / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("READY:", args.output)
