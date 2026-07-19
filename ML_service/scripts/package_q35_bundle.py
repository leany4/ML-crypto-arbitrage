"""Проверка и упаковка legacy q35-модели в serving bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


parser = argparse.ArgumentParser(
    description="Упаковать q35 joblib и создать явный manifest"
)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--name", default="q35_perp")
parser.add_argument(
    "--pair-type",
    choices=["perp_perp_cross_exchange", "spot_perp_same_exchange"],
    default="perp_perp_cross_exchange",
)
parser.add_argument("--version", required=True)
parser.add_argument("--sklearn-version", default="1.7.2")
args = parser.parse_args()

if not args.source.exists():
    raise FileNotFoundError(args.source)
args.output.mkdir(parents=True, exist_ok=True)
artifact_path = args.output / "model.joblib"
shutil.copy2(args.source, artifact_path)
artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
manifest = {
    "name": args.name,
    "kind": "q35",
    "version": args.version,
    "artifact": "model.joblib",
    "pair_types": [args.pair_type],
    "sklearn_version": args.sklearn_version,
    "output": "watch_q35_bps",
    "sha256": artifact_sha256,
}
(args.output / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("READY:", args.output)
