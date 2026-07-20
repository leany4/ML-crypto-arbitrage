"""Serving-адаптер Transformer: история, scaling и интерпретация голов."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml_service.device import resolve_device
from ml_service.predictors.base import (
    Prediction,
    PredictionContext,
    Predictor,
    load_manifest,
)
from ml_service.predictors.transformer.model import (
    GatedMultiscaleDecisionTransformer,
)


class TransformerPredictor(Predictor):
    """Собирает причинный multiscale batch и выполняет Torch-инференс."""

    kind = "transformer"

    def __init__(self, name: str, bundle_dir: Path, requested_device: str):
        super().__init__(name, bundle_dir, requested_device)
        self.model: GatedMultiscaleDecisionTransformer | None = None
        self.contract: dict[str, Any] = {}
        self.category_config: dict[str, Any] = {}
        self.model_config: dict[str, Any] = {}
        self.feature_mean = np.empty(0, dtype=np.float32)
        self.feature_std = np.empty(0, dtype=np.float32)
        self._history_cache: OrderedDict[
            str, tuple[np.ndarray, np.ndarray]
        ] = OrderedDict()
        self._history_cache_lock = threading.RLock()
        self._history_cache_limit = 128

    def load(self) -> None:
        """Проверить bundle-контракт, загрузить scaler, конфиг и checkpoint."""

        manifest = load_manifest(self.bundle_dir)
        artifact = self.bundle_dir / manifest.get(
            "artifact", "multiscale_transformer_best.pt"
        )
        scaler_path = self.bundle_dir / manifest.get("scaler", "feature_scaler.npz")
        contract_path = self.bundle_dir / manifest.get(
            "dataset_contract", "dataset_contract.json"
        )
        category_path = self.bundle_dir / manifest.get(
            "category_config", "category_config.json"
        )
        config_path = self.bundle_dir / manifest.get("model_config", "model_config.json")
        for path in (artifact, scaler_path, contract_path, category_path, config_path):
            if not path.exists():
                raise FileNotFoundError(path)

        self.contract = json.loads(contract_path.read_text(encoding="utf-8"))
        self.category_config = json.loads(category_path.read_text(encoding="utf-8"))
        self.model_config = json.loads(config_path.read_text(encoding="utf-8"))
        scaler = np.load(scaler_path)
        self.feature_mean = scaler["mean"].astype(np.float32)
        self.feature_std = scaler["std"].astype(np.float32)

        feature_columns = self.contract["feature_columns"]
        l2_columns = self.contract["l2_feature_columns"]
        ohlcv_columns = self.contract["ohlcv_feature_columns"]
        if not (
            len(feature_columns)
            == len(self.feature_mean)
            == len(self.feature_std)
        ):
            raise RuntimeError("Transformer scaler and feature contract disagree")
        if not np.isfinite(self.feature_mean).all():
            raise RuntimeError("Transformer scaler mean contains non-finite values")
        if not np.isfinite(self.feature_std).all() or (self.feature_std <= 0).any():
            raise RuntimeError("Transformer scaler std must be finite and positive")

        device_choice = resolve_device(self.requested_device)
        self.device = device_choice.resolved
        self.detail = device_choice.fallback_reason
        exchange_to_id = self.category_config["exchange_to_id"]
        pair_hash_buckets = int(self.category_config["pair_hash_buckets"])
        self.model = GatedMultiscaleDecisionTransformer(
            l2_input_dim=len(l2_columns),
            ohlcv_input_dim=2 * len(ohlcv_columns),
            local_steps=int(self.contract["local_history_steps"]),
            long_tokens=int(self.contract["long_history_tokens"]),
            exchange_count=len(exchange_to_id) + 1,
            pair_hash_buckets=pair_hash_buckets,
            d_model=int(self.model_config.get("d_model", 96)),
            heads=int(self.model_config.get("heads", 4)),
            layers=int(self.model_config.get("layers_per_branch", 2)),
            ff_dim=int(self.model_config.get("ff_dim", 256)),
            dropout=float(self.model_config.get("dropout", 0.10)),
        ).to(self.device)
        try:
            checkpoint = torch.load(
                artifact, map_location=self.device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(artifact, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()
        self.version = str(manifest.get("version", checkpoint.get("epoch", "unknown")))
        self.loaded_at = time.time()

    def _from_history(self, context: PredictionContext) -> dict[str, Any]:
        """Преобразовать online-историю пары в локальную и длинную ветки."""

        feature_columns = list(self.contract["feature_columns"])
        l2_columns = list(self.contract["l2_feature_columns"])
        ohlcv_columns = list(self.contract["ohlcv_feature_columns"])
        local_steps = int(self.contract["local_history_steps"])
        long_tokens = int(self.contract["long_history_tokens"])
        stride = int(self.contract["long_history_stride_steps"])
        needed = max(local_steps, (long_tokens - 1) * stride + 1)
        if len(context.history) < needed:
            raise ValueError(
                f"Transformer history is not ready: {len(context.history)}/{needed}"
            )

        rows = context.history[-needed:]
        cache_key = context.pair_id or str(rows[-1].get("pair_id", ""))
        history_timestamps = (
            np.asarray(context.history_timestamps[-needed:], dtype=np.int64)
            if len(context.history_timestamps) >= needed
            else None
        )
        cache_hit = False
        if cache_key and history_timestamps is not None:
            with self._history_cache_lock:
                cached = self._history_cache.get(cache_key)
                if cached is not None:
                    cached_timestamps, _ = cached
                    cache_hit = np.array_equal(
                        cached_timestamps, history_timestamps
                    ) or (
                        len(cached_timestamps) == len(history_timestamps)
                        and np.array_equal(
                            cached_timestamps[1:], history_timestamps[:-1]
                        )
                    )

        # Cached rows were validated during their first full conversion.
        rows_to_validate = rows[-1:] if cache_hit else rows
        missing = sorted(
            {
                column
                for row in rows_to_validate
                for column in feature_columns
                if column not in row
            }
        )
        if missing:
            raise ValueError(
                f"Transformer missing {len(missing)} contract features: "
                + ", ".join(missing[:8])
            )
        feature_index = {name: index for index, name in enumerate(feature_columns)}
        l2_indices = np.asarray(
            [feature_index[name] for name in l2_columns], dtype=np.int64
        )
        ohlcv_indices = np.asarray(
            [feature_index[name] for name in ohlcv_columns], dtype=np.int64
        )
        clip = float(self.model_config.get("feature_clip", 10.0))
        l2_mean = self.feature_mean[l2_indices]
        l2_std = self.feature_std[l2_indices]

        def scaled_row(row: dict[str, Any]) -> np.ndarray:
            values = np.asarray(
                [row[column] for column in l2_columns], dtype=np.float32
            )
            values = np.nan_to_num(
                values, nan=0.0, posinf=0.0, neginf=0.0
            )
            return np.clip((values - l2_mean) / l2_std, -clip, clip)

        l2_matrix: np.ndarray | None = None
        if cache_key and history_timestamps is not None:
            with self._history_cache_lock:
                cached = self._history_cache.get(cache_key)
                if cached is not None:
                    cached_timestamps, cached_matrix = cached
                    if np.array_equal(cached_timestamps, history_timestamps):
                        cached_matrix[-1] = scaled_row(rows[-1])
                        l2_matrix = cached_matrix
                    elif (
                        len(cached_timestamps) == len(history_timestamps)
                        and np.array_equal(
                            cached_timestamps[1:], history_timestamps[:-1]
                        )
                    ):
                        cached_matrix[:-1] = cached_matrix[1:].copy()
                        cached_matrix[-1] = scaled_row(rows[-1])
                        cached_timestamps[:-1] = cached_timestamps[1:]
                        cached_timestamps[-1] = history_timestamps[-1]
                        l2_matrix = cached_matrix
                    if l2_matrix is not None:
                        self._history_cache.move_to_end(cache_key)

        if l2_matrix is None:
            l2_matrix = np.asarray(
                [[row[column] for column in l2_columns] for row in rows],
                dtype=np.float32,
            )
            l2_matrix = np.nan_to_num(
                l2_matrix, nan=0.0, posinf=0.0, neginf=0.0
            )
            l2_matrix = np.clip(
                (l2_matrix - l2_mean) / l2_std,
                -clip,
                clip,
            )
            if cache_key and history_timestamps is not None:
                with self._history_cache_lock:
                    self._history_cache[cache_key] = (
                        history_timestamps.copy(),
                        l2_matrix,
                    )
                    self._history_cache.move_to_end(cache_key)
                    while len(self._history_cache) > self._history_cache_limit:
                        self._history_cache.popitem(last=False)
        long_indices = np.arange(
            len(l2_matrix) - 1 - (long_tokens - 1) * stride,
            len(l2_matrix),
            stride,
            dtype=np.int64,
        )
        current = rows[-1]
        ohlcv_matrix = np.asarray(
            [
                [rows[int(long_indices[0])][column] for column in ohlcv_columns],
                [current[column] for column in ohlcv_columns],
            ],
            dtype=np.float32,
        )
        ohlcv_matrix = np.nan_to_num(
            ohlcv_matrix, nan=0.0, posinf=0.0, neginf=0.0
        )
        ohlcv_matrix = np.clip(
            (ohlcv_matrix - self.feature_mean[ohlcv_indices])
            / self.feature_std[ohlcv_indices],
            -clip,
            clip,
        )
        exchange_to_id = self.category_config["exchange_to_id"]
        pair_type_to_id = self.category_config["pair_type_to_id"]
        pair_buckets = int(self.category_config["pair_hash_buckets"])
        pair_id = str(current["pair_id"])
        pair_hash = (
            int(hashlib.sha1(pair_id.encode("utf-8")).hexdigest()[:8], 16)
            % pair_buckets
        )
        ohlcv_current = ohlcv_matrix[1]
        ohlcv_change = ohlcv_current - ohlcv_matrix[0]
        position_state = context.position_state or [0.0] * 6
        entry_snapshot = (
            [0.0] * 6
            if position_state[0] > 0
            else (context.entry_snapshot or self._entry_snapshot(current))
        )
        return {
            "local": l2_matrix[-local_steps:].copy(),
            "long": l2_matrix[long_indices],
            "ohlcv_state": np.concatenate([ohlcv_current, ohlcv_change]),
            "position_state": position_state,
            "entry_snapshot": entry_snapshot,
            "pair_type_id": pair_type_to_id.get(str(current["pair_type"]), 0),
            "direction_id": int(current.get("direction_code", 0)) + 1,
            "leg1_exchange_id": exchange_to_id.get(
                str(current["leg1_exchange"]), 0
            ),
            "leg2_exchange_id": exchange_to_id.get(
                str(current["leg2_exchange"]), 0
            ),
            "pair_hash_id": pair_hash,
        }

    def warmup(self) -> None:
        """Прогреть CUDA kernels на настоящих serving-размерах."""

        if self.model is None:
            raise RuntimeError(f"{self.name} is not loaded")
        l2_dim = len(self.contract["l2_feature_columns"])
        ohlcv_dim = 2 * len(self.contract["ohlcv_feature_columns"])
        values = {
            "local": np.zeros(
                (int(self.contract["local_history_steps"]), l2_dim),
                dtype=np.float32,
            ),
            "long": np.zeros(
                (int(self.contract["long_history_tokens"]), l2_dim),
                dtype=np.float32,
            ),
            "ohlcv_state": np.zeros(ohlcv_dim, dtype=np.float32),
            "position_state": np.zeros(6, dtype=np.float32),
            "entry_snapshot": np.zeros(6, dtype=np.float32),
            "pair_type_id": 0,
            "direction_id": 0,
            "leg1_exchange_id": 0,
            "leg2_exchange_id": 0,
            "pair_hash_id": 0,
        }
        context = PredictionContext(features={}, transformer_input=values)
        for _ in range(3):
            self.predict(context)
        if self.device == "cuda":
            torch.cuda.synchronize()

    @staticmethod
    def _entry_snapshot(features: dict[str, Any]) -> list[float]:
        values = [
            float(features.get("current_entry_executable", 0.0)),
            float(features.get("current_entry_fill_share", 0.0)),
            float(features.get("current_open_gross_edge_bps", 0.0)) / 100.0,
            float(features.get("current_open_edge_after_entry_fee_bps", 0.0)) / 100.0,
            float(features.get("current_entry_slippage_bps", 0.0)) / 100.0,
            float(features.get("current_instant_roundtrip_pnl_bps", 0.0)) / 100.0,
        ]
        return np.clip(values, -5.0, 5.0).tolist()

    def _tensor_batch(self, values: dict[str, Any]) -> dict[str, torch.Tensor]:
        floats = ("local", "long", "ohlcv_state", "position_state", "entry_snapshot")
        ids = (
            "pair_type_id",
            "direction_id",
            "leg1_exchange_id",
            "leg2_exchange_id",
            "pair_hash_id",
        )
        batch: dict[str, torch.Tensor] = {}
        for name in floats:
            array = np.asarray(values[name], dtype=np.float32)
            if name in {"local", "long"}:
                array = array[None, :, :]
            else:
                array = array[None, :]
            batch[name] = torch.as_tensor(array, device=self.device)
        for name in ids:
            batch[name] = torch.as_tensor(
                [int(values[name])], dtype=torch.long, device=self.device
            )
        return batch

    def predict(
        self, context: PredictionContext, heads: list[str] | None = None
    ) -> Prediction:
        """Выполнить инференс и перевести logits/quantiles в API-единицы."""

        if self.model is None:
            raise RuntimeError(f"{self.name} is not loaded")
        started = time.perf_counter()
        values = context.transformer_input or self._from_history(context)
        batch = self._tensor_batch(values)
        with torch.inference_mode():
            raw = self.model(batch)
        scale = float(self.model_config.get("target_scale_bps", 100.0))

        def scalar(name: str) -> float:
            return float(raw[name][0].detach().cpu())

        def probability(name: str) -> float:
            return float(torch.sigmoid(raw[name][0]).detach().cpu())

        enter_now = raw["enter_now_quantiles"][0].detach().cpu().numpy() * scale
        wait_best = raw["wait_best_quantiles"][0].detach().cpu().numpy() * scale
        advantage = (
            raw["enter_advantage_quantiles"][0].detach().cpu().numpy() * scale
        )
        outputs = {
            "watch_q35_bps": scalar("watch") * scale,
            "enter_probability": probability("enter"),
            "entry_executable_probability": probability("entry_executable"),
            "enter_now_q20_bps": float(enter_now[0]),
            "enter_now_q35_bps": float(enter_now[1]),
            "enter_now_q50_bps": float(enter_now[2]),
            "wait_executable_probability": probability("wait_executable"),
            "wait_best_q20_bps": float(wait_best[0]),
            "wait_best_q35_bps": float(wait_best[1]),
            "wait_best_q50_bps": float(wait_best[2]),
            "enter_advantage_q20_bps": float(advantage[0]),
            "enter_advantage_q35_bps": float(advantage[1]),
            "enter_advantage_q50_bps": float(advantage[2]),
            "exit_probability": probability("exit"),
            "exit_advantage_bps": scalar("exit_advantage") * scale,
        }
        return self._prediction(outputs, started, heads)

    def close(self) -> None:
        """Освободить модель и очистить неиспользуемый CUDA cache."""

        self.model = None
        with self._history_cache_lock:
            self._history_cache.clear()
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
