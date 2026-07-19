"""Stateful serving recurrent Double-DQN с внешним q35-gate."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ml_service.device import resolve_device
from ml_service.predictors.base import (
    Prediction,
    PredictionContext,
    Predictor,
    load_manifest,
)


ACTION_NAMES = ("wait", "enter", "hold", "exit")
ACTION_MANIFEST = {
    "0": "WAIT",
    "1": "ENTER",
    "2": "HOLD",
    "3": "EXIT",
}
Q35_RUNTIME_FEATURES = {
    "q35_causal_bps",
    "q35_gate_margin_bps",
    "time_from_gate_sec",
    "agent_active",
}
MANUAL_STATE_SIZE = 8


@dataclass
class RecurrentPairState:
    """Скрытое LSTM-состояние одной стратегии, пары и направления."""

    hidden: tuple[Any, Any] | None = None
    gate_active: bool = False
    frozen_q35_bps: float = 0.0
    gate_ts: int | None = None
    last_grid_ts: int | None = None
    last_outputs: dict[str, float | int] | None = None


class RLPredictor(Predictor):
    """Online-адаптер q35-gated recurrent Double-DQN."""

    kind = "rl"

    def __init__(self, name: str, bundle_dir: Path, requested_device: str):
        super().__init__(name, bundle_dir, requested_device)
        self.model = None
        self.torch = None
        self.feature_columns: list[str] = []
        self.feature_mean = np.empty(0, dtype=np.float32)
        self.feature_std = np.empty(0, dtype=np.float32)
        self.obs_dim = 0
        self.hidden_size = 0
        self.q35_gate_bps = 30.0
        self.decision_ms = 100
        self.burn_in_steps = 1200
        self.min_hold_steps = 10
        self.cooldown_steps = 100
        self.max_session_steps = 18_000
        self.max_state_gap_ms = 2_000
        self._states: dict[str, RecurrentPairState] = {}
        self._lock = threading.RLock()

    def load(self) -> None:
        """Проверить RL-контракт и загрузить online network со scaler."""

        manifest = load_manifest(self.bundle_dir)
        if str(manifest.get("loader", "torch_r2d2")) != "torch_r2d2":
            raise ValueError(f"{self.name} requires loader=torch_r2d2")
        if int(manifest.get("decision_ms", 100)) != 100:
            raise ValueError("The deployed recurrent agent must use 100 ms decisions")
        if int(manifest.get("execution_delay_ms", 100)) != 100:
            raise ValueError("The deployed recurrent agent requires 100 ms fills")
        actions = {
            str(key): str(value).upper()
            for key, value in manifest.get("actions", ACTION_MANIFEST).items()
        }
        if actions != ACTION_MANIFEST:
            raise ValueError("RL action contract must be WAIT/ENTER/HOLD/EXIT")

        try:
            import torch
            from torch import nn
        except ImportError as error:
            raise RuntimeError(
                "Install the project with the 'torch' extra to load the RL agent"
            ) from error

        artifact = self.bundle_dir / manifest.get("artifact", "lstm_r2d2.pt")
        scaler_path = self.bundle_dir / manifest.get(
            "scaler", "feature_scaler.joblib"
        )
        if not artifact.exists():
            raise FileNotFoundError(artifact)
        if not scaler_path.exists():
            raise FileNotFoundError(scaler_path)

        device_choice = resolve_device(self.requested_device)
        self.device = device_choice.resolved
        self.detail = device_choice.fallback_reason
        checkpoint = torch.load(
            artifact, map_location=self.device, weights_only=False
        )
        state_dict = checkpoint.get(
            "online_state_dict", checkpoint.get("state_dict")
        )
        if state_dict is None:
            raise RuntimeError("RL checkpoint has no online_state_dict")

        self.feature_columns = [
            str(value) for value in checkpoint["feature_cols"]
        ]
        self.obs_dim = int(checkpoint["obs_dim"])
        self.hidden_size = int(checkpoint["hidden_size"])
        self.q35_gate_bps = float(
            manifest.get(
                "q35_gate_bps", checkpoint.get("q35_gate_bps", 30.0)
            )
        )
        self.decision_ms = int(manifest.get("decision_ms", 100))
        self.burn_in_steps = int(manifest.get("burn_in_steps", 1200))
        self.min_hold_steps = int(manifest.get("min_hold_steps", 10))
        self.cooldown_steps = int(manifest.get("cooldown_steps", 100))
        self.max_session_steps = int(
            manifest.get("max_session_steps", 18_000)
        )
        self.max_state_gap_ms = int(
            manifest.get("max_state_gap_ms", 2_000)
        )
        runtime_contract = (
            (self.burn_in_steps, 1200, "burn_in_steps"),
            (self.min_hold_steps, 10, "min_hold_steps"),
            (self.cooldown_steps, 100, "cooldown_steps"),
            (self.max_session_steps, 18_000, "max_session_steps"),
        )
        mismatched = [
            f"{name}={actual}, expected={expected}"
            for actual, expected, name in runtime_contract
            if actual != expected
        ]
        if mismatched:
            raise RuntimeError(
                "RL manifest differs from the training contract: "
                + "; ".join(mismatched)
            )
        if self.obs_dim != len(self.feature_columns) + MANUAL_STATE_SIZE:
            raise RuntimeError(
                f"obs_dim={self.obs_dim} does not match "
                f"{len(self.feature_columns)} features + {MANUAL_STATE_SIZE} state"
            )
        if not Q35_RUNTIME_FEATURES.issubset(self.feature_columns):
            missing = sorted(Q35_RUNTIME_FEATURES - set(self.feature_columns))
            raise RuntimeError(f"RL feature contract misses: {', '.join(missing)}")

        scaler = joblib.load(scaler_path)
        scaler_columns = [str(value) for value in scaler["feature_cols"]]
        if scaler_columns != self.feature_columns:
            raise RuntimeError("RL checkpoint and scaler feature order differ")
        self.feature_mean = np.asarray(scaler["mean"], dtype=np.float32)
        self.feature_std = np.asarray(scaler["std"], dtype=np.float32)
        if (
            self.feature_mean.shape != (len(self.feature_columns),)
            or self.feature_std.shape != (len(self.feature_columns),)
            or not np.isfinite(self.feature_mean).all()
            or not np.isfinite(self.feature_std).all()
            or (self.feature_std <= 0).any()
        ):
            raise RuntimeError("Invalid RL feature scaler")

        class LSTMQNetwork(nn.Module):
            def __init__(self, obs_dim: int, hidden_size: int, actions: int = 4):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(obs_dim, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.SiLU(),
                )
                self.lstm = nn.LSTM(
                    hidden_size, hidden_size, batch_first=True
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, actions),
                )

            def forward(self, obs, hidden=None):
                encoded = self.encoder(obs)
                recurrent, hidden = self.lstm(encoded, hidden)
                return self.head(recurrent), hidden

        self.model = LSTMQNetwork(self.obs_dim, self.hidden_size).to(
            self.device
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.torch = torch
        self.version = str(manifest.get("version", artifact.stat().st_mtime_ns))
        self.loaded_at = time.time()
        with self._lock:
            self._states.clear()

    def warmup(self) -> None:
        """Прогреть Torch graph до публикации модели в registry."""

        if self.model is None or self.torch is None:
            return
        with self.torch.inference_mode():
            dummy = self.torch.zeros(
                (1, 1, self.obs_dim),
                dtype=self.torch.float32,
                device=self.device,
            )
            self.model(dummy)

    def _state_key(self, context: PredictionContext) -> str:
        if not context.pair_id or not context.strategy_name:
            raise ValueError(
                "RL inference requires pair_id and strategy_name"
            )
        direction = int(context.direction_code or 0)
        return f"{context.strategy_name}|{context.pair_id}|{direction}"

    def _feature_row(
        self,
        features: dict[str, Any],
        q35_bps: float,
        time_from_gate_sec: float,
        active: bool,
    ) -> np.ndarray:
        runtime = {
            "q35_causal_bps": q35_bps,
            "q35_gate_margin_bps": q35_bps - self.q35_gate_bps,
            "time_from_gate_sec": time_from_gate_sec,
            "agent_active": float(active),
        }
        missing = [
            name
            for name in self.feature_columns
            if name not in runtime and name not in features
        ]
        if missing:
            raise ValueError(
                f"{self.name} missing {len(missing)} features: "
                f"{', '.join(missing[:8])}"
            )
        values = np.asarray(
            [
                runtime.get(name, features.get(name, np.nan))
                for name in self.feature_columns
            ],
            dtype=np.float32,
        )
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return (values - self.feature_mean) / self.feature_std

    def _observation(
        self,
        features: dict[str, Any],
        manual_state: list[float],
        q35_bps: float,
        time_from_gate_sec: float,
        active: bool,
    ) -> np.ndarray:
        if len(manual_state) != MANUAL_STATE_SIZE:
            raise ValueError(
                f"RL position_state must have {MANUAL_STATE_SIZE} values"
            )
        row = self._feature_row(
            features, q35_bps, time_from_gate_sec, active
        )
        manual = np.asarray(manual_state, dtype=np.float32)
        return np.clip(
            np.concatenate([row, np.nan_to_num(manual)]), -20.0, 20.0
        ).astype(np.float32)

    def _replay_burn_in(
        self,
        context: PredictionContext,
        state: RecurrentPairState,
    ):
        """Восстановить hidden state по истории перед первым активным шагом."""

        assert self.model is not None and self.torch is not None
        history = context.history[-self.burn_in_steps :]
        timestamps = context.history_timestamps[-len(history) :]
        if not history:
            history = [context.features]
            timestamps = [int(context.grid_ts or context.decision_ts or 0)]
        gate_ts = int(state.gate_ts or context.grid_ts or context.decision_ts or 0)
        rows = []
        for index, features in enumerate(history):
            timestamp = (
                int(timestamps[index])
                if index < len(timestamps)
                else gate_ts - (len(history) - index - 1) * self.decision_ms
            )
            is_current = index == len(history) - 1
            manual = (
                list(context.position_state or [0.0] * MANUAL_STATE_SIZE)
                if is_current
                else [0.0] * MANUAL_STATE_SIZE
            )
            rows.append(
                self._observation(
                    features=features,
                    manual_state=manual,
                    q35_bps=state.frozen_q35_bps,
                    time_from_gate_sec=(timestamp - gate_ts) / 1000.0,
                    active=is_current,
                )
            )
        tensor = self.torch.from_numpy(np.stack(rows)).to(
            self.device
        ).unsqueeze(0)
        q_values, hidden = self.model(tensor)
        return q_values[0, -1], tuple(value.detach() for value in hidden)

    def _action_mask(
        self, context: PredictionContext, gate_active: bool
    ) -> np.ndarray:
        """Запретить действия, несовместимые с позицией и исполнением."""

        manual = list(context.position_state or [0.0] * MANUAL_STATE_SIZE)
        position_open = manual[0] > 0.5
        execution_valid = float(
            context.features.get("execution_valid_numeric", 0.0)
        ) > 0.5
        if not position_open:
            cooldown_complete = manual[6] <= 0.0
            can_enter = gate_active and execution_valid and cooldown_complete
            return np.asarray([True, can_enter, False, False], dtype=bool)
        hold_steps = max(0.0, manual[4]) * 3_000.0
        can_exit = execution_valid and hold_steps >= self.min_hold_steps
        return np.asarray([False, False, True, can_exit], dtype=bool)

    def _wait_outputs(
        self, gate_value: float, forced_exit: bool = False
    ) -> dict[str, float | int]:
        action = 3 if forced_exit else 0
        return {
            "action": action,
            "q_wait": 0.0,
            "q_enter": 0.0,
            "q_hold": 0.0,
            "q_exit": 0.0,
            "gate_active": 0,
            "frozen_q35_bps": gate_value,
            "gate_age_sec": 0.0,
            "state_replayed": 0,
            "forced_safety_exit": int(forced_exit),
        }

    def predict(
        self, context: PredictionContext, heads: list[str] | None = None
    ) -> Prediction:
        """Обновить recurrent state и выбрать допустимое действие по Q-values."""

        if self.model is None or self.torch is None:
            raise RuntimeError(f"{self.name} is not loaded")
        if context.decision_ts is None and context.grid_ts is None:
            raise ValueError("RL inference requires decision_ts or grid_ts")
        started = time.perf_counter()
        key = self._state_key(context)
        grid_ts = int(context.grid_ts or context.decision_ts)
        gate_value = float(context.gate_value or 0.0)
        if not math.isfinite(gate_value):
            gate_value = 0.0
        manual = list(context.position_state or [0.0] * MANUAL_STATE_SIZE)
        position_open = manual[0] > 0.5

        with self._lock, self.torch.inference_mode():
            state = self._states.setdefault(key, RecurrentPairState())
            if state.last_grid_ts == grid_ts and state.last_outputs is not None:
                return self._prediction(dict(state.last_outputs), started, heads)

            if (
                state.last_grid_ts is not None
                and grid_ts - state.last_grid_ts > self.max_state_gap_ms
            ):
                if position_open:
                    outputs = self._wait_outputs(gate_value, forced_exit=True)
                    state.last_grid_ts = grid_ts
                    state.last_outputs = outputs
                    return self._prediction(outputs, started, heads)
                state = RecurrentPairState()
                self._states[key] = state

            if state.gate_active and state.gate_ts is not None:
                age_steps = (grid_ts - state.gate_ts) // self.decision_ms
                if age_steps >= self.max_session_steps:
                    if position_open:
                        outputs = self._wait_outputs(
                            state.frozen_q35_bps, forced_exit=True
                        )
                        state.last_grid_ts = grid_ts
                        state.last_outputs = outputs
                        return self._prediction(outputs, started, heads)
                    state = RecurrentPairState()
                    self._states[key] = state

            replayed = False
            if not state.gate_active:
                if position_open:
                    outputs = self._wait_outputs(gate_value, forced_exit=True)
                    state.last_grid_ts = grid_ts
                    state.last_outputs = outputs
                    return self._prediction(outputs, started, heads)
                if gate_value < self.q35_gate_bps:
                    outputs = self._wait_outputs(gate_value)
                    state.last_grid_ts = grid_ts
                    state.last_outputs = outputs
                    return self._prediction(outputs, started, heads)
                state.gate_active = True
                state.frozen_q35_bps = gate_value
                state.gate_ts = grid_ts
                q_values, state.hidden = self._replay_burn_in(context, state)
                replayed = True
            else:
                time_from_gate = (
                    grid_ts - int(state.gate_ts or grid_ts)
                ) / 1000.0
                observation = self._observation(
                    features=context.features,
                    manual_state=manual,
                    q35_bps=state.frozen_q35_bps,
                    time_from_gate_sec=time_from_gate,
                    active=True,
                )
                tensor = self.torch.from_numpy(observation).to(
                    self.device
                ).view(1, 1, -1)
                q_sequence, hidden = self.model(tensor, state.hidden)
                q_values = q_sequence[0, -1]
                state.hidden = tuple(value.detach() for value in hidden)

            mask = self._action_mask(context, state.gate_active)
            q_numpy = q_values.detach().float().cpu().numpy()
            action = int(np.argmax(np.where(mask, q_numpy, -np.inf)))
            outputs = {
                "action": action,
                **{
                    f"q_{name}": float(q_numpy[index])
                    for index, name in enumerate(ACTION_NAMES)
                },
                "gate_active": int(state.gate_active),
                "frozen_q35_bps": float(state.frozen_q35_bps),
                "gate_age_sec": max(
                    0.0, (grid_ts - int(state.gate_ts or grid_ts)) / 1000.0
                ),
                "state_replayed": int(replayed),
                "forced_safety_exit": 0,
            }
            state.last_grid_ts = grid_ts
            state.last_outputs = outputs
            return self._prediction(outputs, started, heads)

    def close(self) -> None:
        """Сбросить все hidden states и освободить network."""

        with self._lock:
            self._states.clear()
        self.model = None
        self.torch = None
