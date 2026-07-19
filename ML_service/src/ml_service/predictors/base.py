"""Общие контракты model adapters и результатов инференса."""

from __future__ import annotations

import json
import math
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PredictionContext:
    """Все причинно доступные данные одного модельного решения."""

    features: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    history_timestamps: list[int] = field(default_factory=list)
    transformer_input: dict[str, Any] | None = None
    position_state: list[float] | None = None
    entry_snapshot: list[float] | None = None
    pair_id: str | None = None
    pair_type: str | None = None
    direction_code: int | None = None
    decision_ts: int | None = None
    grid_ts: int | None = None
    strategy_name: str | None = None
    gate_value: float | None = None


@dataclass(frozen=True)
class Prediction:
    """Нормализованный ответ модели вместе с latency и устройством."""

    model_name: str
    model_version: str
    outputs: dict[str, float | int]
    latency_ms: float
    device: str


@dataclass(frozen=True)
class PredictorStatus:
    """Операционное состояние загруженного model bundle."""

    name: str
    kind: str
    state: str
    version: str | None = None
    device: str | None = None
    detail: str | None = None
    loaded_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать статус для API."""

        return asdict(self)


def load_manifest(bundle_dir: Path) -> dict[str, Any]:
    """Прочитать обязательный manifest model bundle."""

    path = bundle_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing model manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


class Predictor(ABC):
    """Базовый интерфейс синхронного model adapter."""

    kind: str

    def __init__(self, name: str, bundle_dir: Path, requested_device: str):
        self.name = name
        self.bundle_dir = bundle_dir
        self.requested_device = requested_device
        self.version = "unknown"
        self.device = "cpu"
        self.detail: str | None = None
        self.loaded_at: float | None = None

    @abstractmethod
    def load(self) -> None:
        """Загрузить и проверить артефакты модели."""

        raise NotImplementedError

    @abstractmethod
    def predict(
        self, context: PredictionContext, heads: list[str] | None = None
    ) -> Prediction:
        """Выполнить инференс по причинному контексту."""

        raise NotImplementedError

    def warmup(self) -> None:
        """Опционально прогреть модель перед атомарной заменой в registry."""

    def close(self) -> None:
        """Освободить ресурсы модели."""

        pass

    def _prediction(
        self,
        outputs: dict[str, float | int],
        started: float,
        heads: list[str] | None,
    ) -> Prediction:
        invalid = [
            name
            for name, value in outputs.items()
            if not isinstance(value, (int, float)) or not math.isfinite(float(value))
        ]
        if invalid:
            raise ValueError(
                f"{self.name} returned non-finite outputs: {', '.join(invalid)}"
            )
        if heads is not None:
            selected = set(heads)
            unknown = sorted(selected - outputs.keys())
            if unknown:
                raise ValueError(
                    f"{self.name} has no requested outputs: {', '.join(unknown)}"
                )
            outputs = {key: value for key, value in outputs.items() if key in selected}
        return Prediction(
            model_name=self.name,
            model_version=self.version,
            outputs=outputs,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            device=self.device,
        )
