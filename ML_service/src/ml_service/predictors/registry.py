"""Потокобезопасный registry загружаемых model adapters."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from ml_service.config import ModelSettings
from ml_service.predictors.base import (
    Prediction,
    PredictionContext,
    Predictor,
    PredictorStatus,
)

LOGGER = logging.getLogger(__name__)


def _predictor_type(kind: str) -> type[Predictor]:
    if kind == "q35":
        from ml_service.predictors.q35 import Q35Predictor

        return Q35Predictor
    if kind == "transformer":
        from ml_service.predictors.transformer.predictor import TransformerPredictor

        return TransformerPredictor
    if kind == "rl":
        from ml_service.predictors.rl import RLPredictor

        return RLPredictor
    raise ValueError(f"Unknown predictor kind: {kind}")


class PredictorRegistry:
    """Управляет жизненным циклом моделей и атомарной горячей заменой."""

    def __init__(self, settings: dict[str, ModelSettings]):
        self.settings = settings
        self._lock = threading.RLock()
        self._predictors: dict[str, Predictor] = {}
        self._statuses: dict[str, PredictorStatus] = {
            name: PredictorStatus(name=name, kind=value.kind, state="disabled")
            for name, value in settings.items()
        }

    def load_enabled(self) -> None:
        """Загрузить все модели, включённые в конфигурации."""

        for name, setting in self.settings.items():
            if setting.enabled:
                try:
                    self.load(name)
                except Exception:
                    LOGGER.exception("Model %s could not be loaded", name)

    def load(self, name: str) -> PredictorStatus:
        """Подготовить candidate и опубликовать его только после warm-up."""

        setting = self.settings[name]
        try:
            predictor_type = _predictor_type(setting.kind)
            candidate = predictor_type(name, setting.bundle_dir, setting.device)
            candidate.load()
            candidate.warmup()
        except Exception as error:
            status = PredictorStatus(
                name=name,
                kind=setting.kind,
                state="error",
                detail=str(error),
            )
            with self._lock:
                self._statuses[name] = status
            raise

        with self._lock:
            previous = self._predictors.get(name)
            self._predictors[name] = candidate
            self._statuses[name] = PredictorStatus(
                name=name,
                kind=setting.kind,
                state="ready",
                version=candidate.version,
                device=candidate.device,
                detail=candidate.detail,
                loaded_at=candidate.loaded_at,
            )
        if previous is not None:
            previous.close()
        return self._statuses[name]

    def unload(self, name: str) -> PredictorStatus:
        """Атомарно убрать модель из serving и освободить её ресурсы."""

        with self._lock:
            predictor = self._predictors.pop(name, None)
            setting = self.settings[name]
            status = PredictorStatus(
                name=name, kind=setting.kind, state="unloaded"
            )
            self._statuses[name] = status
        if predictor is not None:
            predictor.close()
        return status

    def predict(
        self,
        name: str,
        context: PredictionContext,
        heads: list[str] | None = None,
    ) -> Prediction:
        """Выполнить инференс через текущую опубликованную версию модели."""

        with self._lock:
            predictor = self._predictors.get(name)
        if predictor is None:
            raise RuntimeError(f"Predictor {name} is not ready")
        return predictor.predict(context, heads)

    def statuses(self) -> list[dict[str, Any]]:
        """Вернуть сериализуемые статусы всех configured models."""

        with self._lock:
            return [status.to_dict() for status in self._statuses.values()]

    def status(self, name: str) -> PredictorStatus:
        """Вернуть статус одной модели."""

        with self._lock:
            return self._statuses[name]

    def ready(self, name: str) -> bool:
        """Проверить, опубликован ли predictor для инференса."""

        with self._lock:
            return name in self._predictors

    def kind(self, name: str) -> str:
        """Вернуть тип model adapter из конфигурации."""

        return self.settings[name].kind

    def readiness(self) -> tuple[bool, list[str]]:
        """Проверить наличие всех моделей с флагом `required`."""

        missing = [
            name
            for name, setting in self.settings.items()
            if setting.required and not self.ready(name)
        ]
        return not missing, missing

    def close(self) -> None:
        """Закрыть все загруженные predictors при остановке процесса."""

        with self._lock:
            predictors = list(self._predictors.values())
            self._predictors.clear()
        for predictor in predictors:
            predictor.close()
