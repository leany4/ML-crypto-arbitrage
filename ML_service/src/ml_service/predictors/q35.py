"""Адаптер legacy q35-регрессора scikit-learn."""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from ml_service.predictors.base import (
    Prediction,
    PredictionContext,
    Predictor,
    load_manifest,
)


class Q35Predictor(Predictor):
    """Проверяет feature contract и выдаёт консервативный прогноз в bps."""

    kind = "q35"

    def __init__(self, name: str, bundle_dir: Path, requested_device: str):
        super().__init__(name, bundle_dir, requested_device)
        self.model = None
        self.feature_columns: list[str] = []
        self.pair_types: set[str] = set()
        self.output_name = "watch_q35_bps"

    def load(self) -> None:
        """Загрузить доверенный joblib с точной версией scikit-learn."""

        manifest = load_manifest(self.bundle_dir)
        expected_version = str(manifest.get("sklearn_version", ""))
        if expected_version and sklearn.__version__ != expected_version:
            raise RuntimeError(
                f"{self.name} requires scikit-learn {expected_version}, "
                f"running {sklearn.__version__}"
            )
        artifact = self.bundle_dir / manifest.get("artifact", "model.joblib")
        if not artifact.exists():
            raise FileNotFoundError(artifact)
        self.model = joblib.load(artifact)
        if not hasattr(self.model, "feature_names_in_"):
            raise RuntimeError("q35 artifact has no feature_names_in_ contract")
        self.feature_columns = [str(value) for value in self.model.feature_names_in_]
        self.pair_types = {
            str(value) for value in manifest.get("pair_types", [])
        }
        self.output_name = str(manifest.get("output", "watch_q35_bps"))
        self.version = str(manifest.get("version", artifact.stat().st_mtime_ns))
        self.device = "cpu"
        self.loaded_at = time.time()

    def predict(
        self, context: PredictionContext, heads: list[str] | None = None
    ) -> Prediction:
        """Собрать строку в обучающем порядке признаков и выполнить прогноз."""

        if self.model is None:
            raise RuntimeError(f"{self.name} is not loaded")
        if (
            context.pair_type is not None
            and self.pair_types
            and context.pair_type not in self.pair_types
        ):
            raise ValueError(
                f"{self.name} does not support pair_type={context.pair_type}"
            )
        started = time.perf_counter()
        missing = [name for name in self.feature_columns if name not in context.features]
        if missing:
            preview = ", ".join(missing[:8])
            raise ValueError(
                f"{self.name} missing {len(missing)} features: {preview}"
            )
        row = {
            name: context.features.get(name, np.nan)
            for name in self.feature_columns
        }
        value = float(self.model.predict(pd.DataFrame([row]))[0])
        return self._prediction({self.output_name: value}, started, heads)

    def close(self) -> None:
        """Удалить ссылку на модель и её runtime-контракт."""

        self.model = None
        self.pair_types.clear()
