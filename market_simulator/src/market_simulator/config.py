"""Конфигурация replay-сервиса из переменных окружения."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SimulatorSettings:
    """Неизменяемые параметры источников данных и ML API."""

    data_dir: Path
    prepared_dir: Path
    ml_url: str
    batch_endpoint: str
    request_timeout_seconds: float
    default_speed: float
    default_duration_seconds: int
    ohlcv_batch_endpoint: str = "/v1/market/ohlcv/batch"
    ohlcv_warmup_minutes: int = 720

    @classmethod
    def load(cls) -> "SimulatorSettings":
        """Загрузить настройки и разрешить локальные пути."""

        return cls(
            data_dir=Path(os.getenv("SIM_DATA_DIR", "../ML_service/raw_last_3d")).resolve(),
            prepared_dir=Path(os.getenv("SIM_PREPARED_DIR", "state/prepared")).resolve(),
            ml_url=os.getenv("SIM_ML_URL", "http://127.0.0.1:8080").rstrip("/"),
            batch_endpoint=os.getenv("SIM_BATCH_ENDPOINT", "/v1/market/l2/batch"),
            request_timeout_seconds=float(os.getenv("SIM_REQUEST_TIMEOUT_SECONDS", "30")),
            default_speed=float(os.getenv("SIM_DEFAULT_SPEED", "1")),
            default_duration_seconds=int(os.getenv("SIM_DEFAULT_DURATION_SECONDS", "900")),
            ohlcv_batch_endpoint=os.getenv(
                "SIM_OHLCV_BATCH_ENDPOINT", "/v1/market/ohlcv/batch"
            ),
            ohlcv_warmup_minutes=int(
                os.getenv("SIM_OHLCV_WARMUP_MINUTES", "720")
            ),
        )
