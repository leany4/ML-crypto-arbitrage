"""Загрузка типизированной конфигурации сервиса из YAML и окружения."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value or {}


@dataclass(frozen=True)
class ModelSettings:
    """Параметры одного model bundle в runtime."""

    name: str
    kind: str
    bundle_dir: Path
    enabled: bool = True
    required: bool = False
    device: str = "auto"


@dataclass(frozen=True)
class StrategySettings:
    """Декларативная конфигурация одной paper-стратегии."""

    name: str
    enabled: bool = False
    values: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Settings:
    """Полный неизменяемый снимок настроек процесса."""

    config_dir: Path
    models_dir: Path
    state_db: Path
    device: str
    log_level: str
    service: dict[str, Any]
    models: dict[str, ModelSettings]
    strategies: dict[str, StrategySettings]

    @classmethod
    def load(cls) -> "Settings":
        """Загрузить YAML, применить env overrides и разрешить пути bundle."""

        config_dir = Path(os.getenv("ML_CONFIG_DIR", "config")).resolve()
        models_dir = Path(os.getenv("ML_MODELS_DIR", "models")).resolve()
        state_db = Path(os.getenv("ML_STATE_DB", "state/paper.sqlite3")).resolve()
        global_device = os.getenv("ML_DEVICE", "auto").lower()

        service = _read_yaml(config_dir / "service.yaml").get("service", {})
        raw_models = _read_yaml(config_dir / "models.yaml").get("models", {})
        raw_strategies = _read_yaml(config_dir / "strategies.yaml").get("strategies", {})

        models: dict[str, ModelSettings] = {}
        for name, value in raw_models.items():
            configured_bundle = Path(value.get("bundle_dir", name))
            if not configured_bundle.is_absolute():
                configured_bundle = models_dir / configured_bundle
            kind = str(value["kind"])
            configured_device = str(value.get("device", "auto")).lower()
            if kind == "q35":
                model_device = "cpu"
            elif global_device != "auto":
                model_device = global_device
            else:
                model_device = configured_device
            models[name] = ModelSettings(
                name=name,
                kind=kind,
                bundle_dir=configured_bundle,
                enabled=bool(value.get("enabled", True)),
                required=bool(value.get("required", False)),
                device=model_device,
            )

        strategies = {
            name: StrategySettings(
                name=name,
                enabled=bool(value.get("enabled", False)),
                values={key: item for key, item in value.items() if key != "enabled"},
            )
            for name, value in raw_strategies.items()
        }

        return cls(
            config_dir=config_dir,
            models_dir=models_dir,
            state_db=state_db,
            device=global_device,
            log_level=os.getenv("ML_LOG_LEVEL", "INFO").upper(),
            service=service,
            models=models,
            strategies=strategies,
        )
