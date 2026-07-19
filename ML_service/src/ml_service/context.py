"""Сборка зависимостей приложения и управление их жизненным циклом."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from ml_service.config import Settings
from ml_service.coordinator import InferenceCoordinator
from ml_service.ohlcv import OHLCVProvider, build_ohlcv_provider
from ml_service.paper.engine import PaperEngine
from ml_service.paper.repository import InMemoryPaperRepository, PaperRepository
from ml_service.predictors.registry import PredictorRegistry
from ml_service.preprocessing.features import PairFeatureEngine
from ml_service.state.market import MarketStateStore

LOGGER = logging.getLogger(__name__)


@dataclass
class AppContext:
    """Общие runtime-компоненты одного процесса ML-сервиса."""

    settings: Settings
    store: MarketStateStore
    features: PairFeatureEngine
    registry: PredictorRegistry
    repository: PaperRepository | InMemoryPaperRepository
    paper: PaperEngine
    ohlcv: OHLCVProvider
    coordinator: InferenceCoordinator

    @classmethod
    async def create(cls) -> "AppContext":
        """Создать хранилища, модели, стратегии и фоновые координаторы."""

        settings = Settings.load()
        service = settings.service
        store = MarketStateStore(
            l2_history_size=int(service.get("l2_history_size", 4096)),
            ohlcv_history_size=int(service.get("ohlcv_history_size", 128)),
        )
        pair_history_size = int(service.get("pair_history_size", 4096))
        for model in settings.models.values():
            if not model.enabled or model.kind != "transformer":
                continue
            manifest_path = model.bundle_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            contract_path = model.bundle_dir / manifest.get(
                "dataset_contract", "dataset_contract.json"
            )
            if not contract_path.exists():
                continue
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            required_history = max(
                int(contract["local_history_steps"]),
                (int(contract["long_history_tokens"]) - 1)
                * int(contract["long_history_stride_steps"])
                + 1,
            )
            if pair_history_size < required_history:
                raise RuntimeError(
                    f"pair_history_size={pair_history_size} is below "
                    f"{model.name} contract requirement {required_history}"
                )
        features = PairFeatureEngine(
            store=store,
            history_size=pair_history_size,
            decision_ms=100,
            max_book_age_ms=int(service.get("max_book_age_ms", 1500)),
            max_pair_skew_ms=int(service.get("max_pair_skew_ms", 500)),
            notional_usd=float(service.get("default_notional_usd", 100.0)),
            fee_bps_per_leg_side=float(
                service.get("fee_bps_per_leg_side", 10.0)
            ),
            allowed_pair_types=service.get("allowed_pair_types"),
        )
        registry = PredictorRegistry(settings.models)
        repository_config = service.get("paper_repository", {})
        repository_mode = os.getenv(
            "ML_PAPER_STORE",
            str(repository_config.get("mode", "memory")),
        ).lower()
        if repository_mode == "memory":
            repository = InMemoryPaperRepository(
                max_trades=int(repository_config.get("max_trades", 2_000)),
                max_decisions=int(
                    repository_config.get("max_decisions", 5_000)
                ),
                max_events=int(repository_config.get("max_events", 1_000)),
            )
        elif repository_mode == "sqlite":
            repository = PaperRepository(settings.state_db)
        else:
            raise ValueError(
                f"Unsupported paper_repository mode: {repository_mode}"
            )
        paper = PaperEngine(
            repository=repository,
            strategies=settings.strategies,
            fee_bps_per_leg_side=float(
                service.get("fee_bps_per_leg_side", 10.0)
            ),
            min_fill_share=float(service.get("min_fill_share", 0.95)),
        )
        ohlcv = build_ohlcv_provider(store, service)
        coordinator = InferenceCoordinator(
            feature_engine=features,
            registry=registry,
            paper=paper,
            ohlcv_provider=ohlcv,
            max_pending_pairs=int(service.get("queue_max_pairs", 10_000)),
        )
        context = cls(
            settings=settings,
            store=store,
            features=features,
            registry=registry,
            repository=repository,
            paper=paper,
            ohlcv=ohlcv,
            coordinator=coordinator,
        )
        registry.load_enabled()
        await coordinator.start()
        return context

    async def close(self) -> None:
        """Корректно остановить фоновые задачи и освободить модели."""

        await self.coordinator.stop()
        self.ohlcv.close()
        self.registry.close()
        self.repository.close()
