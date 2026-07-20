"""Координация latest-wins инференса по зарегистрированным парам."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import OrderedDict
from typing import Any

from ml_service.metrics import (
    DROPPED_PAIR_WORK,
    MODEL_ERRORS,
    MODEL_INFERENCE,
    PENDING_PAIRS,
)
from ml_service.ohlcv import OHLCVProvider
from ml_service.paper.engine import PaperEngine
from ml_service.predictors.base import PredictionContext
from ml_service.predictors.registry import PredictorRegistry
from ml_service.preprocessing.features import PairFeatureEngine, PairSnapshot

LOGGER = logging.getLogger(__name__)


class InferenceCoordinator:
    """Объединяет обновления и всегда оценивает самое новое состояние пары."""

    def __init__(
        self,
        feature_engine: PairFeatureEngine,
        registry: PredictorRegistry,
        paper: PaperEngine,
        ohlcv_provider: OHLCVProvider,
        max_pending_pairs: int = 10_000,
    ):
        self.feature_engine = feature_engine
        self.registry = registry
        self.paper = paper
        self.ohlcv_provider = ohlcv_provider
        self.max_pending_pairs = int(max_pending_pairs)
        self._pending: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.RLock()
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopping = False

    async def start(self) -> None:
        """Запустить единственный worker очереди пар."""

        self._stopping = False
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(self._run(), name="ml-inference-coordinator")

    async def stop(self) -> None:
        """Остановить worker и дождаться завершения фоновой задачи."""

        self._stopping = True
        self._wake.set()
        if self._task is not None:
            await self._task
            self._task = None
        self._loop = None

    def schedule(self, pair_ids: tuple[str, ...] | list[str]) -> None:
        """Поставить пары в очередь, не создавая дубликаты старых задач."""

        with self._lock:
            for pair_id in pair_ids:
                if pair_id in self._pending:
                    self._pending.move_to_end(pair_id)
                    DROPPED_PAIR_WORK.inc()
                else:
                    self._pending[pair_id] = None
                if len(self._pending) > self.max_pending_pairs:
                    self._pending.popitem(last=False)
                    DROPPED_PAIR_WORK.inc()
            PENDING_PAIRS.set(len(self._pending))
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._wake.set)

    async def _run(self) -> None:
        while True:
            await self._wake.wait()
            while True:
                with self._lock:
                    if not self._pending:
                        self._wake.clear()
                        PENDING_PAIRS.set(0)
                        break
                    pair_id, _ = self._pending.popitem(last=False)
                    PENDING_PAIRS.set(len(self._pending))
                await asyncio.to_thread(self.evaluate_pair, pair_id)
            if self._stopping:
                return

    def evaluate_pair(self, pair_id: str) -> None:
        """Синхронно обработать одну пару в worker thread."""

        pair = self.feature_engine.store.get_pair(pair_id)
        if pair is None:
            return
        leg1 = self.feature_engine.store.latest_l2(
            pair.leg1.exchange, pair.leg1.ticker
        )
        leg2 = self.feature_engine.store.latest_l2(
            pair.leg2.exchange, pair.leg2.ticker
        )
        if leg1 is None or leg2 is None:
            return
        decision_ts = max(leg1.machine_ts_final, leg2.machine_ts_final)
        self.ohlcv_provider.ensure_pair(pair, decision_ts)
        snapshot = self.feature_engine.build(pair_id)
        if snapshot is None:
            return
        pair_history = self.feature_engine.history(pair_id)
        history = [item.features for item in pair_history]
        history_timestamps = [item.grid_ts for item in pair_history]
        prediction_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

        for strategy_name in self.paper.active_strategy_names():
            if not self.paper.applies_to(
                strategy_name, snapshot.pair.pair_type
            ):
                continue

            # Решение в t исполняется по стакану t+100 мс. Pending fill нужно
            # применить до построения следующего recurrent observation.
            self.paper.advance(strategy_name, snapshot)
            transformer_position_state = self.paper.position_state(
                strategy_name, snapshot
            )
            rl_position_state = self.paper.rl_position_state(
                strategy_name, snapshot
            )
            models = self.paper.required_models(
                strategy_name, snapshot.pair.pair_type
            )
            gate_source = self.paper.gate_source(
                strategy_name, snapshot.pair.pair_type
            )
            predictions: dict[str, Any] = {}
            ordered_models = sorted(
                models,
                key=lambda name: self.registry.kind(name) == "rl",
            )
            for model_name in ordered_models:
                if not self.registry.ready(model_name):
                    predictions[model_name] = {"_error": "model_not_ready"}
                    continue
                model_kind = self.registry.kind(model_name)
                position_state = (
                    rl_position_state
                    if model_kind == "rl"
                    else transformer_position_state
                )
                cache_key = (
                    model_name,
                    strategy_name if model_kind == "rl" else "",
                    tuple(position_state)
                    if model_kind in {"transformer", "rl"}
                    else (),
                )
                if cache_key in prediction_cache:
                    predictions[model_name] = prediction_cache[cache_key]
                    continue
                try:
                    prediction = self.registry.predict(
                        model_name,
                        PredictionContext(
                            features=snapshot.features,
                            history=history,
                            history_timestamps=history_timestamps,
                            position_state=position_state,
                            pair_id=snapshot.pair.pair_id,
                            pair_type=snapshot.pair.pair_type,
                            direction_code=snapshot.pair.direction_code,
                            decision_ts=snapshot.decision_ts,
                            grid_ts=snapshot.grid_ts,
                            strategy_name=strategy_name,
                            gate_value=self.paper.source_value(
                                gate_source, predictions
                            ),
                        ),
                    )
                    MODEL_INFERENCE.labels(
                        model=model_name, device=prediction.device
                    ).observe(prediction.latency_ms / 1000.0)
                    values = {
                        **prediction.outputs,
                        "_version": prediction.model_version,
                        "_device": prediction.device,
                        "_latency_ms": prediction.latency_ms,
                    }
                except Exception as error:
                    message = str(error)
                    transformer_warmup = (
                        model_kind == "transformer"
                        and message.startswith(
                            "Transformer history is not ready:"
                        )
                    )
                    if transformer_warmup:
                        values = {"_waiting": message}
                    else:
                        MODEL_ERRORS.labels(model=model_name).inc()
                        LOGGER.debug(
                            "Prediction skipped model=%s pair=%s: %s",
                            model_name,
                            pair_id,
                            error,
                        )
                        values = {"_error": message}
                prediction_cache[cache_key] = values
                predictions[model_name] = values
            self.paper.evaluate(strategy_name, snapshot, predictions)
