"""HTTP API для рыночного потока, моделей и paper trading."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ml_service import __version__
from ml_service.context import AppContext
from ml_service.metrics import DUPLICATES, INGESTED
from ml_service.monitor import monitor_overview, monitor_pair_series
from ml_service.predictors.base import PredictionContext
from ml_service.schemas import (
    DirectPredictRequest,
    L2BatchRequest,
    L2Snapshot,
    ModelActionResponse,
    OHLCVBatchRequest,
    OHLCVCandle,
    PairDefinition,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Создать runtime при старте FastAPI и корректно закрыть при остановке."""

    context = await AppContext.create()
    logging.basicConfig(
        level=context.settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    app.state.context = context
    try:
        yield
    finally:
        await context.close()


app = FastAPI(
    title="Arbitrage ML Service",
    version=__version__,
    lifespan=lifespan,
)


def get_context(request: Request) -> AppContext:
    """Вернуть общий контекст текущего процесса."""

    return request.app.state.context


Context = Annotated[AppContext, Depends(get_context)]


@app.get("/")
def root() -> dict[str, str]:
    """Вернуть имя и версию сервиса."""

    return {"service": "arb-ml-service", "version": __version__}


@app.get("/health/live")
def live() -> dict[str, str]:
    """Подтвердить, что HTTP-процесс отвечает."""

    return {"status": "alive"}


@app.get("/health/ready")
def ready(context: Context, response: Response) -> dict[str, object]:
    """Проверить готовность моделей активных стратегий и runtime-хранилищ."""

    registry_ready, missing_required = context.registry.readiness()
    missing_active = sorted(
        {
            model
            for strategy in context.paper.active_strategy_names()
            for model in context.paper.required_models(strategy)
            if not context.registry.ready(model)
        }
    )
    is_ready = registry_ready and not missing_active
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ready" if is_ready else "not_ready",
        "missing_required_models": missing_required,
        "missing_active_strategy_models": missing_active,
        "state": context.store.state_counts(),
        "ohlcv_provider": context.ohlcv.status(),
        "paper_store": context.repository.status(),
    }


@app.get("/metrics")
def metrics() -> Response:
    """Экспортировать метрики в формате Prometheus."""

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/monitor/overview")
def monitoring_overview(context: Context) -> dict[str, object]:
    """Вернуть компактный снимок состояния для dashboard."""

    return monitor_overview(context)


@app.get("/v1/monitor/pairs/{pair_id}/series")
def monitoring_pair_series(
    pair_id: str,
    context: Context,
    limit: int = Query(default=1_800, ge=100, le=8_192),
) -> dict[str, object]:
    """Вернуть историю edge и торговые маркеры выбранной пары."""

    result = monitor_pair_series(context, pair_id, limit)
    if result is None:
        raise HTTPException(status_code=404, detail="unknown pair")
    return result


@app.get("/v1/models")
def models(context: Context) -> list[dict[str, object]]:
    """Показать состояние, устройство и версию всех моделей."""

    return context.registry.statuses()


@app.post("/v1/models/{name}/load", response_model=ModelActionResponse)
async def load_model(name: str, context: Context) -> ModelActionResponse:
    """Загрузить и прогреть модель перед атомарной публикацией."""

    if name not in context.settings.models:
        raise HTTPException(status_code=404, detail="unknown model")
    try:
        result = await asyncio.to_thread(context.registry.load, name)
    except Exception as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return ModelActionResponse(name=name, status=result.state, detail=result.detail)


@app.post("/v1/models/{name}/reload", response_model=ModelActionResponse)
async def reload_model(name: str, context: Context) -> ModelActionResponse:
    """Повторно загрузить model bundle без остановки API."""

    return await load_model(name, context)


@app.post("/v1/models/{name}/unload", response_model=ModelActionResponse)
async def unload_model(
    name: str,
    context: Context,
    force: bool = Query(default=False),
) -> ModelActionResponse:
    """Выгрузить модель, если её не использует активная стратегия."""

    if name not in context.settings.models:
        raise HTTPException(status_code=404, detail="unknown model")
    users = [
        strategy
        for strategy in context.paper.active_strategy_names()
        if name in context.paper.required_models(strategy)
    ]
    if users and not force:
        raise HTTPException(
            status_code=409,
            detail=f"model is used by active strategies: {', '.join(users)}",
        )
    result = await asyncio.to_thread(context.registry.unload, name)
    return ModelActionResponse(name=name, status=result.state)


@app.post("/v1/predict/{name}")
async def predict(
    name: str,
    payload: DirectPredictRequest,
    context: Context,
) -> dict[str, object]:
    """Выполнить stateless direct inference для диагностики контракта."""

    if name not in context.settings.models:
        raise HTTPException(status_code=404, detail="unknown model")
    try:
        prediction = await asyncio.to_thread(
            context.registry.predict,
            name,
            PredictionContext(
                features=payload.features,
                transformer_input=payload.transformer_input,
            ),
            payload.heads,
        )
    except Exception as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {
        "model": prediction.model_name,
        "version": prediction.model_version,
        "device": prediction.device,
        "latency_ms": prediction.latency_ms,
        "outputs": prediction.outputs,
    }


@app.get("/v1/pairs")
def pairs(context: Context) -> list[PairDefinition]:
    """Вернуть зарегистрированные направления торговых пар."""

    return context.store.list_pairs()


@app.put("/v1/pairs/{pair_id}")
def put_pair(
    pair_id: str,
    pair: PairDefinition,
    context: Context,
) -> dict[str, str]:
    """Зарегистрировать или заменить направление пары."""

    if pair_id != pair.pair_id:
        raise HTTPException(status_code=422, detail="path and payload pair_id differ")
    context.store.register_pair(pair)
    return {"status": "registered", "pair_id": pair_id}


@app.post("/v1/market/l2", status_code=status.HTTP_202_ACCEPTED)
def ingest_l2(snapshot: L2Snapshot, context: Context) -> dict[str, object]:
    """Принять один L2-снимок и запланировать затронутые пары."""

    result = context.store.ingest_l2(snapshot)
    if result.accepted:
        INGESTED.labels(kind="l2").inc()
    if result.duplicate_book:
        DUPLICATES.labels(kind="l2").inc()
    context.coordinator.schedule(result.affected_pairs)
    return {
        "accepted": result.accepted,
        "duplicate_book": result.duplicate_book,
        "scheduled_pairs": len(result.affected_pairs),
    }


@app.post("/v1/market/l2/batch", status_code=status.HTTP_202_ACCEPTED)
def ingest_l2_batch(
    batch: L2BatchRequest,
    context: Context,
) -> dict[str, object]:
    """Принять batch L2-снимков с одной виртуальной временной точки."""

    accepted = 0
    duplicate_books = 0
    affected_pairs: set[str] = set()
    for snapshot in batch.snapshots:
        result = context.store.ingest_l2(snapshot)
        accepted += int(result.accepted)
        duplicate_books += int(result.duplicate_book)
        affected_pairs.update(result.affected_pairs)
    if accepted:
        INGESTED.labels(kind="l2").inc(accepted)
    if duplicate_books:
        DUPLICATES.labels(kind="l2").inc(duplicate_books)
    context.coordinator.schedule(sorted(affected_pairs))
    return {
        "received": len(batch.snapshots),
        "accepted": accepted,
        "rejected": len(batch.snapshots) - accepted,
        "duplicate_books": duplicate_books,
        "scheduled_pairs": len(affected_pairs),
        "watermark_ts": max(
            snapshot.machine_ts_final for snapshot in batch.snapshots
        ),
    }


@app.post("/v1/market/ohlcv", status_code=status.HTTP_202_ACCEPTED)
def ingest_ohlcv(candle: OHLCVCandle, context: Context) -> dict[str, object]:
    """Принять одну уже закрытую свечу."""

    result = context.store.ingest_ohlcv(candle)
    if result.accepted:
        INGESTED.labels(kind="ohlcv").inc()
    if result.duplicate_book:
        DUPLICATES.labels(kind="ohlcv").inc()
    context.coordinator.schedule(result.affected_pairs)
    return {
        "accepted": result.accepted,
        "duplicate": result.duplicate_book,
        "scheduled_pairs": len(result.affected_pairs),
    }


@app.post("/v1/market/ohlcv/batch", status_code=status.HTTP_202_ACCEPTED)
def ingest_ohlcv_batch(
    batch: OHLCVBatchRequest,
    context: Context,
) -> dict[str, object]:
    """Принять batch причинно доступных OHLCV-свечей."""

    accepted = 0
    duplicates = 0
    affected_pairs: set[str] = set()
    for candle in batch.candles:
        result = context.store.ingest_ohlcv(candle)
        accepted += int(result.accepted)
        duplicates += int(result.duplicate_book)
        affected_pairs.update(result.affected_pairs)
    if accepted:
        INGESTED.labels(kind="ohlcv").inc(accepted)
    if duplicates:
        DUPLICATES.labels(kind="ohlcv").inc(duplicates)
    context.coordinator.schedule(sorted(affected_pairs))
    return {
        "received": len(batch.candles),
        "accepted": accepted,
        "rejected": len(batch.candles) - accepted,
        "duplicates": duplicates,
        "scheduled_pairs": len(affected_pairs),
        "watermark_ts": max(
            candle.ts + (60_000 if candle.tf == "1m" else 300_000)
            for candle in batch.candles
        ),
    }


@app.get("/v1/strategies")
def strategies(context: Context) -> list[dict[str, object]]:
    """Вернуть конфигурацию и runtime-состояние paper-стратегий."""

    return context.paper.strategies()


@app.post("/v1/strategies/{name}/start")
def start_strategy(name: str, context: Context) -> dict[str, str]:
    """Активировать стратегию после проверки её моделей."""

    try:
        missing = sorted(
            model
            for model in context.paper.required_models(name)
            if not context.registry.ready(model)
        )
        if missing:
            raise HTTPException(
                status_code=409,
                detail=f"strategy models are not ready: {', '.join(missing)}",
            )
        context.paper.set_active(name, True)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="unknown strategy") from error
    return {"name": name, "status": "active"}


@app.post("/v1/strategies/{name}/pause")
def pause_strategy(name: str, context: Context) -> dict[str, str]:
    """Приостановить новые решения стратегии."""

    try:
        context.paper.set_active(name, False)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="unknown strategy") from error
    return {"name": name, "status": "paused"}


@app.get("/v1/paper/positions")
def paper_positions(context: Context) -> list[dict[str, object]]:
    """Вернуть открытые и ожидающие исполнения paper-позиции."""

    return context.paper.positions()


@app.get("/v1/paper/trades")
def paper_trades(
    context: Context,
    limit: int = Query(default=200, ge=1, le=10_000),
    strategy_name: str | None = None,
) -> list[dict[str, object]]:
    """Вернуть последние закрытые сделки с необязательным фильтром стратегии."""

    return context.repository.trades(limit=limit, strategy=strategy_name)


@app.get("/v1/paper/decisions")
def paper_decisions(
    context: Context,
    limit: int = Query(default=200, ge=1, le=10_000),
    strategy_name: str | None = None,
) -> list[dict[str, object]]:
    """Вернуть журнал последних решений моделей и стратегий."""

    return context.repository.decisions(limit=limit, strategy=strategy_name)


@app.get("/v1/paper/stats")
def paper_stats(context: Context) -> list[dict[str, object]]:
    """Агрегировать PnL и риск-метрики по стратегиям."""

    return context.repository.stats()
