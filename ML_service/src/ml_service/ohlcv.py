"""Источники закрытых OHLCV-свечей для online-препроцессинга."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Protocol

import polars as pl

from ml_service.schemas import MarketRef, OHLCVCandle, PairDefinition
from ml_service.state.market import MarketStateStore


LOGGER = logging.getLogger(__name__)
TIMEFRAME_MS = {"1m": 60_000, "5m": 300_000}


class OHLCVProvider(Protocol):
    """Единый интерфейс replay, parquet и live OHLCV providers."""

    def ensure_pair(self, pair: PairDefinition, decision_ts: int) -> int:
        """Обеспечить доступную историю обеих ног."""

        ...

    def status(self) -> dict[str, Any]:
        """Вернуть операционные счётчики provider."""

        ...

    def close(self) -> None:
        """Освободить внешние соединения и ресурсы."""

        ...


class DisabledOHLCVProvider:
    """Пустой provider, когда свечи приходят через HTTP ingest."""

    def ensure_pair(self, pair: PairDefinition, decision_ts: int) -> int:
        """Ничего не загружать при внешнем HTTP ingest свечей."""

        return 0

    def status(self) -> dict[str, Any]:
        """Сообщить dashboard об отключённом provider."""

        return {"mode": "disabled", "series_loaded": 0, "errors": 0}

    def close(self) -> None:
        return None


class ParquetOHLCVProvider:
    """Загружает только свечи, закрытые к текущему виртуальному времени."""

    def __init__(
        self,
        store: MarketStateStore,
        path: Path,
        history_size: int = 128,
    ):
        self.store = store
        self.path = path.resolve()
        self.history_size = int(history_size)
        self._lock = threading.RLock()
        self._loaded_until: dict[tuple[str, str, str], int] = {}
        self._rows_loaded = 0
        self._errors = 0
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        schema = pl.read_parquet_schema(self.path)
        required = {
            "exchange",
            "symbol",
            "tf",
            "ts",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        missing = required - set(schema)
        if missing:
            raise RuntimeError(f"OHLCV parquet misses columns: {sorted(missing)}")

    @staticmethod
    def _latest_closed_ts(decision_ts: int, duration_ms: int) -> int:
        return (int(decision_ts) - duration_ms) // duration_ms * duration_ms

    def _ensure_market(
        self,
        market: MarketRef,
        tf: str,
        decision_ts: int,
    ) -> int:
        duration_ms = TIMEFRAME_MS[tf]
        target_ts = self._latest_closed_ts(decision_ts, duration_ms)
        key = (market.exchange.lower(), market.ticker, tf)
        with self._lock:
            loaded_until = self._loaded_until.get(key)
            if loaded_until is not None and loaded_until >= target_ts:
                return 0
            start_ts = (
                target_ts - (self.history_size - 1) * duration_ms
                if loaded_until is None
                else loaded_until + duration_ms
            )
            rows = (
                pl.scan_parquet(self.path)
                .filter(
                    (pl.col("exchange").str.to_lowercase() == key[0])
                    & (pl.col("symbol") == key[1])
                    & (pl.col("tf") == tf)
                    & (pl.col("ts") >= start_ts)
                    & (pl.col("ts") <= target_ts)
                    & (pl.col("ts") + duration_ms <= decision_ts)
                )
                .select(
                    "exchange",
                    "symbol",
                    "tf",
                    "ts",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                )
                .sort("ts")
                .collect()
            )
            loaded = 0
            for row in rows.iter_rows(named=True):
                result = self.store.ingest_ohlcv(
                    OHLCVCandle(
                        ticker=str(row["symbol"]),
                        exchange=str(row["exchange"]).lower(),
                        tf=str(row["tf"]),
                        ts=int(row["ts"]),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                        is_closed=True,
                    )
                )
                loaded += int(result.accepted)
            self._loaded_until[key] = target_ts
            self._rows_loaded += loaded
            return loaded

    def ensure_pair(self, pair: PairDefinition, decision_ts: int) -> int:
        """Догрузить причинно доступную историю обеих ног пары."""

        loaded = 0
        for market in (pair.leg1, pair.leg2):
            for tf in TIMEFRAME_MS:
                try:
                    loaded += self._ensure_market(market, tf, decision_ts)
                except Exception:
                    with self._lock:
                        self._errors += 1
                    LOGGER.exception(
                        "OHLCV parquet load failed exchange=%s ticker=%s tf=%s",
                        market.exchange,
                        market.ticker,
                        tf,
                    )
        return loaded

    def status(self) -> dict[str, Any]:
        """Вернуть объём загруженной parquet-истории и ошибки."""

        with self._lock:
            return {
                "mode": "parquet",
                "path": str(self.path),
                "series_loaded": len(self._loaded_until),
                "rows_loaded": self._rows_loaded,
                "errors": self._errors,
            }

    def close(self) -> None:
        return None


class CCXTOHLCVProvider:
    """Live-загрузка через CCXT с отдельным кэшем каждой серии."""

    def __init__(
        self,
        store: MarketStateStore,
        history_size: int = 128,
        timeout_ms: int = 10_000,
    ):
        try:
            import ccxt
        except ImportError as error:
            raise RuntimeError("ccxt mode requires the ccxt package") from error
        self.ccxt = ccxt
        self.store = store
        self.history_size = int(history_size)
        self.timeout_ms = int(timeout_ms)
        self._lock = threading.RLock()
        self._exchanges: dict[str, Any] = {}
        self._exchange_locks: dict[str, threading.Lock] = {}
        self._loaded_until: dict[tuple[str, str, str], int] = {}
        self._rows_loaded = 0
        self._errors = 0

    def _exchange(self, name: str) -> tuple[Any, threading.Lock]:
        key = name.lower()
        with self._lock:
            if key not in self._exchanges:
                exchange_class = getattr(self.ccxt, key)
                exchange = exchange_class(
                    {
                        "enableRateLimit": True,
                        "timeout": self.timeout_ms,
                    }
                )
                exchange.load_markets()
                self._exchanges[key] = exchange
                self._exchange_locks[key] = threading.Lock()
            return self._exchanges[key], self._exchange_locks[key]

    def _ensure_market(
        self,
        market: MarketRef,
        tf: str,
        decision_ts: int,
    ) -> int:
        duration_ms = TIMEFRAME_MS[tf]
        target_ts = (decision_ts - duration_ms) // duration_ms * duration_ms
        key = (market.exchange.lower(), market.ticker, tf)
        with self._lock:
            loaded_until = self._loaded_until.get(key)
            if loaded_until is not None and loaded_until >= target_ts:
                return 0
            since = (
                target_ts - (self.history_size - 1) * duration_ms
                if loaded_until is None
                else loaded_until + duration_ms
            )
        exchange, exchange_lock = self._exchange(market.exchange)
        with exchange_lock:
            values = exchange.fetch_ohlcv(
                market.ticker,
                tf,
                since=since,
                limit=self.history_size,
            )
        loaded = 0
        for value in values:
            ts, open_, high, low, close, volume = value[:6]
            if int(ts) + duration_ms > decision_ts or int(ts) > target_ts:
                continue
            result = self.store.ingest_ohlcv(
                OHLCVCandle(
                    ticker=market.ticker,
                    exchange=market.exchange.lower(),
                    tf=tf,
                    ts=int(ts),
                    open=float(open_),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=float(volume),
                    is_closed=True,
                )
            )
            loaded += int(result.accepted)
        with self._lock:
            self._loaded_until[key] = target_ts
            self._rows_loaded += loaded
        return loaded

    def ensure_pair(self, pair: PairDefinition, decision_ts: int) -> int:
        """Обновить OHLCV обеих ног до последней закрытой свечи."""

        loaded = 0
        for market in (pair.leg1, pair.leg2):
            for tf in TIMEFRAME_MS:
                try:
                    loaded += self._ensure_market(market, tf, decision_ts)
                except Exception:
                    with self._lock:
                        self._errors += 1
                    LOGGER.exception(
                        "CCXT OHLCV load failed exchange=%s ticker=%s tf=%s",
                        market.exchange,
                        market.ticker,
                        tf,
                    )
        return loaded

    def status(self) -> dict[str, Any]:
        """Вернуть состояние CCXT-кэша и количество ошибок."""

        with self._lock:
            return {
                "mode": "ccxt",
                "series_loaded": len(self._loaded_until),
                "rows_loaded": self._rows_loaded,
                "exchanges": sorted(self._exchanges),
                "errors": self._errors,
            }

    def close(self) -> None:
        with self._lock:
            for exchange in self._exchanges.values():
                close = getattr(exchange, "close", None)
                if callable(close):
                    close()
            self._exchanges.clear()
            self._exchange_locks.clear()


def build_ohlcv_provider(
    store: MarketStateStore,
    service_config: dict[str, Any],
) -> OHLCVProvider:
    """Создать provider из service config и переменных окружения."""

    config = service_config.get("ohlcv_provider", {})
    mode = os.getenv("ML_OHLCV_MODE", str(config.get("mode", "disabled"))).lower()
    history_size = int(
        os.getenv(
            "ML_OHLCV_FETCH_HISTORY_SIZE",
            str(config.get("history_size", service_config.get("ohlcv_history_size", 128))),
        )
    )
    if mode == "disabled":
        return DisabledOHLCVProvider()
    if mode == "parquet":
        configured_path = os.getenv(
            "ML_OHLCV_PATH",
            str(config.get("path", "raw_last_3d/ohlcv_raw.parquet")),
        )
        return ParquetOHLCVProvider(
            store=store,
            path=Path(configured_path),
            history_size=history_size,
        )
    if mode == "ccxt":
        return CCXTOHLCVProvider(
            store=store,
            history_size=history_size,
            timeout_ms=int(config.get("timeout_ms", 10_000)),
        )
    raise ValueError(f"Unsupported OHLCV provider mode: {mode}")
