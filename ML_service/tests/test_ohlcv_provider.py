from __future__ import annotations

import polars as pl

from ml_service.ohlcv import ParquetOHLCVProvider
from ml_service.schemas import MarketRef, PairDefinition
from ml_service.state.market import MarketStateStore


def test_parquet_provider_loads_only_closed_candles(tmp_path) -> None:
    path = tmp_path / "ohlcv.parquet"
    rows = []
    for exchange, close in (("a", 100.0), ("b", 101.0)):
        for tf, duration in (("1m", 60_000), ("5m", 300_000)):
            for index in range(4):
                rows.append(
                    {
                        "exchange": exchange,
                        "symbol": "BTC/USDT:USDT",
                        "tf": tf,
                        "ts": index * duration,
                        "open": close,
                        "high": close + 1,
                        "low": close - 1,
                        "close": close,
                        "volume": 10.0,
                    }
                )
    pl.DataFrame(rows).write_parquet(path)

    store = MarketStateStore(ohlcv_history_size=128)
    pair = PairDefinition(
        pair_id="BTC",
        base_ticker="BTC/USDT",
        pair_type="perp_perp_cross_exchange",
        leg1=MarketRef(exchange="a", ticker="BTC/USDT:USDT", is_perp=True),
        leg2=MarketRef(exchange="b", ticker="BTC/USDT:USDT", is_perp=True),
    )
    provider = ParquetOHLCVProvider(store, path, history_size=128)

    decision_ts = 600_001
    loaded = provider.ensure_pair(pair, decision_ts)

    assert loaded > 0
    for exchange in ("a", "b"):
        one_minute = store.candles(
            exchange, "BTC/USDT:USDT", "1m", available_at_ms=decision_ts
        )
        five_minute = store.candles(
            exchange, "BTC/USDT:USDT", "5m", available_at_ms=decision_ts
        )
        assert [candle.ts for candle in one_minute] == [0, 60_000, 120_000, 180_000]
        assert [candle.ts for candle in five_minute] == [0, 300_000]
        assert all(candle.ts + 60_000 <= decision_ts for candle in one_minute)
        assert all(candle.ts + 300_000 <= decision_ts for candle in five_minute)

    assert provider.ensure_pair(pair, decision_ts) == 0
    assert provider.status()["series_loaded"] == 4
