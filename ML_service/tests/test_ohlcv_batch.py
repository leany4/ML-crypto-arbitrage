from __future__ import annotations

from dataclasses import dataclass

from ml_service.main import ingest_ohlcv_batch
from ml_service.schemas import (
    MarketRef,
    OHLCVBatchRequest,
    OHLCVCandle,
    PairDefinition,
)
from ml_service.state.market import MarketStateStore


@dataclass
class CoordinatorStub:
    scheduled: list[str] | None = None

    def schedule(self, pair_ids) -> None:
        self.scheduled = list(pair_ids)


@dataclass
class ContextStub:
    store: MarketStateStore
    coordinator: CoordinatorStub


def test_ohlcv_batch_is_causal_and_schedules_pair_once() -> None:
    store = MarketStateStore()
    store.register_pair(
        PairDefinition(
            pair_id="BTC",
            base_ticker="BTC/USDT",
            pair_type="perp_perp_cross_exchange",
            leg1=MarketRef(
                exchange="a", ticker="BTC/USDT:USDT", is_perp=True
            ),
            leg2=MarketRef(
                exchange="b", ticker="BTC/USDT:USDT", is_perp=True
            ),
        )
    )
    context = ContextStub(store=store, coordinator=CoordinatorStub())
    batch = OHLCVBatchRequest(
        candles=[
            OHLCVCandle(
                exchange=exchange,
                ticker="BTC/USDT:USDT",
                tf="5m",
                ts=300_000,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=10.0,
            )
            for exchange in ("a", "b")
        ]
    )

    result = ingest_ohlcv_batch(batch, context)  # type: ignore[arg-type]

    assert result["received"] == 2
    assert result["accepted"] == 2
    assert result["scheduled_pairs"] == 1
    assert result["watermark_ts"] == 600_000
    assert context.coordinator.scheduled == ["BTC"]
    assert store.candles("a", "BTC/USDT:USDT", "5m", 599_999) == []
    assert len(store.candles("a", "BTC/USDT:USDT", "5m", 600_000)) == 1
