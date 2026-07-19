from __future__ import annotations

from dataclasses import dataclass

from ml_service.main import ingest_l2_batch
from ml_service.schemas import L2BatchRequest, MarketRef, PairDefinition
from ml_service.state.market import MarketStateStore

from conftest import make_book


@dataclass
class CoordinatorStub:
    scheduled: list[str] | None = None

    def schedule(self, pair_ids) -> None:
        self.scheduled = list(pair_ids)


@dataclass
class ContextStub:
    store: MarketStateStore
    coordinator: CoordinatorStub


def test_l2_batch_schedules_each_pair_once() -> None:
    store = MarketStateStore()
    store.register_pair(
        PairDefinition(
            pair_id="BTC",
            base_ticker="BTC/USDT",
            pair_type="perp_perp_cross_exchange",
            leg1=MarketRef(exchange="a", ticker="BTC/USDT:USDT", is_perp=True),
            leg2=MarketRef(exchange="b", ticker="BTC/USDT:USDT", is_perp=True),
        )
    )
    context = ContextStub(store=store, coordinator=CoordinatorStub())
    batch = L2BatchRequest(
        snapshots=[
            make_book("a", "BTC/USDT:USDT", 1_000, 99.0, 100.0),
            make_book("b", "BTC/USDT:USDT", 1_000, 101.0, 102.0),
        ]
    )

    result = ingest_l2_batch(batch, context)  # type: ignore[arg-type]

    assert result["received"] == 2
    assert result["accepted"] == 2
    assert result["scheduled_pairs"] == 1
    assert result["watermark_ts"] == 1_000
    assert context.coordinator.scheduled == ["BTC"]
