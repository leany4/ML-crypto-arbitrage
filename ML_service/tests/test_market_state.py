from __future__ import annotations

from ml_service.schemas import MarketRef, PairDefinition
from ml_service.state.market import MarketStateStore

from conftest import make_book


def test_machine_ts_final_does_not_make_a_new_book() -> None:
    store = MarketStateStore()
    pair = PairDefinition(
        pair_id="BTC",
        base_ticker="BTC/USDT",
        pair_type="perp_perp_cross_exchange",
        leg1=MarketRef(exchange="a", ticker="BTC/USDT:USDT", is_perp=True),
        leg2=MarketRef(exchange="b", ticker="BTC/USDT:USDT", is_perp=True),
    )
    store.register_pair(pair)
    first = store.ingest_l2(make_book("a", "BTC/USDT:USDT", 1000, 99.0, 100.0))
    second = store.ingest_l2(make_book("a", "BTC/USDT:USDT", 1100, 99.0, 100.0))

    assert first.accepted and not first.duplicate_book
    assert second.accepted and second.duplicate_book
    assert second.affected_pairs == ("BTC",)
    assert store.latest_l2("a", "BTC/USDT:USDT").machine_ts_final == 1100
    assert store.state_counts()["changed_l2_rows"] == 1

