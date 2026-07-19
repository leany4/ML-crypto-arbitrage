from __future__ import annotations

from pathlib import Path

from ml_service.config import StrategySettings
from ml_service.paper.engine import PaperEngine
from ml_service.paper.repository import PaperRepository
from ml_service.preprocessing.features import PairSnapshot
from ml_service.schemas import MarketRef, PairDefinition

from conftest import make_book


def snapshot(pair: PairDefinition, timestamp: int, long_bid: float, short_bid: float):
    long_book = make_book("a", "BTC", timestamp, long_bid, long_bid + 0.01)
    short_book = make_book("b", "BTC", timestamp, short_bid, short_bid + 0.01)
    return PairSnapshot(
        pair=pair,
        decision_ts=timestamp,
        grid_ts=timestamp,
        long_book=long_book,
        short_book=short_book,
        features={
            "entry_edge_top1_bps": (
                short_book.bids[0].price / long_book.asks[0].price - 1
            )
            * 10_000,
            "current_entry_fill_share": 1.0,
        },
    )


def test_paper_round_trip_uses_four_fees(tmp_path: Path) -> None:
    repository = PaperRepository(tmp_path / "paper.sqlite3")
    strategy = StrategySettings(
        name="test",
        enabled=True,
        values={
            "opportunity_source": "q35.watch_q35_bps",
            "opportunity_min_bps": 50.0,
            "max_hold_ms": 100,
            "execution_delay_ms": 100,
            "notional_usd": 100.0,
        },
    )
    engine = PaperEngine(repository, {"test": strategy})
    pair = PairDefinition(
        pair_id="BTC",
        base_ticker="BTC",
        pair_type="perp_perp_cross_exchange",
        leg1=MarketRef(exchange="a", ticker="BTC", is_perp=True),
        leg2=MarketRef(exchange="b", ticker="BTC", is_perp=True),
    )
    predictions = {"q35": {"watch_q35_bps": 100.0}}

    engine.evaluate("test", snapshot(pair, 1000, 100.0, 101.0), predictions)
    engine.evaluate("test", snapshot(pair, 1100, 100.0, 101.0), predictions)
    positions = engine.positions()
    assert len(positions) == 1
    assert positions[0]["opened_ts"] == 1000
    assert positions[0]["entry_fill_ts"] == 1100
    assert abs(
        positions[0]["long_cost"] / positions[0]["entry_long_price"]
        - positions[0]["short_proceeds"] / positions[0]["entry_short_price"]
    ) < 1e-12
    engine.evaluate("test", snapshot(pair, 1200, 100.0, 101.0), predictions)
    engine.evaluate("test", snapshot(pair, 1300, 100.0, 101.0), predictions)

    trades = repository.trades()
    assert len(trades) == 1
    assert 39.0 <= trades[0]["fee_bps"] <= 41.0
    assert abs(
        trades[0]["net_pnl_bps"]
        - (trades[0]["gross_pnl_bps"] - trades[0]["fee_bps"])
    ) < 1e-9
    stats = repository.stats()
    assert stats[0]["trades"] == 1
    assert stats[0]["median_trade_pnl_bps"] == trades[0]["net_pnl_bps"]
    assert stats[0]["max_drawdown_bps"] <= 0.0
    repository.close()


def test_rl_actions_and_fill_order_match_training_contract(
    tmp_path: Path,
) -> None:
    repository = PaperRepository(tmp_path / "rl-paper.sqlite3")
    strategy = StrategySettings(
        name="rl",
        enabled=True,
        values={
            "action_source": "agent.action",
            "max_hold_ms": 300_000,
            "execution_delay_ms": 100,
            "notional_usd": 100.0,
            "cooldown_steps": 100,
        },
    )
    engine = PaperEngine(repository, {"rl": strategy})
    pair = PairDefinition(
        pair_id="BTC",
        base_ticker="BTC",
        pair_type="perp_perp_cross_exchange",
        leg1=MarketRef(exchange="a", ticker="BTC", is_perp=True),
        leg2=MarketRef(exchange="b", ticker="BTC", is_perp=True),
    )

    at_1000 = snapshot(pair, 1000, 100.0, 101.0)
    engine.evaluate("rl", at_1000, {"agent": {"action": 1}})
    assert engine.position("rl", pair.pair_id) is None

    at_1100 = snapshot(pair, 1100, 100.0, 101.0)
    engine.advance("rl", at_1100)
    state = engine.rl_position_state("rl", at_1100)
    assert state[0] == 1.0
    assert abs(state[4] - (1.0 / 3000.0)) < 1e-12

    engine.evaluate("rl", at_1100, {"agent": {"action": 2}})
    assert engine.position("rl", pair.pair_id) is not None

    at_1200 = snapshot(pair, 1200, 100.0, 101.0)
    engine.evaluate("rl", at_1200, {"agent": {"action": 3}})
    assert engine.position("rl", pair.pair_id) is not None

    at_1300 = snapshot(pair, 1300, 100.0, 101.0)
    engine.advance("rl", at_1300)
    assert engine.position("rl", pair.pair_id) is None
    flat_state = engine.rl_position_state("rl", at_1300)
    assert abs(flat_state[6] - 0.99) < 1e-12

    trades = repository.trades()
    assert len(trades) == 1
    assert trades[0]["opened_ts"] == 1000
    assert trades[0]["entry_fill_ts"] == 1100
    assert trades[0]["exit_decision_ts"] == 1200
    assert trades[0]["closed_ts"] == 1300
    assert trades[0]["hold_ms"] == 200
    repository.close()
