from __future__ import annotations

from ml_service.paper.repository import InMemoryPaperRepository


def test_memory_repository_is_bounded_and_keeps_events() -> None:
    repository = InMemoryPaperRepository(
        max_trades=2,
        max_decisions=3,
        max_events=2,
    )
    for index, action in enumerate(
        ("wait", "schedule_entry", "hold", "schedule_exit", "wait")
    ):
        repository.add_decision(
            decision_ts=1_000 + index * 100,
            strategy="test",
            pair_id="BTC",
            state="flat",
            action=action,
            reason=action,
            predictions={
                "q35": {
                    "watch_q35_bps": 50.0 + index,
                    "_latency_ms": 0.2,
                }
            },
            features={"entry_edge_top1_bps": 40.0 + index},
        )

    assert repository.status()["mode"] == "memory"
    assert repository.status()["decisions"] == 3
    assert [row["action"] for row in repository.events()] == [
        "schedule_exit",
        "schedule_entry",
    ]
    assert repository.decisions(limit=10)[0]["decision_ts"] == 1_400


def test_memory_repository_calculates_session_stats() -> None:
    repository = InMemoryPaperRepository(max_trades=2)
    for index, pnl in enumerate((10.0, -5.0, 20.0)):
        repository.add_trade(
            {
                "strategy": "test",
                "pair_id": "BTC",
                "opened_ts": index * 1_000,
                "closed_ts": index * 1_000 + 500,
                "hold_ms": 500,
                "net_pnl_bps": pnl,
            }
        )

    trades = repository.trades()
    assert [trade["net_pnl_bps"] for trade in trades] == [20.0, -5.0]
    stats = repository.stats()[0]
    assert stats["trades"] == 2
    assert stats["total_net_pnl_bps"] == 15.0
    assert stats["win_rate"] == 0.5
