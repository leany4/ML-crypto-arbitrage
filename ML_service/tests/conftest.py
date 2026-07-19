from __future__ import annotations

from ml_service.schemas import L2Snapshot, PriceLevel


def make_book(
    exchange: str,
    ticker: str,
    timestamp: int,
    bid: float,
    ask: float,
    is_perp: bool = True,
) -> L2Snapshot:
    return L2Snapshot(
        ticker=ticker,
        exchange=exchange,
        exchange_ts=timestamp,
        machine_ts=timestamp,
        machine_ts_final=timestamp,
        is_perp=is_perp,
        bids=[
            PriceLevel(price=bid - index * 0.01, volume=1000.0)
            for index in range(5)
        ],
        asks=[
            PriceLevel(price=ask + index * 0.01, volume=1000.0)
            for index in range(5)
        ],
    )

