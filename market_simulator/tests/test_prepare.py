from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from market_simulator.prepare import prepare_replay


def test_prepare_selects_pairs_and_sorts_l2(tmp_path) -> None:
    raw_path = tmp_path / "l2.parquet"
    rows = []
    for exchange in ("a", "b"):
        for timestamp in (300, 100, 200):
            row = {
                "ticker": "BTC/USDT:USDT",
                "exchange": exchange,
                "exchange_ts": timestamp,
                "machine_ts": timestamp,
                "machine_ts_final": timestamp,
            }
            for level in range(1, 6):
                row[f"bid_price_{level}"] = 100.0 - level
                row[f"bid_vol_{level}"] = 10.0
                row[f"ask_price_{level}"] = 100.0 + level
                row[f"ask_vol_{level}"] = 10.0
            rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), raw_path)
    ohlcv_path = tmp_path / "ohlcv.parquet"
    ohlcv_rows = []
    for exchange in ("a", "b"):
        for tf, duration in (("1m", 60_000), ("5m", 300_000)):
            for index in range(4):
                ohlcv_rows.append(
                    {
                        "exchange": exchange,
                        "symbol": "BTC/USDT:USDT",
                        "tf": tf,
                        "ts": index * duration,
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.5,
                        "volume": 10.0,
                    }
                )
    pq.write_table(pa.Table.from_pylist(ohlcv_rows), ohlcv_path)

    output = tmp_path / "prepared"
    manifest = prepare_replay(
        raw_path,
        ohlcv_path,
        output,
        max_pairs=2,
        min_ohlcv_rows_per_tf=4,
        min_concurrent_seconds=1,
    )

    pairs = json.loads((output / "pairs.json").read_text())
    table = pq.read_table(output / "l2_selected_sorted.parquet")
    ohlcv_table = pq.read_table(output / "ohlcv_selected_sorted.parquet")
    timestamps = table.column("machine_ts_final").to_pylist()
    available_timestamps = ohlcv_table.column("available_ts").to_pylist()

    assert manifest["pairs"] == 2
    assert len(pairs) == 2
    assert {pair["direction_code"] for pair in pairs} == {0, 1}
    assert timestamps == sorted(timestamps)
    assert available_timestamps == sorted(available_timestamps)
    assert manifest["ohlcv_is_causal"] is True
    assert manifest["ohlcv_rows"] == 16
    assert manifest["replay_start_ts"] == 100
    assert manifest["replay_end_ts"] == 300
    assert manifest["selected_pairs_share_replay_window"] is True
    assert manifest["selected_min_concurrent_seconds"] == 1


def test_prepare_applies_pair_limit_per_enabled_type(tmp_path) -> None:
    raw_path = tmp_path / "l2.parquet"
    rows = []
    markets = (
        ("a", "BTC/USDT"),
        ("a", "BTC/USDT:USDT"),
        ("b", "BTC/USDT:USDT"),
    )
    for exchange, ticker in markets:
        for timestamp in (100, 200):
            row = {
                "ticker": ticker,
                "exchange": exchange,
                "exchange_ts": timestamp,
                "machine_ts": timestamp,
                "machine_ts_final": timestamp,
            }
            for level in range(1, 6):
                row[f"bid_price_{level}"] = 100.0 - level
                row[f"bid_vol_{level}"] = 10.0
                row[f"ask_price_{level}"] = 100.0 + level
                row[f"ask_vol_{level}"] = 10.0
            rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), raw_path)

    ohlcv_path = tmp_path / "ohlcv.parquet"
    ohlcv_rows = []
    for exchange, ticker in markets:
        for tf, duration in (("1m", 60_000), ("5m", 300_000)):
            for index in range(2):
                ohlcv_rows.append(
                    {
                        "exchange": exchange,
                        "symbol": ticker,
                        "tf": tf,
                        "ts": index * duration,
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.5,
                        "volume": 10.0,
                    }
                )
    pq.write_table(pa.Table.from_pylist(ohlcv_rows), ohlcv_path)

    output = tmp_path / "prepared"
    manifest = prepare_replay(
        raw_path,
        ohlcv_path,
        output,
        max_pairs=1,
        include_spot_perp=True,
        min_ohlcv_rows_per_tf=2,
        min_concurrent_seconds=1,
    )
    pairs = json.loads((output / "pairs.json").read_text())

    assert manifest["pairs"] == 3
    assert sum(
        pair["pair_type"] == "perp_perp_cross_exchange" for pair in pairs
    ) == 2
    assert {pair["pair_type"] for pair in pairs} == {
        "perp_perp_cross_exchange",
        "spot_perp_same_exchange",
    }


def test_prepare_rejects_markets_with_only_range_overlap(tmp_path) -> None:
    raw_path = tmp_path / "l2.parquet"
    rows = []
    timestamps_by_exchange = {
        "a": (100, 1_100, 10_100),
        "b": (5_100, 6_100, 10_100),
    }
    for exchange, timestamps in timestamps_by_exchange.items():
        for timestamp in timestamps:
            row = {
                "ticker": "BTC/USDT:USDT",
                "exchange": exchange,
                "exchange_ts": timestamp,
                "machine_ts": timestamp,
                "machine_ts_final": timestamp,
            }
            for level in range(1, 6):
                row[f"bid_price_{level}"] = 100.0 - level
                row[f"bid_vol_{level}"] = 10.0
                row[f"ask_price_{level}"] = 100.0 + level
                row[f"ask_vol_{level}"] = 10.0
            rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), raw_path)

    with pytest.raises(RuntimeError, match="continuous concurrent L2"):
        prepare_replay(
            raw_path,
            None,
            tmp_path / "prepared",
            max_pairs=1,
            min_concurrent_seconds=2,
        )
