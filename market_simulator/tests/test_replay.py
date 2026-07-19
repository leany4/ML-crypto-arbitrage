from __future__ import annotations

import json
import time

import pyarrow as pa
import pyarrow.parquet as pq

from market_simulator.config import SimulatorSettings
from market_simulator.replay import ReplayController


class FakeResponse:
    def __init__(self, payload=None):
        self.payload = payload or {}

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


class FakeClient:
    puts = []
    posts = []

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def put(self, url, json):
        self.puts.append((url, json))
        return FakeResponse()

    def post(self, url, json):
        self.posts.append((url, json))
        if "candles" in json:
            return FakeResponse(
                {
                    "accepted": len(json["candles"]),
                    "duplicates": 0,
                    "scheduled_pairs": 1,
                }
            )
        return FakeResponse(
            {
                "accepted": len(json["snapshots"]),
                "duplicate_books": 0,
                "scheduled_pairs": 1,
            }
        )


def test_replay_groups_market_updates_by_100ms(tmp_path, monkeypatch) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    rows = []
    for exchange, timestamp in (("a", 100), ("a", 150), ("b", 180), ("a", 201)):
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
    pq.write_table(
        pa.Table.from_pylist(rows),
        prepared / "l2_selected_sorted.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "exchange": "a",
                    "symbol": "BTC/USDT:USDT",
                    "tf": "1m",
                    "ts": -60_000,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 10.0,
                    "available_ts": 0,
                }
            ]
        ),
        prepared / "ohlcv_selected_sorted.parquet",
    )
    pair = {
        "pair_id": "BTC",
        "base_ticker": "BTC/USDT",
        "pair_type": "perp_perp_cross_exchange",
        "leg1": {"exchange": "a", "ticker": "BTC/USDT:USDT", "is_perp": True},
        "leg2": {"exchange": "b", "ticker": "BTC/USDT:USDT", "is_perp": True},
        "direction_code": 0,
        "enabled": True,
    }
    (prepared / "pairs.json").write_text(json.dumps([pair]))
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "min_ts": 100,
                "max_ts": 201,
                "ohlcv_path": "/source/ohlcv.parquet",
            }
        )
    )

    FakeClient.puts = []
    FakeClient.posts = []
    monkeypatch.setattr("market_simulator.replay.httpx.Client", FakeClient)
    settings = SimulatorSettings(
        data_dir=tmp_path,
        prepared_dir=prepared,
        ml_url="http://ml",
        batch_endpoint="/v1/market/l2/batch",
        request_timeout_seconds=1,
        default_speed=1,
        default_duration_seconds=1,
    )
    replay = ReplayController(settings)
    replay.start(speed=1_000_000, start_ts=100, end_ts=300, duration_seconds=None)
    for _ in range(100):
        if replay.status()["state"] in {"completed", "failed"}:
            break
        time.sleep(0.01)

    status = replay.status()
    assert status["state"] == "completed"
    assert status["batches_sent"] == 2
    assert status["snapshots_sent"] == 3
    assert status["ohlcv_candles_sent"] == 1
    assert status["accepted_ohlcv_candles"] == 1
    assert len(FakeClient.puts) == 1
    l2_posts = [payload for _, payload in FakeClient.posts if "snapshots" in payload]
    ohlcv_posts = [payload for _, payload in FakeClient.posts if "candles" in payload]
    assert len(l2_posts[0]["snapshots"]) == 2
    assert l2_posts[0]["snapshots"][0]["machine_ts_final"] == 150
    assert ohlcv_posts[0]["candles"][0]["ts"] == -60_000
