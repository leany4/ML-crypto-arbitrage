from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from ml_service.predictors.base import PredictionContext
from ml_service.predictors.q35 import Q35Predictor
from ml_service.preprocessing.features import PairFeatureEngine
from ml_service.schemas import MarketRef, OHLCVCandle, PairDefinition
from ml_service.state.market import MarketStateStore

from conftest import make_book


@pytest.mark.integration
def test_real_q35_bundle_predicts_from_online_features() -> None:
    bundle_value = os.getenv("Q35_BUNDLE_DIR")
    if not bundle_value:
        pytest.skip("Q35_BUNDLE_DIR is not set")

    store = MarketStateStore()
    pair = PairDefinition(
        pair_id="BTC",
        base_ticker="BTC/USDT",
        pair_type="perp_perp_cross_exchange",
        leg1=MarketRef(exchange="a", ticker="BTC/USDT:USDT", is_perp=True),
        leg2=MarketRef(exchange="b", ticker="BTC/USDT:USDT", is_perp=True),
    )
    store.register_pair(pair)

    for exchange, price in (("a", 100.0), ("b", 101.0)):
        for index in range(60):
            ts = 1_000_000 + index * 300_000
            movement = 1.0 + index * 0.0001
            store.ingest_ohlcv(
                OHLCVCandle(
                    ticker="BTC/USDT:USDT",
                    exchange=exchange,
                    tf="5m",
                    ts=ts,
                    open=price * movement,
                    high=price * movement * 1.002,
                    low=price * movement * 0.998,
                    close=price * movement * 1.0005,
                    volume=100.0 + index,
                )
            )
        for index in range(10):
            ts = 19_000_000 + index * 60_000
            movement = 1.0 + index * 0.0002
            store.ingest_ohlcv(
                OHLCVCandle(
                    ticker="BTC/USDT:USDT",
                    exchange=exchange,
                    tf="1m",
                    ts=ts,
                    open=price * movement,
                    high=price * movement * 1.001,
                    low=price * movement * 0.999,
                    close=price * movement * 1.0003,
                    volume=50.0 + index,
                )
            )

    decision_ts = 20_000_000
    store.ingest_l2(make_book("a", "BTC/USDT:USDT", decision_ts, 99.9, 100.0))
    store.ingest_l2(make_book("b", "BTC/USDT:USDT", decision_ts, 101.0, 101.1))
    snapshot = PairFeatureEngine(store).build("BTC")
    assert snapshot is not None

    predictor = Q35Predictor("q35_perp", Path(bundle_value), "cpu")
    predictor.load()
    prediction = predictor.predict(PredictionContext(features=snapshot.features))
    expected_version = json.loads(
        (Path(bundle_value) / "manifest.json").read_text(encoding="utf-8")
    )["version"]

    assert len(predictor.feature_columns) == 98
    assert prediction.device == "cpu"
    assert prediction.model_version == expected_version
    assert math.isfinite(float(prediction.outputs["watch_q35_bps"]))
