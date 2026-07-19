from __future__ import annotations

from ml_service.preprocessing.features import PairFeatureEngine
from ml_service.schemas import MarketRef, OHLCVCandle, PairDefinition
from ml_service.state.market import MarketStateStore

from conftest import make_book


def test_q35_features_are_causal_and_complete() -> None:
    store = MarketStateStore()
    pair = PairDefinition(
        pair_id="BTC",
        base_ticker="BTC/USDT",
        pair_type="perp_perp_cross_exchange",
        leg1=MarketRef(exchange="a", ticker="BTC/USDT:USDT", is_perp=True),
        leg2=MarketRef(exchange="b", ticker="BTC/USDT:USDT", is_perp=True),
    )
    store.register_pair(pair)

    for exchange, close in (("a", 100.0), ("b", 101.0)):
        for index in range(55):
            ts = 1_000_000 + index * 300_000
            store.ingest_ohlcv(
                OHLCVCandle(
                    ticker="BTC/USDT:USDT",
                    exchange=exchange,
                    tf="5m",
                    ts=ts,
                    open=close,
                    high=close * 1.01,
                    low=close * 0.99,
                    close=close,
                    volume=10.0,
                )
            )
        for index in range(7):
            ts = 17_500_000 + index * 60_000
            store.ingest_ohlcv(
                OHLCVCandle(
                    ticker="BTC/USDT:USDT",
                    exchange=exchange,
                    tf="1m",
                    ts=ts,
                    open=close,
                    high=close * 1.01,
                    low=close * 0.99,
                    close=close,
                    volume=10.0,
                )
            )

    decision_ts = 18_300_000
    store.ingest_l2(make_book("a", "BTC/USDT:USDT", decision_ts, 99.9, 100.0))
    store.ingest_l2(make_book("b", "BTC/USDT:USDT", decision_ts, 101.0, 101.1))
    snapshot = PairFeatureEngine(store).build("BTC")

    assert snapshot is not None
    assert snapshot.features["leg1_ohlcv_available"] == 1.0
    assert snapshot.features["leg2_ohlcv_available"] == 1.0
    assert snapshot.features["pair_5m_has_both"] == 1.0
    assert snapshot.features["entry_edge_top1_bps"] > 0
    assert "edge_z_300s" in snapshot.features
    assert "directional_ret_gap_48_bars_bps" in snapshot.features
    assert "pair_1m_ret_5tf_bps_ratio_short_to_long" in snapshot.features
    rl_ohlcv_columns = {
        "leg1_ohlcv_log_volume",
        "leg2_ohlcv_log_volume",
        "leg1_ohlcv_volume_z_12",
        "leg2_ohlcv_volume_z_12",
        "directional_vol_gap_12_bars_bps",
    }
    assert not (rl_ohlcv_columns - set(snapshot.features))
    assert snapshot.features["leg1_ohlcv_log_volume"] == snapshot.features[
        "leg2_ohlcv_log_volume"
    ]
    assert snapshot.features["directional_vol_gap_12_bars_bps"] == 0.0

    l2_columns = [
        "entry_edge_top1_bps",
        "edge_change_1s_bps", "edge_change_5s_bps",
        "edge_change_30s_bps", "edge_change_120s_bps",
        "edge_mean_5s_bps", "edge_std_5s_bps",
        "edge_mean_30s_bps", "edge_std_30s_bps",
        "edge_mean_120s_bps", "edge_std_120s_bps",
        "edge_mean_300s_bps", "edge_std_300s_bps",
        "edge_min_30s_bps", "edge_max_30s_bps",
        "edge_min_120s_bps", "edge_max_120s_bps",
        "edge_min_300s_bps", "edge_max_300s_bps",
        "edge_z_30s", "edge_z_120s", "edge_z_300s",
        "distance_from_max_30s_bps", "distance_from_max_300s_bps",
        "distance_from_min_30s_bps", "distance_from_min_300s_bps",
        "candidate_starts_30s", "candidate_starts_300s",
        "candidate_active_share_30s", "candidate_active_share_300s",
        "edge_position_30s", "edge_position_120s", "edge_position_300s",
        "edge_slope_5s_bps_per_sec", "edge_slope_30s_bps_per_sec",
        "edge_slope_120s_bps_per_sec", "edge_acceleration_5s_vs_30s",
        "leg1_mid_return_1s_bps", "leg1_mid_return_5s_bps",
        "leg1_mid_return_30s_bps", "leg2_mid_return_1s_bps",
        "leg2_mid_return_5s_bps", "leg2_mid_return_30s_bps",
        "relative_mid_return_1s_bps", "relative_mid_return_5s_bps",
        "relative_mid_return_30s_bps",
        "leg1_imbalance_change_1s", "leg1_imbalance_change_5s",
        "leg2_imbalance_change_1s", "leg2_imbalance_change_5s",
        "leg1_book_spread_bps", "leg2_book_spread_bps",
        "leg1_imbalance", "leg2_imbalance",
        "leg1_log_bid_depth", "leg1_log_ask_depth",
        "leg2_log_bid_depth", "leg2_log_ask_depth",
        "leg1_book_age_sec", "leg2_book_age_sec", "pair_skew_sec",
        "context_valid_numeric", "execution_valid_numeric",
        "is_spot_perp", "direction_numeric",
    ]
    ohlcv_base = [
        *[f"ret_{bars}_bars_bps" for bars in [1, 3, 6, 12, 24, 48]],
        "range_bps", "body_bps", "upper_wick_bps", "lower_wick_bps",
        "close_location",
        *[f"rv_{bars}_bars_bps" for bars in [6, 12, 24, 48]],
        "atr_14_bps",
        *[f"close_vs_mean_{bars}_bps" for bars in [6, 12, 24, 48]],
        "log_quote_volume", "quote_volume_z_12", "quote_volume_z_48",
        "log_quote_volume_change_1", "zero_volume", "rsi_14",
        "bb_width_20_bps",
    ]
    ohlcv_columns = [
        *[f"leg1_ohlcv_{name}" for name in ohlcv_base],
        *[f"leg2_ohlcv_{name}" for name in ohlcv_base],
        "leg1_ohlcv_age_sec", "leg2_ohlcv_age_sec",
        "leg1_ohlcv_available", "leg2_ohlcv_available",
        *[f"directional_ret_gap_{bars}_bars_bps" for bars in [1, 3, 6, 12, 24, 48]],
        *[f"absolute_ret_gap_{bars}_bars_bps" for bars in [3, 12, 48]],
        "directional_body_gap_bps",
        *[f"directional_trend_gap_{bars}_bps" for bars in [6, 24, 48]],
        *[f"mean_rv_{bars}_bars_bps" for bars in [12, 48]],
        *[f"absolute_rv_gap_{bars}_bars_bps" for bars in [12, 48]],
        "directional_rsi_gap", "quote_volume_z_gap_12",
        "quote_volume_z_gap_48", "log_quote_volume_ratio",
    ]
    assert len(l2_columns) == 65
    assert len(ohlcv_columns) == 79
    assert not (set(l2_columns + ohlcv_columns) - set(snapshot.features))


def test_unclosed_candle_is_not_ingested() -> None:
    store = MarketStateStore()
    result = store.ingest_ohlcv(
        OHLCVCandle(
            ticker="BTC",
            exchange="a",
            tf="5m",
            ts=1_000,
            open=100,
            high=101,
            low=99,
            close=100,
            volume=1,
            is_closed=False,
        )
    )
    assert not result.accepted


def test_pair_type_allowlist_blocks_spot_perp_before_features() -> None:
    store = MarketStateStore()
    pair = PairDefinition(
        pair_id="BTC-spot-perp",
        base_ticker="BTC/USDT",
        pair_type="spot_perp_same_exchange",
        leg1=MarketRef(exchange="gate", ticker="BTC/USDT", is_perp=False),
        leg2=MarketRef(
            exchange="gate", ticker="BTC/USDT:USDT", is_perp=True
        ),
    )
    store.register_pair(pair)
    store.ingest_l2(make_book("gate", "BTC/USDT", 1_000, 99.9, 100.0))
    store.ingest_l2(
        make_book("gate", "BTC/USDT:USDT", 1_000, 101.0, 101.1)
    )

    engine = PairFeatureEngine(
        store,
        allowed_pair_types=["perp_perp_cross_exchange"],
    )

    assert engine.build(pair.pair_id) is None
    assert engine.history(pair.pair_id) == []
