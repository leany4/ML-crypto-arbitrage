"""Построение причинных признаков q35, Transformer и RL из L2/OHLCV."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import fmean, pstdev, stdev
from typing import Any

from ml_service.schemas import L2Snapshot, OHLCVCandle, PairDefinition
from ml_service.state.market import MarketStateStore


@dataclass(frozen=True)
class PairSnapshot:
    """Согласованный снимок двух ног и признаки одной точки решения."""

    pair: PairDefinition
    decision_ts: int
    grid_ts: int
    long_book: L2Snapshot
    short_book: L2Snapshot
    features: dict[str, Any]


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / (abs(denominator) + 1e-9)


def _depth_usd(levels: list[Any]) -> float:
    return sum(level.price * level.volume for level in levels)


def _imbalance(book: L2Snapshot) -> float:
    bid = _depth_usd(book.bids)
    ask = _depth_usd(book.asks)
    return (bid - ask) / (bid + ask + 1e-9)


def _book_spread_bps(book: L2Snapshot) -> float:
    mid = (book.bids[0].price + book.asks[0].price) / 2.0
    return (book.asks[0].price - book.bids[0].price) / mid * 10_000.0


def _fill_share(levels: list[Any], notional_usd: float) -> float:
    remaining = float(notional_usd)
    for level in levels:
        remaining -= level.price * level.volume
        if remaining <= 0:
            return 1.0
    return max(0.0, min(1.0, 1.0 - remaining / notional_usd))


def _vwap(levels: list[Any], notional_usd: float) -> tuple[float, float]:
    remaining = float(notional_usd)
    quote_total = 0.0
    base_total = 0.0
    for level in levels:
        quote = min(remaining, level.price * level.volume)
        if quote <= 0:
            continue
        quote_total += quote
        base_total += quote / level.price
        remaining -= quote
        if remaining <= 1e-9:
            break
    return (
        quote_total / base_total if base_total > 0 else 0.0,
        quote_total / notional_usd if notional_usd > 0 else 0.0,
    )


def _candle_values(candles: list[OHLCVCandle]) -> dict[str, float]:
    if not candles:
        return {
            "body_bps": math.nan,
            "range_bps": math.nan,
            "log_volume": math.nan,
            "ret_1tf_bps": math.nan,
            "ret_5tf_bps": math.nan,
            "range_mean_5tf_bps": math.nan,
            "log_volume_mean_5tf": math.nan,
            "has": 0.0,
        }

    current = candles[-1]
    ranges = [
        (item.high - item.low) / item.close * 10_000.0
        for item in candles[-5:]
    ]
    log_volumes = [math.log1p(item.volume) for item in candles[-5:]]
    ret_1 = (
        (current.close / candles[-2].close - 1.0) * 10_000.0
        if len(candles) >= 2
        else math.nan
    )
    ret_5 = (
        (current.close / candles[-6].close - 1.0) * 10_000.0
        if len(candles) >= 6
        else math.nan
    )
    return {
        "body_bps": (current.close - current.open) / current.open * 10_000.0,
        "range_bps": ranges[-1],
        "log_volume": log_volumes[-1],
        "ret_1tf_bps": ret_1,
        "ret_5tf_bps": ret_5,
        "range_mean_5tf_bps": fmean(ranges) if len(ranges) >= 2 else math.nan,
        "log_volume_mean_5tf": (
            fmean(log_volumes) if len(log_volumes) >= 2 else math.nan
        ),
        "has": 1.0,
    }


class PairFeatureEngine:
    """Формирует общую 100-мс историю без доступа к будущим данным."""

    Q35_OHLCV_VALUES = (
        "body_bps",
        "range_bps",
        "log_volume",
        "ret_1tf_bps",
        "ret_5tf_bps",
        "range_mean_5tf_bps",
        "log_volume_mean_5tf",
    )
    Q35_ADDITIVE_VALUES = {
        "range_bps",
        "log_volume",
        "range_mean_5tf_bps",
        "log_volume_mean_5tf",
    }

    def __init__(
        self,
        store: MarketStateStore,
        history_size: int = 4096,
        decision_ms: int = 100,
        max_book_age_ms: int = 1500,
        max_pair_skew_ms: int = 500,
        notional_usd: float = 100.0,
        fee_bps_per_leg_side: float = 10.0,
        allowed_pair_types: list[str] | tuple[str, ...] | None = None,
    ):
        self.store = store
        self.history_size = int(history_size)
        self.decision_ms = int(decision_ms)
        self.max_book_age_ms = int(max_book_age_ms)
        self.max_pair_skew_ms = int(max_pair_skew_ms)
        self.notional_usd = float(notional_usd)
        self.fee_bps_per_leg_side = float(fee_bps_per_leg_side)
        self.allowed_pair_types = (
            None
            if allowed_pair_types is None
            else frozenset(str(value) for value in allowed_pair_types)
        )
        self._history: dict[str, deque[PairSnapshot]] = {}

    def build(self, pair_id: str) -> PairSnapshot | None:
        """Построить следующий валидный снимок пары или вернуть `None`."""

        pair = self.store.get_pair(pair_id)
        if pair is None or not pair.enabled:
            return None
        if (
            self.allowed_pair_types is not None
            and pair.pair_type not in self.allowed_pair_types
        ):
            return None
        leg1 = self.store.latest_l2(pair.leg1.exchange, pair.leg1.ticker)
        leg2 = self.store.latest_l2(pair.leg2.exchange, pair.leg2.ticker)
        if leg1 is None or leg2 is None:
            return None

        decision_ts = max(leg1.machine_ts_final, leg2.machine_ts_final)
        age1 = decision_ts - leg1.machine_ts_final
        age2 = decision_ts - leg2.machine_ts_final
        skew = abs(leg1.machine_ts_final - leg2.machine_ts_final)
        if max(age1, age2) > self.max_book_age_ms or skew > self.max_pair_skew_ms:
            return None

        long_book, short_book = (leg1, leg2) if pair.direction_code == 0 else (leg2, leg1)
        edge = (short_book.bids[0].price / long_book.asks[0].price - 1.0) * 10_000.0
        grid_ts = decision_ts // self.decision_ms * self.decision_ms
        history = self._history.setdefault(
            pair_id, deque(maxlen=self.history_size)
        )
        past_edges = [float(item.features["entry_edge_top1_bps"]) for item in history]

        def change(seconds: int) -> float:
            steps = max(1, seconds * 1000 // self.decision_ms)
            return edge - past_edges[-steps] if len(past_edges) >= steps else 0.0

        def rolling(seconds: int) -> tuple[float, float]:
            steps = max(1, seconds * 1000 // self.decision_ms)
            values = [*past_edges[-(steps - 1) :], edge] if steps > 1 else [edge]
            return fmean(values), pstdev(values) if len(values) > 1 else 0.0

        long_entry_price, long_fill = _vwap(long_book.asks, self.notional_usd)
        short_entry_price, short_fill = _vwap(short_book.bids, self.notional_usd)
        long_exit_price, long_exit_fill = _vwap(long_book.bids, self.notional_usd)
        short_exit_price, short_exit_fill = _vwap(short_book.asks, self.notional_usd)
        fill_share = min(long_fill, short_fill)
        vwap_edge = (
            (short_entry_price / long_entry_price - 1.0) * 10_000.0
            if long_entry_price > 0
            else -200.0
        )
        instant_roundtrip = (
            (
                long_exit_price / long_entry_price
                + short_entry_price / short_exit_price
                - 2.0
            )
            * 10_000.0
            - 4.0 * self.fee_bps_per_leg_side
            if min(long_entry_price, short_entry_price, long_exit_price, short_exit_price)
            > 0
            else -200.0
        )

        features: dict[str, Any] = {
            "entry_edge_top1_bps": edge,
            "edge_change_1s_bps": change(1),
            "edge_change_5s_bps": change(5),
            "edge_change_30s_bps": change(30),
            "edge_change_120s_bps": change(120),
            "leg1_book_spread_bps": _book_spread_bps(leg1),
            "leg2_book_spread_bps": _book_spread_bps(leg2),
            "leg1_imbalance": _imbalance(leg1),
            "leg2_imbalance": _imbalance(leg2),
            "leg1_log_bid_depth": math.log1p(_depth_usd(leg1.bids)),
            "leg1_log_ask_depth": math.log1p(_depth_usd(leg1.asks)),
            "leg2_log_bid_depth": math.log1p(_depth_usd(leg2.bids)),
            "leg2_log_ask_depth": math.log1p(_depth_usd(leg2.asks)),
            "leg1_book_age_sec": age1 / 1000.0,
            "leg2_book_age_sec": age2 / 1000.0,
            "pair_skew_sec": skew / 1000.0,
            "is_spot_perp": float(pair.pair_type == "spot_perp_same_exchange"),
            "direction_numeric": float(pair.direction_code),
            "current_entry_fill_share": fill_share,
            "current_entry_executable": 0.0,
            "current_open_gross_edge_bps": vwap_edge,
            "current_open_edge_after_entry_fee_bps": (
                vwap_edge - 2.0 * self.fee_bps_per_leg_side
            ),
            "current_entry_slippage_bps": edge - vwap_edge,
            "current_instant_roundtrip_pnl_bps": instant_roundtrip,
            "pair_type": pair.pair_type,
            "pair_id": pair.pair_id,
            "direction_code": pair.direction_code,
            "leg1_exchange": pair.leg1.exchange,
            "leg2_exchange": pair.leg2.exchange,
        }
        features["current_entry_executable"] = float(
            features["current_entry_fill_share"] >= 0.95
        )
        self._add_transformer_l2_features(
            features, history, leg1, leg2, edge, decision_ts
        )
        self._add_transformer_ohlcv_features(
            features, pair, leg1, leg2, decision_ts
        )
        features.update(self._build_q35_features(pair, long_book, short_book, decision_ts))
        snapshot = PairSnapshot(
            pair=pair,
            decision_ts=decision_ts,
            grid_ts=grid_ts,
            long_book=long_book,
            short_book=short_book,
            features=features,
        )
        if history and history[-1].grid_ts == grid_ts:
            history[-1] = snapshot
        else:
            history.append(snapshot)
        return snapshot

    def history(self, pair_id: str) -> list[PairSnapshot]:
        """Вернуть ограниченную историю уже построенных pair snapshots."""

        return list(self._history.get(pair_id, ()))

    def _add_transformer_l2_features(
        self,
        features: dict[str, Any],
        history: deque[PairSnapshot],
        leg1: L2Snapshot,
        leg2: L2Snapshot,
        edge: float,
        decision_ts: int,
    ) -> None:
        """Добавить динамику edge, глубины и imbalance на разных горизонтах."""

        def historical(name: str, steps: int, default: float) -> float:
            if len(history) < steps:
                return default
            return float(history[-steps].features.get(name, default))

        def rolling_values(name: str, steps: int, current: float) -> list[float]:
            previous = [
                float(item.features.get(name, current))
                for item in list(history)[-(steps - 1) :]
            ]
            return [*previous, current]

        leg1_mid = (leg1.bids[0].price + leg1.asks[0].price) / 2.0
        leg2_mid = (leg2.bids[0].price + leg2.asks[0].price) / 2.0
        features["_leg1_mid"] = leg1_mid
        features["_leg2_mid"] = leg2_mid
        features["_candidate_active"] = float(
            features["current_entry_executable"] > 0 and edge >= 10.0
        )

        for steps, label in ((10, "1s"), (50, "5s"), (300, "30s"), (1200, "120s")):
            features[f"edge_change_{label}_bps"] = edge - historical(
                "entry_edge_top1_bps", steps, edge
            )
        for steps, label in ((10, "1s"), (50, "5s"), (300, "30s")):
            old_leg1 = historical("_leg1_mid", steps, leg1_mid)
            old_leg2 = historical("_leg2_mid", steps, leg2_mid)
            features[f"leg1_mid_return_{label}_bps"] = (
                leg1_mid / old_leg1 - 1.0
            ) * 10_000.0
            features[f"leg2_mid_return_{label}_bps"] = (
                leg2_mid / old_leg2 - 1.0
            ) * 10_000.0
            features[f"relative_mid_return_{label}_bps"] = (
                features[f"leg1_mid_return_{label}_bps"]
                - features[f"leg2_mid_return_{label}_bps"]
            )
        for steps, label in ((10, "1s"), (50, "5s")):
            features[f"leg1_imbalance_change_{label}"] = float(
                features["leg1_imbalance"]
            ) - historical("leg1_imbalance", steps, float(features["leg1_imbalance"]))
            features[f"leg2_imbalance_change_{label}"] = float(
                features["leg2_imbalance"]
            ) - historical("leg2_imbalance", steps, float(features["leg2_imbalance"]))

        windows = ((50, "5s"), (300, "30s"), (1200, "120s"), (3000, "300s"))
        for steps, label in windows:
            values = rolling_values("entry_edge_top1_bps", steps, edge)
            features[f"edge_mean_{label}_bps"] = fmean(values)
            features[f"edge_std_{label}_bps"] = (
                stdev(values) if len(values) >= 2 else 0.0
            )

        for steps, label in ((300, "30s"), (1200, "120s"), (3000, "300s")):
            values = rolling_values("entry_edge_top1_bps", steps, edge)
            minimum = min(values)
            maximum = max(values)
            mean = float(features[f"edge_mean_{label}_bps"])
            std = float(features[f"edge_std_{label}_bps"])
            features[f"edge_min_{label}_bps"] = minimum
            features[f"edge_max_{label}_bps"] = maximum
            features[f"edge_z_{label}"] = (edge - mean) / (std + 1e-6)
            features[f"edge_position_{label}"] = (
                (edge - minimum) / (maximum - minimum + 1e-6)
            )

        features["distance_from_max_30s_bps"] = (
            features["edge_max_30s_bps"] - edge
        )
        features["distance_from_max_300s_bps"] = (
            features["edge_max_300s_bps"] - edge
        )
        features["distance_from_min_30s_bps"] = (
            edge - features["edge_min_30s_bps"]
        )
        features["distance_from_min_300s_bps"] = (
            edge - features["edge_min_300s_bps"]
        )
        previous_candidate = historical("_candidate_active", 1, 0.0)
        candidate_start = float(
            features["_candidate_active"] > 0 and previous_candidate <= 0
        )
        features["_candidate_start"] = candidate_start
        for steps, label in ((300, "30s"), (3000, "300s")):
            candidates = rolling_values(
                "_candidate_active", steps, float(features["_candidate_active"])
            )
            starts = rolling_values("_candidate_start", steps, candidate_start)
            features[f"candidate_starts_{label}"] = sum(starts)
            features[f"candidate_active_share_{label}"] = fmean(candidates)

        features["edge_slope_5s_bps_per_sec"] = (
            features["edge_change_5s_bps"] / 5.0
        )
        features["edge_slope_30s_bps_per_sec"] = (
            features["edge_change_30s_bps"] / 30.0
        )
        features["edge_slope_120s_bps_per_sec"] = (
            features["edge_change_120s_bps"] / 120.0
        )
        features["edge_acceleration_5s_vs_30s"] = (
            features["edge_slope_5s_bps_per_sec"]
            - features["edge_slope_30s_bps_per_sec"]
        )
        features["context_valid_numeric"] = 1.0
        features["execution_valid_numeric"] = float(
            features["current_entry_executable"]
        )

    def _add_transformer_ohlcv_features(
        self,
        features: dict[str, Any],
        pair: PairDefinition,
        leg1: L2Snapshot,
        leg2: L2Snapshot,
        decision_ts: int,
    ) -> None:
        """Добавить доступные на момент решения свечные и парные признаки."""

        values: dict[int, dict[str, float]] = {}
        for leg_no, book in ((1, leg1), (2, leg2)):
            candles = self.store.candles(
                book.exchange, book.ticker, "5m", available_at_ms=decision_ts
            )
            current = self._transformer_candle_values(candles)
            latest_available_ts = candles[-1].ts + 300_000 if candles else None
            age = (
                (decision_ts - latest_available_ts) / 1000.0
                if latest_available_ts is not None
                else 900.0
            )
            available = float(
                bool(candles) and len(candles) >= 48 and 0.0 <= age <= 900.0
            )
            values[leg_no] = {
                name: value if available else 0.0
                for name, value in current.items()
            }
            features[f"leg{leg_no}_ohlcv_age_sec"] = min(900.0, max(0.0, age))
            features[f"leg{leg_no}_ohlcv_available"] = available
            for name, value in values[leg_no].items():
                features[f"leg{leg_no}_ohlcv_{name}"] = value

        first = values[1]
        second = values[2]
        sign = 1.0 if pair.direction_code == 0 else -1.0
        for bars in (1, 3, 6, 12, 24, 48):
            features[f"directional_ret_gap_{bars}_bars_bps"] = sign * (
                first[f"ret_{bars}_bars_bps"] - second[f"ret_{bars}_bars_bps"]
            )
        for bars in (3, 12, 48):
            features[f"absolute_ret_gap_{bars}_bars_bps"] = abs(
                first[f"ret_{bars}_bars_bps"] - second[f"ret_{bars}_bars_bps"]
            )
        features["directional_body_gap_bps"] = sign * (
            first["body_bps"] - second["body_bps"]
        )
        for bars in (6, 24, 48):
            features[f"directional_trend_gap_{bars}_bps"] = sign * (
                first[f"close_vs_mean_{bars}_bps"]
                - second[f"close_vs_mean_{bars}_bps"]
            )
        for bars in (12, 48):
            features[f"mean_rv_{bars}_bars_bps"] = (
                first[f"rv_{bars}_bars_bps"]
                + second[f"rv_{bars}_bars_bps"]
            ) / 2.0
            features[f"absolute_rv_gap_{bars}_bars_bps"] = abs(
                first[f"rv_{bars}_bars_bps"]
                - second[f"rv_{bars}_bars_bps"]
            )
        features["directional_vol_gap_12_bars_bps"] = sign * (
            first["rv_12_bars_bps"] - second["rv_12_bars_bps"]
        )
        features["directional_rsi_gap"] = sign * (
            first["rsi_14"] - second["rsi_14"]
        )
        features["quote_volume_z_gap_12"] = (
            first["quote_volume_z_12"] - second["quote_volume_z_12"]
        )
        features["quote_volume_z_gap_48"] = (
            first["quote_volume_z_48"] - second["quote_volume_z_48"]
        )
        features["log_quote_volume_ratio"] = (
            first["log_quote_volume"] - second["log_quote_volume"]
        )

    @staticmethod
    def _transformer_candle_values(
        candles: list[OHLCVCandle],
    ) -> dict[str, float]:
        """Рассчитать returns, volatility, volume, RSI и Bollinger width."""

        names = [
            *[f"ret_{bars}_bars_bps" for bars in (1, 3, 6, 12, 24, 48)],
            "range_bps",
            "body_bps",
            "upper_wick_bps",
            "lower_wick_bps",
            "close_location",
            *[f"rv_{bars}_bars_bps" for bars in (6, 12, 24, 48)],
            "atr_14_bps",
            *[f"close_vs_mean_{bars}_bps" for bars in (6, 12, 24, 48)],
            "log_volume",
            "volume_z_12",
            "log_quote_volume",
            "quote_volume_z_12",
            "quote_volume_z_48",
            "log_quote_volume_change_1",
            "zero_volume",
            "rsi_14",
            "bb_width_20_bps",
        ]
        if not candles:
            return {name: 0.0 for name in names}

        closes = [item.close for item in candles]
        current = candles[-1]
        log_returns = [0.0]
        true_ranges = [candles[0].high - candles[0].low]
        volumes = []
        log_volumes = []
        quote_volumes = []
        log_quote_volumes = []
        for index, item in enumerate(candles):
            volumes.append(item.volume)
            log_volumes.append(math.log1p(item.volume))
            quote_volume = item.volume * item.close
            quote_volumes.append(quote_volume)
            log_quote_volumes.append(math.log1p(quote_volume))
            if index:
                previous_close = candles[index - 1].close
                log_returns.append(math.log(item.close / previous_close))
                true_ranges.append(
                    max(
                        item.high - item.low,
                        abs(item.high - previous_close),
                        abs(item.low - previous_close),
                    )
                )

        result: dict[str, float] = {}
        for bars in (1, 3, 6, 12, 24, 48):
            result[f"ret_{bars}_bars_bps"] = (
                math.log(current.close / closes[-bars - 1]) * 10_000.0
                if len(closes) > bars
                else 0.0
            )
        result.update(
            {
                "range_bps": (current.high - current.low)
                / current.open
                * 10_000.0,
                "body_bps": (current.close - current.open)
                / current.open
                * 10_000.0,
                "upper_wick_bps": (
                    current.high - max(current.open, current.close)
                )
                / current.open
                * 10_000.0,
                "lower_wick_bps": (
                    min(current.open, current.close) - current.low
                )
                / current.open
                * 10_000.0,
                "close_location": (current.close - current.low)
                / (current.high - current.low + 1e-12),
                "atr_14_bps": (
                    fmean(true_ranges[-14:]) / current.close * 10_000.0
                    if len(true_ranges) >= 2
                    else 0.0
                ),
            }
        )
        for bars in (6, 12, 24, 48):
            returns = log_returns[-bars:]
            result[f"rv_{bars}_bars_bps"] = (
                stdev(returns) * 10_000.0 if len(returns) >= 2 else 0.0
            )
            close_mean = fmean(closes[-bars:])
            result[f"close_vs_mean_{bars}_bps"] = (
                current.close / close_mean - 1.0
            ) * 10_000.0

        def z_score(values: list[float], window: int) -> float:
            values = values[-window:]
            if len(values) < 2:
                return 0.0
            value = (values[-1] - fmean(values)) / (stdev(values) + 1e-9)
            return min(10.0, max(-10.0, value))

        changes = [
            closes[index] - closes[index - 1]
            for index in range(max(1, len(closes) - 14), len(closes))
        ]
        gains = [max(value, 0.0) for value in changes]
        losses = [max(-value, 0.0) for value in changes]
        if len(changes) >= 2:
            ratio = fmean(gains) / (fmean(losses) + 1e-12)
            rsi = 100.0 - 100.0 / (1.0 + ratio)
        else:
            rsi = 50.0
        close_20 = closes[-20:]
        bb_width = (
            4.0 * stdev(close_20) / fmean(close_20) * 10_000.0
            if len(close_20) >= 2
            else 0.0
        )
        result.update(
            {
                "log_volume": log_volumes[-1],
                "volume_z_12": z_score(volumes, 12),
                "log_quote_volume": log_quote_volumes[-1],
                "quote_volume_z_12": z_score(quote_volumes, 12),
                "quote_volume_z_48": z_score(quote_volumes, 48),
                "log_quote_volume_change_1": (
                    log_quote_volumes[-1] - log_quote_volumes[-2]
                    if len(log_quote_volumes) >= 2
                    else 0.0
                ),
                "zero_volume": float(current.volume <= 0),
                "rsi_14": rsi,
                "bb_width_20_bps": bb_width,
            }
        )
        return result

    def _build_q35_features(
        self,
        pair: PairDefinition,
        long_book: L2Snapshot,
        short_book: L2Snapshot,
        decision_ts: int,
    ) -> dict[str, Any]:
        """Воспроизвести legacy 98-feature contract q35-регрессора."""

        timestamp = datetime.fromtimestamp(decision_ts / 1000.0, tz=timezone.utc)
        result: dict[str, Any] = {
            "pair_type": pair.pair_type,
            "base_ticker": pair.base_ticker,
            "leg1_ticker": long_book.ticker,
            "leg2_ticker": short_book.ticker,
            "leg1_exchange": long_book.exchange,
            "leg2_exchange": short_book.exchange,
            "leg1_is_perp": long_book.is_perp,
            "leg2_is_perp": short_book.is_perp,
            "exchanges_key": f"{long_book.exchange}_{short_book.exchange}",
            "start_spread_bps": (
                short_book.bids[0].price / long_book.asks[0].price - 1.0
            )
            * 10_000.0,
            "entry_hour_utc": timestamp.hour,
            "entry_dow_utc": timestamp.weekday(),
            "hour_sin": math.sin(2.0 * math.pi * timestamp.hour / 24.0),
            "hour_cos": math.cos(2.0 * math.pi * timestamp.hour / 24.0),
        }

        leg_values: dict[tuple[int, str], dict[str, float]] = {}
        for leg_no, book in ((1, long_book), (2, short_book)):
            for tf in ("1m", "5m"):
                candles = self.store.candles(
                    book.exchange, book.ticker, tf, available_at_ms=decision_ts
                )
                duration_ms = 60_000 if tf == "1m" else 300_000
                tolerance_ms = 10 * 60_000 if tf == "1m" else 30 * 60_000
                if (
                    candles
                    and decision_ts - (candles[-1].ts + duration_ms) > tolerance_ms
                ):
                    candles = []
                values = _candle_values(candles)
                leg_values[(leg_no, tf)] = values
                for name, value in values.items():
                    result[f"leg{leg_no}_{tf}_{name}"] = value

        for tf in ("1m", "5m"):
            long_values = leg_values[(1, tf)]
            short_values = leg_values[(2, tf)]
            result[f"pair_{tf}_has_both"] = (
                long_values["has"] * short_values["has"]
            )
            for metric in self.Q35_OHLCV_VALUES:
                long_value = long_values[metric]
                short_value = short_values[metric]
                diff = short_value - long_value
                prefix = f"pair_{tf}_{metric}"
                result[f"{prefix}_short_minus_long"] = diff
                result[f"{prefix}_absdiff"] = abs(diff)
                result[f"{prefix}_ratio_short_to_long"] = _safe_ratio(
                    short_value, long_value
                )
                if metric in self.Q35_ADDITIVE_VALUES:
                    result[f"{prefix}_sum"] = short_value + long_value
        return result
