"""Paper execution: решения, pending fills, позиции и PnL."""

from __future__ import annotations

import math
import threading
from dataclasses import asdict, dataclass
from typing import Any, Literal

from ml_service.config import StrategySettings
from ml_service.paper.repository import (
    InMemoryPaperRepository,
    PaperPosition,
    PaperRepository,
)
from ml_service.preprocessing.features import PairSnapshot


@dataclass
class PendingOrder:
    """Решение, ожидающее исполнения после заданной задержки."""

    strategy: str
    pair_id: str
    side: Literal["entry", "exit"]
    decision_ts: int
    due_ts: int
    reason: str


@dataclass
class SessionState:
    """Накопленное состояние одной RL-сессии по паре."""

    started_ts: int
    realized_bps: float = 0.0
    trades: int = 0
    cooldown_until: int = 0


def _source_value(source: str | None, predictions: dict[str, Any]) -> float | None:
    if source is None:
        return None
    value: Any = predictions
    for part in source.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _walk_quantity(
    levels: list[Any], quantity: float
) -> tuple[float, float, float]:
    remaining = float(quantity)
    filled = 0.0
    quote_total = 0.0
    for level in levels:
        take = min(remaining, float(level.volume))
        if take <= 0:
            continue
        quote_total += take * float(level.price)
        filled += take
        remaining -= take
        if remaining <= 1e-9:
            break
    price = quote_total / filled if filled > 0 else 0.0
    return filled, price, quote_total


class PaperEngine:
    """Независимо исполняет декларативные стратегии по общему рынку."""

    def __init__(
        self,
        repository: PaperRepository | InMemoryPaperRepository,
        strategies: dict[str, StrategySettings],
        fee_bps_per_leg_side: float = 10.0,
        min_fill_share: float = 0.95,
    ):
        self.repository = repository
        self._lock = threading.RLock()
        self._positions = repository.load_positions()
        self._pending: dict[tuple[str, str], PendingOrder] = {}
        self._sessions: dict[tuple[str, str], SessionState] = {}
        self._strategies = strategies
        self._active = {
            name: setting.enabled for name, setting in strategies.items()
        }
        self.fee_bps_per_leg_side = float(fee_bps_per_leg_side)
        self.min_fill_share = float(min_fill_share)

    def strategies(self) -> list[dict[str, Any]]:
        """Вернуть конфигурацию и флаг активности всех стратегий."""

        with self._lock:
            return [
                {
                    "name": name,
                    "active": self._active[name],
                    **setting.values,
                }
                for name, setting in self._strategies.items()
            ]

    def set_active(self, name: str, active: bool) -> None:
        """Включить или приостановить стратегию без перезапуска."""

        with self._lock:
            if name not in self._strategies:
                raise KeyError(name)
            self._active[name] = active

    def active_strategy_names(self) -> list[str]:
        """Вернуть имена стратегий, принимающих решения."""

        with self._lock:
            return [name for name, active in self._active.items() if active]

    def strategy_values(self, strategy_name: str) -> dict[str, Any]:
        """Вернуть параметры одной стратегии."""

        return self._strategies[strategy_name].values

    def applies_to(self, strategy_name: str, pair_type: str) -> bool:
        """Проверить, разрешён ли тип пары для стратегии."""

        configured = self._strategies[strategy_name].values.get("pair_types")
        return configured is None or pair_type in configured

    def gate_source(self, strategy_name: str, pair_type: str) -> str | None:
        """Найти prediction source, активирующий opportunity gate."""

        values = self._strategies[strategy_name].values
        sources = values.get("gate_sources")
        if isinstance(sources, dict):
            source = sources.get(pair_type)
            return str(source) if source else None
        source = values.get("opportunity_source")
        return str(source) if source else None

    @staticmethod
    def source_value(
        source: str | None, predictions: dict[str, Any]
    ) -> float | None:
        """Безопасно извлечь числовой выход по dotted path."""

        return _source_value(source, predictions)

    def required_models(
        self, strategy_name: str, pair_type: str | None = None
    ) -> set[str]:
        """Извлечь модели, на которые ссылается конфигурация стратегии."""

        values = self._strategies[strategy_name].values
        result: set[str] = set()
        for key, source in values.items():
            if key.endswith("_source") and isinstance(source, str) and "." in source:
                result.add(source.split(".", 1)[0])
        gate_sources = values.get("gate_sources")
        if isinstance(gate_sources, dict):
            selected = (
                [gate_sources.get(pair_type)]
                if pair_type is not None
                else gate_sources.values()
            )
            for source in selected:
                if isinstance(source, str) and "." in source:
                    result.add(source.split(".", 1)[0])
        return result

    def position(self, strategy: str, pair_id: str) -> PaperPosition | None:
        """Вернуть открытую позицию стратегии по паре."""

        with self._lock:
            return self._positions.get((strategy, pair_id))

    def positions(self) -> list[dict[str, Any]]:
        """Сериализовать все открытые позиции."""

        with self._lock:
            return [asdict(value) for value in self._positions.values()]

    def position_state(
        self, strategy: str, snapshot: PairSnapshot
    ) -> list[float]:
        """Сформировать шестимерное состояние позиции для Transformer."""
        position = self.position(strategy, snapshot.pair.pair_id)
        if position is None:
            return [0.0] * 6
        current = self._mark_to_market(position, snapshot)
        if not math.isfinite(current):
            current = position.mae_bps
        hold_sec = max(
            0.0, (snapshot.decision_ts - position.opened_ts) / 1000.0
        )
        return [
            1.0,
            current / 100.0,
            position.mfe_bps / 100.0,
            position.mae_bps / 100.0,
            hold_sec / 300.0,
            position.entry_edge_bps / 100.0,
        ]

    def rl_position_state(
        self, strategy: str, snapshot: PairSnapshot
    ) -> list[float]:
        """Сформировать восьмимерное состояние позиции и RL-сессии."""

        with self._lock:
            key = (strategy, snapshot.pair.pair_id)
            values = self._strategies[strategy].values
            session = self._sessions.get(key)
            if session is None:
                session = SessionState(started_ts=snapshot.decision_ts)
                self._sessions[key] = session
            position = self._positions.get(key)
            reset_ms = int(values.get("session_reset_ms", 1_800_000))
            if (
                position is None
                and snapshot.decision_ts - session.started_ts >= reset_ms
                and snapshot.decision_ts >= session.cooldown_until
            ):
                session = SessionState(started_ts=snapshot.decision_ts)
                self._sessions[key] = session

            cooldown_steps = max(
                0.0,
                (session.cooldown_until - snapshot.decision_ts) / 100.0,
            )
            cooldown_ratio = cooldown_steps / float(
                values.get("cooldown_steps", 100)
            )
            if position is None:
                return [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    session.realized_bps / 100.0,
                    max(0.0, cooldown_ratio),
                    min(session.trades, 10) / 10.0,
                ]

            current = self._mark_to_market(position, snapshot)
            if not math.isfinite(current):
                current = position.mae_bps
            hold_ms = max(0, snapshot.decision_ts - position.opened_ts)
            return [
                1.0,
                current / 100.0,
                position.mfe_bps / 100.0,
                position.mae_bps / 100.0,
                hold_ms / 300_000.0,
                session.realized_bps / 100.0,
                0.0,
                min(session.trades, 10) / 10.0,
            ]

    def advance(self, strategy_name: str, snapshot: PairSnapshot) -> None:
        """Исполнить due fill до следующего recurrent inference step."""
        with self._lock:
            if not self._active.get(strategy_name, False):
                return
            if not self.applies_to(strategy_name, snapshot.pair.pair_type):
                return
            self._advance_locked(strategy_name, snapshot)

    def _advance_locked(
        self, strategy_name: str, snapshot: PairSnapshot
    ) -> None:
        key = (strategy_name, snapshot.pair.pair_id)
        values = self._strategies[strategy_name].values
        pending = self._pending.get(key)
        if pending is None or snapshot.decision_ts < pending.due_ts:
            position = self._positions.get(key)
            if position is not None:
                mark = self._mark_to_market(position, snapshot)
                if math.isfinite(mark):
                    position.mfe_bps = max(position.mfe_bps, mark)
                    position.mae_bps = min(position.mae_bps, mark)
                    self.repository.save_position(position)
            return

        position = self._positions.get(key)
        executed = False
        execution_reason = pending.reason
        if pending.side == "entry" and position is None:
            executed, execution_reason = self._execute_entry(
                strategy_name, snapshot, values, pending.decision_ts
            )
        elif pending.side == "exit" and position is not None:
            executed, execution_reason = self._execute_exit(
                strategy_name,
                snapshot,
                position,
                pending.reason,
                values,
                pending.decision_ts,
            )
        self._pending.pop(key, None)
        current = self._positions.get(key)
        if current is not None:
            mark = self._mark_to_market(current, snapshot)
            if math.isfinite(mark):
                current.mfe_bps = max(current.mfe_bps, mark)
                current.mae_bps = min(current.mae_bps, mark)
                self.repository.save_position(current)
        self._record(
            strategy_name,
            snapshot,
            {},
            "open" if current is not None else "flat",
            f"{pending.side}_{'executed' if executed else 'rejected'}",
            execution_reason,
        )

    def evaluate(
        self,
        strategy_name: str,
        snapshot: PairSnapshot,
        predictions: dict[str, Any],
    ) -> None:
        """Принять решение стратегии и запланировать entry/exit."""

        with self._lock:
            if not self._active.get(strategy_name, False):
                return
            if not self.applies_to(strategy_name, snapshot.pair.pair_type):
                return
            self._advance_locked(strategy_name, snapshot)
            key = (strategy_name, snapshot.pair.pair_id)
            position = self._positions.get(key)
            pending = self._pending.get(key)
            values = self._strategies[strategy_name].values

            if pending is not None:
                self._record(
                    strategy_name, snapshot, predictions, "pending", "wait", pending.reason
                )
                return

            delay = int(values.get("execution_delay_ms", 100))
            if position is None:
                session = self._sessions.setdefault(
                    key, SessionState(started_ts=snapshot.decision_ts)
                )
                if snapshot.decision_ts < session.cooldown_until:
                    enter, reason = False, "entry_cooldown"
                else:
                    enter, reason = self._should_enter(values, predictions)
                if enter:
                    self._pending[key] = PendingOrder(
                        strategy=strategy_name,
                        pair_id=snapshot.pair.pair_id,
                        side="entry",
                        decision_ts=snapshot.decision_ts,
                        due_ts=snapshot.decision_ts + delay,
                        reason=reason,
                    )
                    action = "schedule_entry"
                else:
                    action = "wait"
                self._record(
                    strategy_name, snapshot, predictions, "flat", action, reason
                )
                return

            exit_now, reason = self._should_exit(values, predictions, position, snapshot)
            if exit_now:
                self._pending[key] = PendingOrder(
                    strategy=strategy_name,
                    pair_id=snapshot.pair.pair_id,
                    side="exit",
                    decision_ts=snapshot.decision_ts,
                    due_ts=snapshot.decision_ts + delay,
                    reason=reason,
                )
                action = "schedule_exit"
            else:
                action = "hold"
            self._record(strategy_name, snapshot, predictions, "open", action, reason)

    def _should_enter(
        self, values: dict[str, Any], predictions: dict[str, Any]
    ) -> tuple[bool, str]:
        action_source = values.get("entry_action_source") or values.get(
            "action_source"
        )
        if action_source:
            action = _source_value(action_source, predictions)
            return action == 1.0, f"rl_action={action}"

        checks = (
            ("opportunity_source", "opportunity_min_bps", "opportunity"),
            (
                "entry_probability_source",
                "entry_probability_min",
                "entry_probability",
            ),
            (
                "entry_executable_source",
                "entry_executable_min",
                "entry_executable",
            ),
            (
                "enter_advantage_source",
                "enter_advantage_min_bps",
                "enter_advantage",
            ),
        )
        reasons = []
        for source_key, threshold_key, label in checks:
            source = values.get(source_key)
            threshold = values.get(threshold_key)
            if source is None or threshold is None:
                continue
            actual = _source_value(source, predictions)
            reasons.append(f"{label}={actual}")
            if actual is None or actual < float(threshold):
                return False, ", ".join(reasons)
        return bool(reasons), ", ".join(reasons)

    def _should_exit(
        self,
        values: dict[str, Any],
        predictions: dict[str, Any],
        position: PaperPosition,
        snapshot: PairSnapshot,
    ) -> tuple[bool, str]:
        action_source = values.get("exit_action_source") or values.get(
            "action_source"
        )
        if action_source:
            action = _source_value(action_source, predictions)
            if action == 3.0:
                return True, "rl_action=3"

        emergency_stop = values.get("emergency_stop_bps")
        if emergency_stop is not None:
            mark = self._mark_to_market(position, snapshot)
            if math.isfinite(mark) and mark <= float(emergency_stop):
                return True, f"emergency_stop={mark:.4f}"

        hold_ms = snapshot.decision_ts - position.opened_ts
        if hold_ms >= int(values.get("max_hold_ms", 300_000)):
            return True, "max_hold"

        exit_probability = _source_value(
            values.get("exit_probability_source"), predictions
        )
        exit_probability_min = values.get("exit_probability_min")
        if (
            exit_probability is not None
            and exit_probability_min is not None
            and exit_probability >= float(exit_probability_min)
        ):
            return True, f"exit_probability={exit_probability}"

        exit_advantage = _source_value(
            values.get("exit_advantage_source"), predictions
        )
        exit_advantage_min = values.get("exit_advantage_min_bps")
        if (
            exit_advantage is not None
            and exit_advantage_min is not None
            and exit_advantage >= float(exit_advantage_min)
        ):
            return True, f"exit_advantage={exit_advantage}"
        return False, "hold"

    def _execute_entry(
        self,
        strategy: str,
        snapshot: PairSnapshot,
        values: dict[str, Any],
        entry_decision_ts: int,
    ) -> tuple[bool, str]:
        """Открыть обе ноги одинаковым base quantity по реальной глубине L2."""

        notional = float(values.get("notional_usd", 100.0))
        best_ask = float(snapshot.long_book.asks[0].price)
        target_quantity = notional / best_ask
        available = min(
            sum(float(level.volume) for level in snapshot.long_book.asks),
            sum(float(level.volume) for level in snapshot.short_book.bids),
        )
        quantity = min(target_quantity, available)
        fill_share = quantity / target_quantity if target_quantity > 0 else 0.0
        if fill_share < self.min_fill_share:
            return False, f"fill_share={fill_share:.4f}"
        long_filled, long_price, long_cost = _walk_quantity(
            snapshot.long_book.asks, quantity
        )
        short_filled, short_price, short_proceeds = _walk_quantity(
            snapshot.short_book.bids, quantity
        )
        fill_share = min(long_filled, short_filled) / target_quantity
        if fill_share < self.min_fill_share:
            return False, f"fill_share={fill_share:.4f}"
        fee_rate = self.fee_bps_per_leg_side / 10_000.0
        entry_fees = long_cost * fee_rate + short_proceeds * fee_rate
        entry_edge = (short_price / long_price - 1.0) * 10_000.0
        position = PaperPosition(
            strategy=strategy,
            pair_id=snapshot.pair.pair_id,
            opened_ts=entry_decision_ts,
            entry_fill_ts=snapshot.decision_ts,
            entry_long_price=long_price,
            entry_short_price=short_price,
            entry_edge_bps=entry_edge,
            notional_usd=notional,
            quantity=quantity,
            long_cost=long_cost,
            short_proceeds=short_proceeds,
            entry_fees=entry_fees,
            long_fee_rate=fee_rate,
            short_fee_rate=fee_rate,
        )
        self._positions[(strategy, snapshot.pair.pair_id)] = position
        self.repository.save_position(position)
        return True, f"fill_share={fill_share:.4f}"

    def _execute_exit(
        self,
        strategy: str,
        snapshot: PairSnapshot,
        position: PaperPosition,
        reason: str,
        strategy_values: dict[str, Any],
        exit_decision_ts: int,
    ) -> tuple[bool, str]:
        """Закрыть обе ноги и записать полный PnL с четырьмя комиссиями."""

        quantity = (
            position.quantity
            if position.quantity > 0
            else position.notional_usd / position.entry_long_price
        )
        long_filled, exit_long, long_exit = _walk_quantity(
            snapshot.long_book.bids, quantity
        )
        short_filled, exit_short, short_cover = _walk_quantity(
            snapshot.short_book.asks, quantity
        )
        fill_share = min(long_filled, short_filled) / quantity
        if fill_share < self.min_fill_share:
            return False, f"{reason}, fill_share={fill_share:.4f}"

        long_cost = (
            position.long_cost
            if position.long_cost > 0
            else quantity * position.entry_long_price
        )
        short_proceeds = (
            position.short_proceeds
            if position.short_proceeds > 0
            else quantity * position.entry_short_price
        )
        entry_fees = (
            position.entry_fees
            if position.entry_fees > 0
            else (
                long_cost * position.long_fee_rate
                + short_proceeds * position.short_fee_rate
            )
        )
        exit_fees = (
            long_exit * position.long_fee_rate
            + short_cover * position.short_fee_rate
        )
        gross_usd = short_proceeds - long_cost + long_exit - short_cover
        gross = gross_usd / position.notional_usd * 10_000.0
        fees = (
            (entry_fees + exit_fees)
            / position.notional_usd
            * 10_000.0
        )
        net = gross - fees
        values = {
            "strategy": strategy,
            "pair_id": snapshot.pair.pair_id,
            "opened_ts": position.opened_ts,
            "closed_ts": snapshot.decision_ts,
            "entry_fill_ts": position.entry_fill_ts,
            "exit_decision_ts": exit_decision_ts,
            "hold_ms": exit_decision_ts - position.opened_ts,
            "entry_long_price": position.entry_long_price,
            "entry_short_price": position.entry_short_price,
            "exit_long_price": exit_long,
            "exit_short_price": exit_short,
            "entry_edge_bps": position.entry_edge_bps,
            "gross_pnl_bps": gross,
            "fee_bps": fees,
            "net_pnl_bps": net,
            "mfe_bps": position.mfe_bps,
            "mae_bps": position.mae_bps,
            "exit_reason": reason,
        }
        self.repository.add_trade(values)
        self._positions.pop((strategy, snapshot.pair.pair_id), None)
        self.repository.delete_position(strategy, snapshot.pair.pair_id)
        session = self._sessions.setdefault(
            (strategy, snapshot.pair.pair_id),
            SessionState(started_ts=position.opened_ts),
        )
        session.realized_bps += net
        session.trades += 1
        cooldown_steps = int(strategy_values.get("cooldown_steps", 100))
        session.cooldown_until = exit_decision_ts + cooldown_steps * 100
        return True, reason

    def _mark_to_market(
        self, position: PaperPosition, snapshot: PairSnapshot
    ) -> float:
        """Оценить net PnL немедленного закрытия по текущему стакану."""

        quantity = (
            position.quantity
            if position.quantity > 0
            else position.notional_usd / position.entry_long_price
        )
        long_filled, _, long_exit = _walk_quantity(
            snapshot.long_book.bids, quantity
        )
        short_filled, _, short_cover = _walk_quantity(
            snapshot.short_book.asks, quantity
        )
        if min(long_filled, short_filled) / quantity < self.min_fill_share:
            return math.nan
        long_cost = (
            position.long_cost
            if position.long_cost > 0
            else quantity * position.entry_long_price
        )
        short_proceeds = (
            position.short_proceeds
            if position.short_proceeds > 0
            else quantity * position.entry_short_price
        )
        entry_fees = (
            position.entry_fees
            if position.entry_fees > 0
            else (
                long_cost * position.long_fee_rate
                + short_proceeds * position.short_fee_rate
            )
        )
        exit_fees = (
            long_exit * position.long_fee_rate
            + short_cover * position.short_fee_rate
        )
        pnl_usd = (
            short_proceeds
            - long_cost
            - entry_fees
            + long_exit
            - short_cover
            - exit_fees
        )
        return pnl_usd / position.notional_usd * 10_000.0

    def _record(
        self,
        strategy: str,
        snapshot: PairSnapshot,
        predictions: dict[str, Any],
        state: str,
        action: str,
        reason: str,
    ) -> None:
        self.repository.add_decision(
            decision_ts=snapshot.decision_ts,
            strategy=strategy,
            pair_id=snapshot.pair.pair_id,
            state=state,
            action=action,
            reason=reason,
            predictions=predictions,
            features=snapshot.features,
        )
