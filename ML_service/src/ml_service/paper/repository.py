"""Хранилища paper telemetry: ограниченная память или SQLite."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any


@dataclass
class PaperPosition:
    """Исполненная двухногая позиция и её running MFE/MAE."""

    strategy: str
    pair_id: str
    opened_ts: int
    entry_long_price: float
    entry_short_price: float
    entry_edge_bps: float
    notional_usd: float
    entry_fill_ts: int = 0
    quantity: float = 0.0
    long_cost: float = 0.0
    short_proceeds: float = 0.0
    entry_fees: float = 0.0
    long_fee_rate: float = 0.001
    short_fee_rate: float = 0.001
    mfe_bps: float = 0.0
    mae_bps: float = 0.0


def _compact_features(features: dict[str, Any]) -> dict[str, Any]:
    return {
        key: features[key]
        for key in (
            "entry_edge_top1_bps",
            "current_entry_fill_share",
            "current_entry_executable",
            "current_open_gross_edge_bps",
            "leg1_book_age_sec",
            "leg2_book_age_sec",
            "pair_skew_sec",
        )
        if key in features
    }


def _trade_stats(rows: list[Any]) -> list[dict[str, Any]]:
    """Посчитать доходность и риск отдельно для каждой стратегии."""

    grouped: dict[str, list[Any]] = {}
    for row in rows:
        grouped.setdefault(str(row["strategy"]), []).append(row)

    result = []
    for strategy, strategy_rows in grouped.items():
        pnls = [float(row["net_pnl_bps"]) for row in strategy_rows]
        sorted_pnls = sorted(pnls)
        p05_index = max(0, int(0.05 * (len(sorted_pnls) - 1)))
        equity = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for pnl in pnls:
            equity += pnl
            peak = max(peak, equity)
            max_drawdown = min(max_drawdown, equity - peak)
        result.append(
            {
                "strategy": strategy,
                "trades": len(pnls),
                "total_net_pnl_bps": sum(pnls),
                "mean_trade_pnl_bps": fmean(pnls),
                "median_trade_pnl_bps": median(pnls),
                "win_rate": sum(value > 0 for value in pnls) / len(pnls),
                "p05_trade_pnl_bps": sorted_pnls[p05_index],
                "max_drawdown_bps": max_drawdown,
                "mean_hold_seconds": fmean(
                    float(row["hold_ms"]) / 1000.0 for row in strategy_rows
                ),
                "last_closed_ts": int(strategy_rows[-1]["closed_ts"]),
            }
        )
    return result


class PaperRepository:
    """SQLite-реализация для сохранения paper state между рестартами."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    strategy TEXT NOT NULL,
                    pair_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (strategy, pair_id)
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy TEXT NOT NULL,
                    pair_id TEXT NOT NULL,
                    opened_ts INTEGER NOT NULL,
                    closed_ts INTEGER NOT NULL,
                    hold_ms INTEGER NOT NULL,
                    gross_pnl_bps REAL NOT NULL,
                    fee_bps REAL NOT NULL,
                    net_pnl_bps REAL NOT NULL,
                    exit_reason TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_ts INTEGER NOT NULL,
                    strategy TEXT NOT NULL,
                    pair_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    predictions TEXT NOT NULL,
                    features TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_trades_strategy_time
                ON trades(strategy, closed_ts);

                CREATE INDEX IF NOT EXISTS idx_decisions_strategy_time
                ON decisions(strategy, decision_ts);
                """
            )

    def load_positions(self) -> dict[tuple[str, str], PaperPosition]:
        """Восстановить открытые позиции после старта процесса."""

        with self._lock:
            rows = self._connection.execute("SELECT payload FROM positions").fetchall()
        positions = [PaperPosition(**json.loads(row["payload"])) for row in rows]
        return {(item.strategy, item.pair_id): item for item in positions}

    def save_position(self, position: PaperPosition) -> None:
        """Атомарно создать или обновить открытую позицию."""

        payload = json.dumps(asdict(position), separators=(",", ":"))
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO positions(strategy, pair_id, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(strategy, pair_id)
                DO UPDATE SET payload = excluded.payload
                """,
                (position.strategy, position.pair_id, payload),
            )

    def delete_position(self, strategy: str, pair_id: str) -> None:
        """Удалить закрытую позицию."""

        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM positions WHERE strategy = ? AND pair_id = ?",
                (strategy, pair_id),
            )

    def add_trade(self, values: dict[str, Any]) -> None:
        """Добавить завершённую сделку."""

        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO trades(
                    strategy, pair_id, opened_ts, closed_ts, hold_ms,
                    gross_pnl_bps, fee_bps, net_pnl_bps, exit_reason, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    values["strategy"],
                    values["pair_id"],
                    values["opened_ts"],
                    values["closed_ts"],
                    values["hold_ms"],
                    values["gross_pnl_bps"],
                    values["fee_bps"],
                    values["net_pnl_bps"],
                    values["exit_reason"],
                    json.dumps(values, separators=(",", ":")),
                ),
            )

    def add_decision(
        self,
        decision_ts: int,
        strategy: str,
        pair_id: str,
        state: str,
        action: str,
        reason: str,
        predictions: dict[str, Any],
        features: dict[str, Any],
    ) -> None:
        """Сохранить решение с компактным набором признаков."""

        compact_features = _compact_features(features)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO decisions(
                    decision_ts, strategy, pair_id, state, action, reason,
                    predictions, features
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_ts,
                    strategy,
                    pair_id,
                    state,
                    action,
                    reason,
                    json.dumps(predictions, separators=(",", ":")),
                    json.dumps(compact_features, separators=(",", ":")),
                ),
            )

    def trades(
        self,
        limit: int = 200,
        strategy: str | None = None,
        pair_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Вернуть последние сделки с фильтрами стратегии и пары."""

        query = "SELECT payload FROM trades"
        filters = []
        parameters: list[Any] = []
        if strategy is not None:
            filters.append("strategy = ?")
            parameters.append(strategy)
        if pair_id is not None:
            filters.append("pair_id = ?")
            parameters.append(pair_id)
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY id DESC LIMIT ?"
        parameters.append(int(limit))
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def decisions(
        self,
        limit: int = 200,
        strategy: str | None = None,
        pair_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Вернуть журнал решений с необязательными фильтрами."""

        query = "SELECT * FROM decisions"
        filters = []
        parameters: list[Any] = []
        if strategy is not None:
            filters.append("strategy = ?")
            parameters.append(strategy)
        if pair_id is not None:
            filters.append("pair_id = ?")
            parameters.append(pair_id)
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY id DESC LIMIT ?"
        parameters.append(int(limit))
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def events(self, limit: int = 100) -> list[dict[str, Any]]:
        """Вернуть только решения, изменяющие торговое состояние."""

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM decisions
                WHERE action NOT IN ('wait', 'hold')
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> list[dict[str, Any]]:
        """Вернуть агрегированные метрики всех закрытых сделок."""

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT
                    strategy,
                    closed_ts,
                    hold_ms,
                    net_pnl_bps
                FROM trades
                ORDER BY strategy, closed_ts, id
                """
            ).fetchall()
        return _trade_stats(rows)

    def status(self) -> dict[str, Any]:
        """Вернуть размеры SQLite-таблиц."""

        with self._lock:
            positions = self._connection.execute(
                "SELECT count(*) FROM positions"
            ).fetchone()[0]
            trades = self._connection.execute(
                "SELECT count(*) FROM trades"
            ).fetchone()[0]
            decisions = self._connection.execute(
                "SELECT count(*) FROM decisions"
            ).fetchone()[0]
        return {
            "mode": "sqlite",
            "positions": int(positions),
            "trades": int(trades),
            "decisions": int(decisions),
        }

    def close(self) -> None:
        """Закрыть SQLite connection."""

        with self._lock:
            self._connection.close()


class InMemoryPaperRepository:
    """Bounded-memory telemetry без записи рыночных данных на диск."""

    def __init__(
        self,
        max_trades: int = 2_000,
        max_decisions: int = 5_000,
        max_events: int = 1_000,
    ):
        self._lock = threading.RLock()
        self._positions: dict[tuple[str, str], PaperPosition] = {}
        self._trades: deque[dict[str, Any]] = deque(maxlen=max_trades)
        self._decisions: deque[dict[str, Any]] = deque(maxlen=max_decisions)
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._next_decision_id = 1

    def load_positions(self) -> dict[tuple[str, str], PaperPosition]:
        """Вернуть копии текущих позиций."""

        with self._lock:
            return {
                key: PaperPosition(**asdict(value))
                for key, value in self._positions.items()
            }

    def save_position(self, position: PaperPosition) -> None:
        """Создать или обновить позицию в памяти."""

        with self._lock:
            self._positions[(position.strategy, position.pair_id)] = (
                PaperPosition(**asdict(position))
            )

    def delete_position(self, strategy: str, pair_id: str) -> None:
        """Удалить позицию из памяти."""

        with self._lock:
            self._positions.pop((strategy, pair_id), None)

    def add_trade(self, values: dict[str, Any]) -> None:
        """Добавить сделку в bounded deque."""

        with self._lock:
            self._trades.append(dict(values))

    def add_decision(
        self,
        decision_ts: int,
        strategy: str,
        pair_id: str,
        state: str,
        action: str,
        reason: str,
        predictions: dict[str, Any],
        features: dict[str, Any],
    ) -> None:
        """Добавить решение и отдельно индексировать значимое событие."""

        row = {
            "id": self._next_decision_id,
            "decision_ts": int(decision_ts),
            "strategy": strategy,
            "pair_id": pair_id,
            "state": state,
            "action": action,
            "reason": reason,
            "predictions": json.dumps(predictions, separators=(",", ":")),
            "features": json.dumps(
                _compact_features(features), separators=(",", ":")
            ),
        }
        with self._lock:
            self._next_decision_id += 1
            self._decisions.append(row)
            if action not in {"wait", "hold"}:
                self._events.append(row)

    def trades(
        self,
        limit: int = 200,
        strategy: str | None = None,
        pair_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Вернуть последние сделки в обратном хронологическом порядке."""

        with self._lock:
            rows = list(self._trades)
        rows = [
            row
            for row in rows
            if (strategy is None or row["strategy"] == strategy)
            and (pair_id is None or row["pair_id"] == pair_id)
        ]
        return [dict(row) for row in reversed(rows[-int(limit) :])]

    def decisions(
        self,
        limit: int = 200,
        strategy: str | None = None,
        pair_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Вернуть последние решения с фильтрами."""

        with self._lock:
            rows = list(self._decisions)
        rows = [
            row
            for row in rows
            if (strategy is None or row["strategy"] == strategy)
            and (pair_id is None or row["pair_id"] == pair_id)
        ]
        return [dict(row) for row in reversed(rows[-int(limit) :])]

    def events(self, limit: int = 100) -> list[dict[str, Any]]:
        """Вернуть последние значимые события."""

        with self._lock:
            rows = list(self._events)
        return [dict(row) for row in reversed(rows[-int(limit) :])]

    def stats(self) -> list[dict[str, Any]]:
        """Агрегировать метрики по текущему bounded trade history."""

        with self._lock:
            rows = list(self._trades)
        return _trade_stats(rows)

    def status(self) -> dict[str, Any]:
        """Вернуть заполнение bounded-memory буферов."""

        with self._lock:
            return {
                "mode": "memory",
                "positions": len(self._positions),
                "trades": len(self._trades),
                "decisions": len(self._decisions),
                "events": len(self._events),
                "max_trades": int(self._trades.maxlen or 0),
                "max_decisions": int(self._decisions.maxlen or 0),
                "max_events": int(self._events.maxlen or 0),
            }

    def close(self) -> None:
        """Не требует освобождения внешних ресурсов."""

        return None
