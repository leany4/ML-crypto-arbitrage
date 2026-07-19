"""Публикация подготовленного рынка по виртуальным 100-мс шагам."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from market_simulator.config import SimulatorSettings


class ReplayController:
    """Управляет одним детерминированным replay в фоновом потоке."""

    def __init__(self, settings: SimulatorSettings):
        self.settings = settings
        self._lock = threading.RLock()
        self._run_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._status: dict[str, Any] = {
            "state": "idle",
            "virtual_ts": None,
            "batches_sent": 0,
            "snapshots_sent": 0,
            "accepted_snapshots": 0,
            "duplicate_books": 0,
            "ohlcv_candles_sent": 0,
            "accepted_ohlcv_candles": 0,
            "duplicate_ohlcv_candles": 0,
            "scheduled_pairs": 0,
            "error": None,
        }

    @property
    def l2_path(self) -> Path:
        """Путь к глобально отсортированному L2."""

        return self.settings.prepared_dir / "l2_selected_sorted.parquet"

    @property
    def pairs_path(self) -> Path:
        """Путь к определениям направлений пар."""

        return self.settings.prepared_dir / "pairs.json"

    @property
    def ohlcv_path(self) -> Path:
        """Путь к OHLCV с рассчитанным `available_ts`."""

        return self.settings.prepared_dir / "ohlcv_selected_sorted.parquet"

    @property
    def manifest_path(self) -> Path:
        """Путь к границам и метаданным replay."""

        return self.settings.prepared_dir / "manifest.json"

    def status(self) -> dict[str, Any]:
        """Вернуть потокобезопасную копию runtime-счётчиков."""

        with self._lock:
            return dict(self._status)

    def pairs(self) -> list[dict[str, Any]]:
        """Прочитать зарегистрированные направления пар."""

        if not self.pairs_path.exists():
            return []
        return json.loads(self.pairs_path.read_text(encoding="utf-8"))

    def start(
        self,
        speed: float,
        start_ts: int | None,
        end_ts: int | None,
        duration_seconds: int | None,
    ) -> dict[str, Any]:
        """Проверить prepared dataset и запустить виртуальные часы."""

        if speed <= 0:
            raise ValueError("speed must be positive")
        if not all(path.exists() for path in (self.l2_path, self.pairs_path, self.manifest_path)):
            raise RuntimeError(
                "Replay is not prepared. Run python -m market_simulator.prepare first."
            )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("sorted_ohlcv_path") and not self.ohlcv_path.exists():
            raise RuntimeError(
                "Prepared replay declares OHLCV but the sorted OHLCV file is missing"
            )
        replay_start = int(
            start_ts
            if start_ts is not None
            else manifest.get("replay_start_ts", manifest["min_ts"])
        )
        replay_end = int(
            end_ts
            if end_ts is not None
            else manifest.get("replay_end_ts", manifest["max_ts"] + 1)
        )
        if duration_seconds is not None:
            replay_end = min(replay_end, replay_start + int(duration_seconds) * 1_000)
        if replay_end <= replay_start:
            raise ValueError("end_ts must be greater than start_ts")

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Replay is already running")
            self._stop_event.clear()
            self._run_event.set()
            self._status = {
                "state": "starting",
                "virtual_ts": replay_start,
                "start_ts": replay_start,
                "end_ts": replay_end,
                "speed": speed,
                "batches_sent": 0,
                "snapshots_sent": 0,
                "accepted_snapshots": 0,
                "duplicate_books": 0,
                "ohlcv_available": self.ohlcv_path.exists(),
                "ohlcv_candles_sent": 0,
                "accepted_ohlcv_candles": 0,
                "duplicate_ohlcv_candles": 0,
                "scheduled_pairs": 0,
                "error": None,
            }
            self._thread = threading.Thread(
                target=self._run,
                args=(speed, replay_start, replay_end),
                name="l2-market-replay",
                daemon=True,
            )
            self._thread.start()
            return dict(self._status)

    def pause(self) -> dict[str, Any]:
        """Заморозить отправку, сохранив виртуальный timestamp."""

        with self._lock:
            if self._status["state"] != "running":
                raise RuntimeError("Replay is not running")
            self._run_event.clear()
            self._status["state"] = "paused"
            return dict(self._status)

    def resume(self) -> dict[str, Any]:
        """Возобновить отправку с компенсацией времени паузы."""

        with self._lock:
            if self._status["state"] != "paused":
                raise RuntimeError("Replay is not paused")
            self._status["state"] = "running"
            self._run_event.set()
            return dict(self._status)

    def stop(self) -> dict[str, Any]:
        """Запросить остановку и дождаться завершения worker."""

        self._stop_event.set()
        self._run_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5)
        with self._lock:
            if self._status["state"] not in {"completed", "failed"}:
                self._status["state"] = "stopped"
            return dict(self._status)

    def _register_pairs(self, client: httpx.Client) -> None:
        for pair in self.pairs():
            response = client.put(
                f"{self.settings.ml_url}/v1/pairs/{pair['pair_id']}",
                json=pair,
            )
            response.raise_for_status()

    @staticmethod
    def _snapshot(row: dict[str, Any]) -> dict[str, Any]:
        timestamp = int(row["machine_ts_final"])
        return {
            "ticker": str(row["ticker"]),
            "exchange": str(row["exchange"]).lower(),
            "exchange_ts": int(row["exchange_ts"] or timestamp),
            "machine_ts": (
                int(row["machine_ts"]) if row.get("machine_ts") is not None else None
            ),
            "machine_ts_final": timestamp,
            "is_perp": ":" in str(row["ticker"]),
            "bids": [
                {
                    "price": float(row[f"bid_price_{level}"]),
                    "volume": float(row[f"bid_vol_{level}"]),
                }
                for level in range(1, 6)
            ],
            "asks": [
                {
                    "price": float(row[f"ask_price_{level}"]),
                    "volume": float(row[f"ask_vol_{level}"]),
                }
                for level in range(1, 6)
            ],
        }

    @staticmethod
    def _candle(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "ticker": str(row["symbol"]),
            "exchange": str(row["exchange"]).lower(),
            "tf": str(row["tf"]),
            "ts": int(row["ts"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "is_closed": True,
        }

    def _iter_ohlcv_rows(
        self,
        start_ts: int,
        end_ts: int,
    ):
        if not self.ohlcv_path.exists():
            return
        warmup_start = start_ts - self.settings.ohlcv_warmup_minutes * 60_000
        dataset = ds.dataset(self.ohlcv_path, format="parquet")
        scanner = dataset.scanner(
            filter=(
                (ds.field("available_ts") >= warmup_start)
                & (ds.field("available_ts") < end_ts)
            ),
            batch_size=8192,
        )
        for batch in scanner.to_batches():
            yield from batch.to_pylist()

    def _send_ohlcv_batch(
        self,
        client: httpx.Client,
        candles: list[dict[str, Any]],
    ) -> None:
        if not candles:
            return
        response = client.post(
            f"{self.settings.ml_url}{self.settings.ohlcv_batch_endpoint}",
            json={"candles": candles},
        )
        response.raise_for_status()
        result = response.json()
        with self._lock:
            self._status["ohlcv_candles_sent"] += len(candles)
            self._status["accepted_ohlcv_candles"] += int(
                result["accepted"]
            )
            self._status["duplicate_ohlcv_candles"] += int(
                result["duplicates"]
            )
            self._status["scheduled_pairs"] += int(
                result["scheduled_pairs"]
            )

    def _send_ohlcv_until(
        self,
        client: httpx.Client,
        rows,
        pending: dict[str, Any] | None,
        available_at_ts: int,
    ) -> dict[str, Any] | None:
        """Отправить только свечи с `available_ts` не позже virtual time."""

        candles: list[dict[str, Any]] = []
        while (
            pending is not None
            and int(pending["available_ts"]) <= available_at_ts
        ):
            candles.append(self._candle(pending))
            if len(candles) == 5_000:
                self._send_ohlcv_batch(client, candles)
                candles = []
            pending = next(rows, None)
        self._send_ohlcv_batch(client, candles)
        return pending

    def _wait_until(self, target_wall_time: float) -> float:
        pause_started: float | None = None
        while not self._stop_event.is_set():
            if not self._run_event.is_set():
                if pause_started is None:
                    pause_started = time.monotonic()
                self._run_event.wait(timeout=0.25)
                continue
            if pause_started is not None:
                target_wall_time += time.monotonic() - pause_started
                pause_started = None
            remaining = target_wall_time - time.monotonic()
            if remaining <= 0:
                return target_wall_time
            time.sleep(min(remaining, 0.05))
        return target_wall_time

    def _send_batch(
        self,
        client: httpx.Client,
        grid_ts: int,
        snapshots: dict[tuple[str, str], dict[str, Any]],
    ) -> None:
        """Опубликовать последние снимки рынков одного 100-мс bucket."""

        if not snapshots:
            return
        response = client.post(
            f"{self.settings.ml_url}{self.settings.batch_endpoint}",
            json={"snapshots": list(snapshots.values())},
        )
        response.raise_for_status()
        result = response.json()
        with self._lock:
            self._status["virtual_ts"] = grid_ts
            self._status["batches_sent"] += 1
            self._status["snapshots_sent"] += len(snapshots)
            self._status["accepted_snapshots"] += int(result["accepted"])
            self._status["duplicate_books"] += int(result["duplicate_books"])
            self._status["scheduled_pairs"] += int(result["scheduled_pairs"])

    def _run(self, speed: float, start_ts: int, end_ts: int) -> None:
        """Основной цикл с causal OHLCV и синхронизацией virtual/wall time."""

        try:
            with httpx.Client(timeout=self.settings.request_timeout_seconds) as client:
                self._register_pairs(client)
                ohlcv_rows = iter(self._iter_ohlcv_rows(start_ts, end_ts))
                pending_ohlcv = next(ohlcv_rows, None)
                parquet = pq.ParquetFile(self.l2_path)
                first_grid: int | None = None
                wall_start = time.monotonic()
                current_grid: int | None = None
                snapshots: dict[tuple[str, str], dict[str, Any]] = {}
                with self._lock:
                    self._status["state"] = "running"

                for record_batch in parquet.iter_batches(batch_size=8192):
                    if self._stop_event.is_set():
                        break
                    for row in record_batch.to_pylist():
                        timestamp = int(row["machine_ts_final"])
                        if timestamp < start_ts:
                            continue
                        if timestamp >= end_ts:
                            if current_grid is not None:
                                pending_ohlcv = self._send_ohlcv_until(
                                    client,
                                    ohlcv_rows,
                                    pending_ohlcv,
                                    current_grid,
                                )
                                self._send_batch(client, current_grid, snapshots)
                            with self._lock:
                                self._status["state"] = "completed"
                            return
                        grid_ts = timestamp // 100 * 100
                        if current_grid is None:
                            current_grid = grid_ts
                            first_grid = grid_ts
                        elif grid_ts != current_grid:
                            assert first_grid is not None
                            target = wall_start + (current_grid - first_grid) / 1000.0 / speed
                            wall_start += self._wait_until(target) - target
                            if self._stop_event.is_set():
                                break
                            pending_ohlcv = self._send_ohlcv_until(
                                client,
                                ohlcv_rows,
                                pending_ohlcv,
                                current_grid,
                            )
                            self._send_batch(client, current_grid, snapshots)
                            snapshots = {}
                            current_grid = grid_ts
                        snapshot = self._snapshot(row)
                        key = (snapshot["exchange"], snapshot["ticker"])
                        snapshots[key] = snapshot
                    if self._stop_event.is_set():
                        break

                if current_grid is not None and not self._stop_event.is_set():
                    self._send_ohlcv_until(
                        client,
                        ohlcv_rows,
                        pending_ohlcv,
                        current_grid,
                    )
                    self._send_batch(client, current_grid, snapshots)
                with self._lock:
                    self._status["state"] = (
                        "stopped" if self._stop_event.is_set() else "completed"
                    )
        except Exception as error:
            with self._lock:
                self._status["state"] = "failed"
                self._status["error"] = f"{type(error).__name__}: {error}"
