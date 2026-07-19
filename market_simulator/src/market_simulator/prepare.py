"""Отбор непрерывных пар и подготовка глобально отсортированного replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import duckdb


def _base_ticker(ticker: str) -> str:
    return ticker.split(":", 1)[0]


def _pair_id(
    pair_type: str,
    base_ticker: str,
    leg1_exchange: str,
    leg1_ticker: str,
    leg2_exchange: str,
    leg2_ticker: str,
    direction_code: int,
) -> str:
    raw = "|".join(
        (
            pair_type,
            base_ticker,
            leg1_exchange,
            leg1_ticker,
            leg2_exchange,
            leg2_ticker,
            str(direction_code),
        )
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", base_ticker)[:24].strip("_")
    return f"{label}_{direction_code}_{digest}"


def prepare_replay(
    raw_l2_path: Path,
    ohlcv_path: Path | None,
    output_dir: Path,
    max_pairs: int = 8,
    include_spot_perp: bool = False,
    min_ohlcv_rows_per_tf: int = 128,
    min_concurrent_seconds: int = 900,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Выбрать совместимые рынки и записать компактный причинный набор."""

    raw_l2_path = raw_l2_path.resolve()
    ohlcv_path = ohlcv_path.resolve() if ohlcv_path is not None else None
    output_dir = output_dir.resolve()
    if not raw_l2_path.exists():
        raise FileNotFoundError(raw_l2_path)
    if ohlcv_path is not None and not ohlcv_path.exists():
        raise FileNotFoundError(ohlcv_path)
    if max_pairs < 1:
        raise ValueError("max_pairs must be positive")
    if min_concurrent_seconds < 1:
        raise ValueError("min_concurrent_seconds must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_l2_path = output_dir / "l2_selected_sorted.parquet"
    sorted_ohlcv_path = output_dir / "ohlcv_selected_sorted.parquet"
    pairs_path = output_dir / "pairs.json"
    manifest_path = output_dir / "manifest.json"
    output_paths = [sorted_l2_path, pairs_path, manifest_path]
    if ohlcv_path is not None:
        output_paths.append(sorted_ohlcv_path)
    if not overwrite and all(path.exists() for path in output_paths):
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    for path in output_paths:
        path.unlink(missing_ok=True)

    connection = duckdb.connect()
    temp_directory = str(output_dir / "duckdb_tmp").replace("'", "''")
    connection.execute(f"SET temp_directory='{temp_directory}'")
    market_rows = connection.execute(
        """
        SELECT
            lower(exchange) AS exchange,
            ticker,
            count(*)::BIGINT AS rows,
            min(machine_ts_final)::BIGINT AS min_ts,
            max(machine_ts_final)::BIGINT AS max_ts
        FROM read_parquet(?)
        WHERE
            exchange IS NOT NULL
            AND ticker IS NOT NULL
            AND machine_ts_final IS NOT NULL
        GROUP BY 1, 2
        """,
        [str(raw_l2_path)],
    ).fetchall()
    ohlcv_markets: dict[tuple[str, str], dict[str, int]] = {}
    if ohlcv_path is not None:
        ohlcv_rows = connection.execute(
            """
            SELECT
                lower(exchange) AS exchange,
                symbol,
                count(*) FILTER (WHERE tf = '1m')::BIGINT AS rows_1m,
                count(*) FILTER (WHERE tf = '5m')::BIGINT AS rows_5m
            FROM read_parquet(?)
            WHERE tf IN ('1m', '5m')
            GROUP BY 1, 2
            """,
            [str(ohlcv_path)],
        ).fetchall()
        ohlcv_markets = {
            (str(exchange), str(symbol)): {
                "rows_1m": int(rows_1m),
                "rows_5m": int(rows_5m),
            }
            for exchange, symbol, rows_1m, rows_5m in ohlcv_rows
        }

    markets: list[dict[str, Any]] = []
    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for exchange, ticker, rows, min_ts, max_ts in market_rows:
        market = {
            "exchange": str(exchange),
            "ticker": str(ticker),
            "is_perp": ":" in str(ticker),
            "base_ticker": _base_ticker(str(ticker)),
            "rows": int(rows),
            "min_ts": int(min_ts),
            "max_ts": int(max_ts),
            "ohlcv_rows_1m": ohlcv_markets.get(
                (str(exchange), str(ticker)), {}
            ).get("rows_1m", 0),
            "ohlcv_rows_5m": ohlcv_markets.get(
                (str(exchange), str(ticker)), {}
            ).get("rows_5m", 0),
        }
        markets.append(market)
        by_base[market["base_ticker"]].append(market)

    candidates: list[tuple[tuple[int, int], dict[str, Any]]] = []
    for base_ticker, base_markets in by_base.items():
        covered_markets = [
            market
            for market in base_markets
            if ohlcv_path is None
            or (
                market["ohlcv_rows_1m"] >= min_ohlcv_rows_per_tf
                and market["ohlcv_rows_5m"] >= min_ohlcv_rows_per_tf
            )
        ]
        perpetuals = [market for market in covered_markets if market["is_perp"]]
        for left, right in combinations(perpetuals, 2):
            if left["exchange"] == right["exchange"]:
                continue
            overlap_start = max(left["min_ts"], right["min_ts"])
            overlap_end = min(left["max_ts"], right["max_ts"])
            if overlap_end <= overlap_start:
                continue
            legs = sorted(
                (left, right),
                key=lambda market: (market["exchange"], market["ticker"]),
            )
            for direction_code in (0, 1):
                pair = {
                    "pair_id": _pair_id(
                        "perp_perp_cross_exchange",
                        base_ticker,
                        legs[0]["exchange"],
                        legs[0]["ticker"],
                        legs[1]["exchange"],
                        legs[1]["ticker"],
                        direction_code,
                    ),
                    "base_ticker": base_ticker,
                    "pair_type": "perp_perp_cross_exchange",
                    "leg1": {
                        "exchange": legs[0]["exchange"],
                        "ticker": legs[0]["ticker"],
                        "is_perp": True,
                    },
                    "leg2": {
                        "exchange": legs[1]["exchange"],
                        "ticker": legs[1]["ticker"],
                        "is_perp": True,
                    },
                    "direction_code": direction_code,
                    "enabled": True,
                    "overlap_start_ts": overlap_start,
                    "overlap_end_ts": overlap_end,
                }
                score = (overlap_end - overlap_start, min(left["rows"], right["rows"]))
                candidates.append((score, pair))

        if include_spot_perp:
            spots = [market for market in covered_markets if not market["is_perp"]]
            for spot in spots:
                for perpetual in perpetuals:
                    if spot["exchange"] != perpetual["exchange"]:
                        continue
                    overlap_start = max(spot["min_ts"], perpetual["min_ts"])
                    overlap_end = min(spot["max_ts"], perpetual["max_ts"])
                    if overlap_end <= overlap_start:
                        continue
                    pair = {
                        "pair_id": _pair_id(
                            "spot_perp_same_exchange",
                            base_ticker,
                            spot["exchange"],
                            spot["ticker"],
                            perpetual["exchange"],
                            perpetual["ticker"],
                            0,
                        ),
                        "base_ticker": base_ticker,
                        "pair_type": "spot_perp_same_exchange",
                        "leg1": {
                            "exchange": spot["exchange"],
                            "ticker": spot["ticker"],
                            "is_perp": False,
                        },
                        "leg2": {
                            "exchange": perpetual["exchange"],
                            "ticker": perpetual["ticker"],
                            "is_perp": True,
                        },
                        "direction_code": 0,
                        "enabled": True,
                        "overlap_start_ts": overlap_start,
                        "overlap_end_ts": overlap_end,
                    }
                    score = (
                        overlap_end - overlap_start,
                        min(spot["rows"], perpetual["rows"]),
                    )
                    candidates.append((score, pair))

    market_pair_keys = sorted(
        {
            (
                pair["leg1"]["exchange"],
                pair["leg1"]["ticker"],
                pair["leg2"]["exchange"],
                pair["leg2"]["ticker"],
            )
            for _, pair in candidates
        }
    )
    connection.execute(
        """
        CREATE TEMP TABLE candidate_streams(
            stream_id BIGINT,
            leg1_exchange VARCHAR,
            leg1_ticker VARCHAR,
            leg2_exchange VARCHAR,
            leg2_ticker VARCHAR
        )
        """
    )
    connection.executemany(
        "INSERT INTO candidate_streams VALUES (?, ?, ?, ?, ?)",
        [
            (stream_id, *market_pair_key)
            for stream_id, market_pair_key in enumerate(market_pair_keys)
        ],
    )
    connection.execute(
        """
        CREATE TEMP TABLE market_seconds AS
        SELECT
            lower(exchange) AS exchange,
            ticker,
            machine_ts_final // 1000 AS second
        FROM read_parquet(?)
        WHERE
            exchange IS NOT NULL
            AND ticker IS NOT NULL
            AND machine_ts_final IS NOT NULL
        GROUP BY 1, 2, 3
        """,
        [str(raw_l2_path)],
    )
    concurrent_rows = connection.execute(
        """
        WITH shared_seconds AS (
            SELECT streams.stream_id, left_seconds.second
            FROM candidate_streams AS streams
            INNER JOIN market_seconds AS left_seconds
                ON left_seconds.exchange = streams.leg1_exchange
                AND left_seconds.ticker = streams.leg1_ticker
            INNER JOIN market_seconds AS right_seconds
                ON right_seconds.exchange = streams.leg2_exchange
                AND right_seconds.ticker = streams.leg2_ticker
                AND right_seconds.second = left_seconds.second
        ),
        numbered AS (
            SELECT
                stream_id,
                second,
                second - row_number() OVER (
                    PARTITION BY stream_id ORDER BY second
                ) AS run_id
            FROM shared_seconds
        ),
        runs AS (
            SELECT
                stream_id,
                min(second)::BIGINT AS start_second,
                max(second)::BIGINT AS end_second,
                count(*)::BIGINT AS concurrent_seconds
            FROM numbered
            GROUP BY stream_id, run_id
        ),
        ranked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY stream_id
                    ORDER BY concurrent_seconds DESC, start_second
                ) AS rank
            FROM runs
        )
        SELECT stream_id, start_second, end_second, concurrent_seconds
        FROM ranked
        WHERE rank = 1
        """
    ).fetchall()
    concurrent_by_market_pair = {
        market_pair_keys[int(stream_id)]: {
            "start_ts": int(start_second) * 1000,
            "end_ts": (int(end_second) + 1) * 1000,
            "seconds": int(concurrent_seconds),
        }
        for stream_id, start_second, end_second, concurrent_seconds in concurrent_rows
    }

    concurrent_candidates: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    for old_score, pair in candidates:
        market_pair_key = (
            pair["leg1"]["exchange"],
            pair["leg1"]["ticker"],
            pair["leg2"]["exchange"],
            pair["leg2"]["ticker"],
        )
        concurrent = concurrent_by_market_pair.get(market_pair_key)
        if (
            concurrent is None
            or concurrent["seconds"] < min_concurrent_seconds
        ):
            continue
        overlap_start = max(
            int(pair["overlap_start_ts"]),
            int(concurrent["start_ts"]),
        )
        overlap_end = min(
            int(pair["overlap_end_ts"]),
            int(concurrent["end_ts"]),
        )
        if overlap_end <= overlap_start:
            continue
        pair["overlap_start_ts"] = overlap_start
        pair["overlap_end_ts"] = overlap_end
        pair["concurrent_seconds"] = concurrent["seconds"]
        concurrent_candidates.append(
            ((concurrent["seconds"], old_score[0], old_score[1]), pair)
        )
    candidates = concurrent_candidates

    def candidate_sort_key(
        item: tuple[tuple[int, int, int], dict[str, Any]],
    ) -> tuple[tuple[int, int, int], str, str]:
        return (
            item[0],
            item[1]["base_ticker"],
            item[1]["pair_id"],
        )

    pair_types = ["perp_perp_cross_exchange"]
    if include_spot_perp:
        pair_types.append("spot_perp_same_exchange")
    selected_pairs: list[dict[str, Any]] = []
    common_overlap_start: int | None = None
    common_overlap_end: int | None = None
    for pair_type in pair_types:
        candidates_for_type = [
            item for item in candidates if item[1]["pair_type"] == pair_type
        ]
        candidates_for_type.sort(key=candidate_sort_key, reverse=True)
        selected_market_pairs = 0
        seen_market_pairs: set[tuple[str, str, str, str]] = set()
        for _, pair in candidates_for_type:
            market_pair_key = (
                pair["leg1"]["exchange"],
                pair["leg1"]["ticker"],
                pair["leg2"]["exchange"],
                pair["leg2"]["ticker"],
            )
            if market_pair_key in seen_market_pairs:
                continue
            if selected_market_pairs >= max_pairs:
                break
            pair_overlap_start = int(pair["overlap_start_ts"])
            pair_overlap_end = int(pair["overlap_end_ts"])
            next_common_start = (
                pair_overlap_start
                if common_overlap_start is None
                else max(common_overlap_start, pair_overlap_start)
            )
            next_common_end = (
                pair_overlap_end
                if common_overlap_end is None
                else min(common_overlap_end, pair_overlap_end)
            )
            minimum_common_duration_ms = max(
                0,
                (min_concurrent_seconds - 1) * 1000,
            )
            if next_common_end - next_common_start < minimum_common_duration_ms:
                continue
            seen_market_pairs.add(market_pair_key)
            selected_market_pairs += 1
            common_overlap_start = next_common_start
            common_overlap_end = next_common_end
            selected_pairs.extend(
                candidate_pair
                for _, candidate_pair in candidates_for_type
                if (
                    candidate_pair["leg1"]["exchange"],
                    candidate_pair["leg1"]["ticker"],
                    candidate_pair["leg2"]["exchange"],
                    candidate_pair["leg2"]["ticker"],
                )
                == market_pair_key
            )
    if not selected_pairs:
        raise RuntimeError(
            "No replay pairs have a continuous concurrent L2 window of at least "
            f"{min_concurrent_seconds} seconds"
        )

    selected_market_keys = sorted(
        {
            (pair[leg]["exchange"], pair[leg]["ticker"])
            for pair in selected_pairs
            for leg in ("leg1", "leg2")
        }
    )
    connection.execute(
        "CREATE TEMP TABLE selected_markets(exchange VARCHAR, ticker VARCHAR)"
    )
    connection.executemany(
        "INSERT INTO selected_markets VALUES (?, ?)",
        selected_market_keys,
    )
    source_sql = str(raw_l2_path).replace("'", "''")
    output_sql = str(sorted_l2_path).replace("'", "''")
    connection.execute(
        f"""
        COPY (
            SELECT source.*
            FROM read_parquet('{source_sql}') AS source
            INNER JOIN selected_markets
                ON lower(source.exchange) = selected_markets.exchange
                AND source.ticker = selected_markets.ticker
            ORDER BY source.machine_ts_final, lower(source.exchange), source.ticker
        )
        TO '{output_sql}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    ohlcv_rows_count = 0
    ohlcv_market_count = 0
    if ohlcv_path is not None:
        ohlcv_source_sql = str(ohlcv_path).replace("'", "''")
        ohlcv_output_sql = str(sorted_ohlcv_path).replace("'", "''")
        connection.execute(
            f"""
            COPY (
                SELECT
                    source.*,
                    source.ts
                        + CASE source.tf
                            WHEN '1m' THEN 60000
                            WHEN '5m' THEN 300000
                          END AS available_ts
                FROM read_parquet('{ohlcv_source_sql}') AS source
                INNER JOIN selected_markets
                    ON lower(source.exchange) = selected_markets.exchange
                    AND source.symbol = selected_markets.ticker
                WHERE source.tf IN ('1m', '5m')
                ORDER BY
                    available_ts,
                    lower(source.exchange),
                    source.symbol,
                    source.tf,
                    source.ts
            )
            TO '{ohlcv_output_sql}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """
        )
        ohlcv_rows_count, ohlcv_market_count = connection.execute(
            """
            SELECT
                count(*)::BIGINT,
                count(DISTINCT (lower(exchange), symbol))::BIGINT
            FROM read_parquet(?)
            """,
            [str(sorted_ohlcv_path)],
        ).fetchone()
    rows, min_ts, max_ts, market_count = connection.execute(
        """
        SELECT
            count(*)::BIGINT,
            min(machine_ts_final)::BIGINT,
            max(machine_ts_final)::BIGINT,
            count(DISTINCT (lower(exchange), ticker))::BIGINT
        FROM read_parquet(?)
        """,
        [str(sorted_l2_path)],
    ).fetchone()
    connection.close()

    public_pairs = [
        {
            key: value
            for key, value in pair.items()
            if key
            not in {
                "overlap_start_ts",
                "overlap_end_ts",
                "concurrent_seconds",
            }
        }
        for pair in selected_pairs
    ]
    assert common_overlap_start is not None
    assert common_overlap_end is not None
    replay_start_ts = common_overlap_start
    replay_end_ts = common_overlap_end
    pairs_path.write_text(
        json.dumps(public_pairs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "raw_l2_path": str(raw_l2_path),
        "ohlcv_path": str(ohlcv_path) if ohlcv_path is not None else None,
        "sorted_l2_path": str(sorted_l2_path),
        "sorted_ohlcv_path": (
            str(sorted_ohlcv_path) if ohlcv_path is not None else None
        ),
        "pairs_path": str(pairs_path),
        "rows": int(rows),
        "markets": int(market_count),
        "ohlcv_rows": int(ohlcv_rows_count),
        "ohlcv_markets": int(ohlcv_market_count),
        "pairs": len(public_pairs),
        "max_market_pairs_per_enabled_type": max_pairs,
        "pair_types": pair_types,
        "min_ts": int(min_ts),
        "max_ts": int(max_ts),
        "replay_start_ts": replay_start_ts,
        "replay_end_ts": replay_end_ts,
        "min_concurrent_seconds": min_concurrent_seconds,
        "selected_min_concurrent_seconds": min(
            int(pair["concurrent_seconds"]) for pair in selected_pairs
        ),
        "selected_pairs_share_replay_window": True,
        "decision_ms": 100,
        "min_ohlcv_rows_per_tf": min_ohlcv_rows_per_tf,
        "all_source_columns_preserved": True,
        "sort_order": ["machine_ts_final", "exchange", "ticker"],
        "ohlcv_sort_order": [
            "available_ts",
            "exchange",
            "symbol",
            "tf",
            "ts",
        ],
        "ohlcv_is_causal": ohlcv_path is not None,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    """CLI подготовки replay-артефактов."""

    parser = argparse.ArgumentParser(
        description="Подготовить компактный отсортированный L2 replay"
    )
    parser.add_argument("--l2", type=Path, default=Path("/data/l2_raw.parquet"))
    parser.add_argument("--ohlcv", type=Path, default=Path("/data/ohlcv_raw.parquet"))
    parser.add_argument("--output", type=Path, default=Path("/prepared"))
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--min-ohlcv-rows-per-tf", type=int, default=128)
    parser.add_argument("--min-concurrent-seconds", type=int, default=900)
    parser.add_argument("--include-spot-perp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = prepare_replay(
        raw_l2_path=args.l2,
        ohlcv_path=args.ohlcv,
        output_dir=args.output,
        max_pairs=args.max_pairs,
        include_spot_perp=args.include_spot_perp,
        min_ohlcv_rows_per_tf=args.min_ohlcv_rows_per_tf,
        min_concurrent_seconds=args.min_concurrent_seconds,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
