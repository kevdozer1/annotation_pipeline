from __future__ import annotations

import argparse
import time
from pathlib import Path

from .duckdb_helpers import demo_queries, preview, run_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BridgeEngine demo DuckDB queries.")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--max-ms", type=float, default=500.0)
    args = parser.parse_args()
    data_root = Path(args.data_root) if args.data_root else None
    preview(args.snapshot, data_root)
    for name, sql in demo_queries().items():
        t0 = time.perf_counter()
        df = run_query(args.snapshot, sql, data_root)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if df.empty:
            raise RuntimeError(f"Demo query {name!r} returned no rows")
        if dt_ms > args.max_ms:
            raise RuntimeError(f"Demo query {name!r} took {dt_ms:.1f}ms, above {args.max_ms:.1f}ms")
        print(f"{name}: {len(df)} rows in {dt_ms:.1f}ms")


if __name__ == "__main__":
    main()

