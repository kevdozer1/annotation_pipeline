from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from bridgeengine.paths import data_root as resolve_data_root


def connect_snapshot(snapshot_id: str, data_root: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    root = resolve_data_root(data_root)
    snapshot_path = root / "snapshots" / snapshot_id
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")
    con = duckdb.connect(database=":memory:")
    for table in ("episodes", "steps", "sensors"):
        con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet({_sql_string(snapshot_path / f'{table}.parquet')})")
    con.execute(
        f"""
        CREATE VIEW labels AS
        SELECT
          *,
          provenance_json AS provenance
        FROM read_parquet({_sql_string(snapshot_path / 'labels.parquet')})
        """
    )
    return con


def run_query(
    snapshot_id: str,
    sql: str,
    data_root: str | Path | None = None,
) -> pd.DataFrame:
    con = connect_snapshot(snapshot_id, data_root)
    try:
        return con.execute(sql).fetchdf()
    finally:
        con.close()


def preview(snapshot_id: str, data_root: str | Path | None = None) -> dict[str, Any]:
    con = connect_snapshot(snapshot_id, data_root)
    try:
        counts = {
            table: int(con.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
            for table in ("episodes", "steps", "sensors", "labels")
        }
        coverage = con.execute(
            """
            SELECT labeler_name, count(*) AS rows, avg(confidence) AS avg_confidence
            FROM labels
            GROUP BY labeler_name
            ORDER BY labeler_name
            """
        ).fetchdf()
        result = {"snapshot_id": snapshot_id, "row_counts": counts, "label_coverage": coverage}
        print(f"Snapshot: {snapshot_id}")
        for table, count in counts.items():
            print(f"  {table}: {count}")
        print(coverage.to_string(index=False))
        return result
    finally:
        con.close()


def demo_queries() -> dict[str, str]:
    return {
        "mask_coverage": """
            SELECT e.episode_id, e.language_instruction, l.confidence
            FROM episodes e
            JOIN labels l ON l.episode_id = e.episode_id
            WHERE l.labeler_name = 'masks'
              AND l.confidence > 0.7
            ORDER BY l.confidence DESC
        """,
        "caption_put": """
            SELECT e.episode_id, l.label_payload_path
            FROM episodes e
            JOIN labels l ON l.episode_id = e.episode_id
            WHERE l.labeler_name = 'captions'
              AND json_extract_string(l.provenance, '$.caption_text') LIKE '%put%'
            ORDER BY e.episode_id
        """,
        "caption_depth_range": """
            SELECT
              e.episode_id,
              e.language_instruction,
              CAST(json_extract_string(d.provenance, '$.depth_max') AS DOUBLE)
                - CAST(json_extract_string(d.provenance, '$.depth_min') AS DOUBLE) AS depth_dynamic_range,
              json_extract_string(c.provenance, '$.caption_text') AS caption_text
            FROM episodes e
            JOIN labels c ON c.episode_id = e.episode_id AND c.labeler_name = 'captions'
            JOIN labels d ON d.episode_id = e.episode_id AND d.labeler_name = 'depth'
            ORDER BY depth_dynamic_range DESC
        """,
        "labeler_success_counts": """
            SELECT
              labeler_name,
              count(*) AS episode_rows,
              count(confidence) AS rows_with_confidence,
              avg(confidence) AS avg_confidence
            FROM labels
            GROUP BY labeler_name
            ORDER BY labeler_name
        """,
        "provenance_trace": """
            SELECT
              e.episode_id,
              l.labeler_name,
              l.labeler_version,
              json_extract_string(l.provenance, '$.input_sha256') AS input_sha256,
              CAST(json_extract_string(l.provenance, '$.wall_clock_seconds') AS DOUBLE) AS seconds
            FROM episodes e
            JOIN labels l ON l.episode_id = e.episode_id
            WHERE e.episode_id = (SELECT episode_id FROM episodes ORDER BY episode_id LIMIT 1)
            ORDER BY l.labeler_name
        """,
    }


def _sql_string(path: Path) -> str:
    text = str(path).replace("\\", "/").replace("'", "''")
    return f"'{text}'"
