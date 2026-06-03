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
        "subtask_coverage": """
            SELECT e.episode_id, e.language_instruction, l.confidence
            FROM episodes e
            JOIN labels l ON l.episode_id = e.episode_id
            WHERE l.labeler_name = 'subtask_segmenter'
              AND l.confidence > 0.7
            ORDER BY l.confidence DESC
        """,
        "metadata_quality": """
            SELECT
              e.episode_id,
              e.language_instruction,
              CAST(json_extract_string(l.metadata_payload_json, '$.quality') AS INTEGER) AS quality,
              CAST(json_extract_string(l.metadata_payload_json, '$.mistake') AS BOOLEAN) AS mistake,
              json_extract_string(l.metadata_payload_json, '$.control_mode') AS control_mode
            FROM episodes e
            JOIN labels l ON l.episode_id = e.episode_id
            WHERE l.labeler_name = 'episode_metadata'
            ORDER BY quality DESC, e.episode_id
        """,
        "subgoal_paths": """
            SELECT
              e.episode_id,
              l.segment_idx,
              e.language_instruction,
              l.subgoal_image_path
            FROM episodes e
            JOIN labels l ON l.episode_id = e.episode_id
            WHERE l.labeler_name = 'subgoal_images'
            ORDER BY e.episode_id, l.segment_idx
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
        "pi07_prompt_trace": """
            SELECT
              e.episode_id,
              'Task: ' || e.language_instruction ||
              '. Speed: ' || json_extract_string(m.metadata_payload_json, '$.speed') ||
              '. Quality: ' || json_extract_string(m.metadata_payload_json, '$.quality') || '/5' ||
              '. Mistake: ' || json_extract_string(m.metadata_payload_json, '$.mistake') ||
              '. Control Mode: ' || json_extract_string(m.metadata_payload_json, '$.control_mode') || '.' AS prompt_prefix,
              s.label_payload_path AS subtask_segments_json
            FROM episodes e
            JOIN labels s ON s.episode_id = e.episode_id AND s.labeler_name = 'subtask_segmenter'
            JOIN labels m ON m.episode_id = e.episode_id AND m.labeler_name = 'episode_metadata'
            WHERE e.episode_id = (SELECT episode_id FROM episodes ORDER BY episode_id LIMIT 1)
        """,
    }


def _sql_string(path: Path) -> str:
    text = str(path).replace("\\", "/").replace("'", "''")
    return f"'{text}'"
