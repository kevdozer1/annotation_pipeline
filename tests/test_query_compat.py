from __future__ import annotations

from pathlib import Path

import pandas as pd

from bridgeengine.ingest import ingest_bridge_v2
from bridgeengine.orchestrate import run_labelers
from bridgeengine.query import demo_queries, run_query


def test_demo_queries_tolerate_pre_pivot_label_schema(tmp_path: Path) -> None:
    result = ingest_bridge_v2(source="synthetic", episodes=2, data_root=tmp_path)
    run_labelers(result["snapshot_id"], data_root=tmp_path, allow_fallback=True)

    labels_path = Path(result["snapshot_path"]) / "labels.parquet"
    labels = pd.read_parquet(labels_path).drop(
        columns=["segment_idx", "metadata_payload_json", "subgoal_image_path"]
    )
    labels.to_parquet(labels_path, index=False)

    for sql in demo_queries().values():
        result_df = run_query(result["snapshot_id"], sql, data_root=tmp_path)
        assert isinstance(result_df, pd.DataFrame)
