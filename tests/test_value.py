from __future__ import annotations

from pathlib import Path

import pandas as pd

from bridgeengine.ingest import ingest_bridge_v2
from bridgeengine.value import score_snapshot


def test_embedding_value_report_updates_snapshot_and_compression(tmp_path: Path) -> None:
    result = ingest_bridge_v2(source="synthetic", episodes=5, data_root=tmp_path)

    report = score_snapshot(
        result["snapshot_id"],
        method="embedding-distance",
        data_root=tmp_path,
        top_n=3,
        high_value_percentile=0.8,
    )

    snapshot_path = Path(result["snapshot_path"])
    episodes = pd.read_parquet(snapshot_path / "episodes.parquet")
    assert {"value_score", "value_percentile", "value_rank", "value_method", "value_score_version"} <= set(episodes.columns)
    assert episodes["value_score"].notna().all()
    assert episodes["value_rank"].min() == 1
    assert report.method == "embedding-distance"
    assert len(report.top_outliers) == 3
    assert report.compression["tiered_size_bytes"] > 0
    assert (snapshot_path / "value_report.json").exists()
    assert (snapshot_path / "value_compression" / "embedding-distance_p80" / "report.json").exists()
