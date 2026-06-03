from __future__ import annotations

from pathlib import Path

import pandas as pd

from bridgeengine.ingest import ingest_bridge_v2
from bridgeengine.orchestrate import run_labelers


def test_labelers_populate_four_families(tmp_path: Path) -> None:
    result = ingest_bridge_v2(source="synthetic", episodes=2, data_root=tmp_path)
    run_labelers(result["snapshot_id"], data_root=tmp_path, allow_fallback=True)
    labels = pd.read_parquet(Path(result["snapshot_path"]) / "labels.parquet")
    assert {"subtask_segmenter", "episode_metadata", "subgoal_images"}.issubset(set(labels["labeler_name"]))
    assert len(labels) >= 8
    assert "segment_idx" in labels.columns
    assert "metadata_payload_json" in labels.columns
    assert "subgoal_image_path" in labels.columns
    for payload_path in labels["label_payload_path"]:
        assert Path(payload_path).exists()
    for subgoal_path in labels["subgoal_image_path"].dropna():
        assert Path(subgoal_path).exists()
