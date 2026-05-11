from __future__ import annotations

from pathlib import Path

import pandas as pd

from bridgeengine.ingest import ingest_bridge_v2
from bridgeengine.orchestrate import run_labelers


def test_labelers_populate_four_families(tmp_path: Path) -> None:
    result = ingest_bridge_v2(source="synthetic", episodes=2, data_root=tmp_path)
    run_labelers(result["snapshot_id"], data_root=tmp_path)
    labels = pd.read_parquet(Path(result["snapshot_path"]) / "labels.parquet")
    assert len(labels) == 8
    assert set(labels["labeler_name"]) == {"captions", "masks", "depth", "tracks"}
    for payload_path in labels["label_payload_path"]:
        assert Path(payload_path).exists()

