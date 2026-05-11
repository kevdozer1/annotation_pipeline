from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bridgeengine.ingest import ingest_bridge_v2


def test_ingest_writes_deterministic_snapshot(tmp_path: Path) -> None:
    first = ingest_bridge_v2(source="synthetic", episodes=3, data_root=tmp_path)
    second = ingest_bridge_v2(source="synthetic", episodes=3, data_root=tmp_path)
    assert first["snapshot_id"] == second["snapshot_id"]
    snapshot_path = Path(first["snapshot_path"])
    for name in ("manifest.json", "episodes.parquet", "steps.parquet", "sensors.parquet", "labels.parquet"):
        assert (snapshot_path / name).exists()
    manifest = json.loads((snapshot_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_episode_count"] == 3
    episodes = pd.read_parquet(snapshot_path / "episodes.parquet")
    assert len(episodes) == 3

