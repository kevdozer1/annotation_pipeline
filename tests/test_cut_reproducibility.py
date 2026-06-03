from __future__ import annotations

from pathlib import Path

from bridgeengine.export import BridgeCutDataset, export_cut
from bridgeengine.ingest import ingest_bridge_v2
from bridgeengine.orchestrate import run_labelers


def test_export_cut_is_reproducible(tmp_path: Path) -> None:
    result = ingest_bridge_v2(source="synthetic", episodes=3, data_root=tmp_path)
    run_labelers(result["snapshot_id"], data_root=tmp_path, allow_fallback=True)
    out = tmp_path / "cuts"
    export_cut(result["snapshot_id"], "TRUE", out, "cut_a", data_root=tmp_path)
    first_manifest = (out / "cut_a" / "manifest.json").read_bytes()
    first_list = (out / "cut_a" / "episode_list.txt").read_bytes()
    export_cut(result["snapshot_id"], "TRUE", out, "cut_a", data_root=tmp_path)
    assert first_manifest == (out / "cut_a" / "manifest.json").read_bytes()
    assert first_list == (out / "cut_a" / "episode_list.txt").read_bytes()
    dataset = BridgeCutDataset(out / "cut_a")
    sample = dataset[0]
    assert sample["actions"].ndim == 2
    assert sample["labels"]
