from __future__ import annotations

from pathlib import Path

from bridgeengine.ingest import ingest_bridge_v2
from bridgeengine.orchestrate import run_labelers
from bridgeengine.orchestrate.runner import run_perceptive_labelers
from bridgeengine.perceptive_status import perceptive_status


def test_perceptive_status_rejects_synthetic_fallbacks(tmp_path: Path) -> None:
    result = ingest_bridge_v2(source="synthetic", episodes=2, data_root=tmp_path)
    run_labelers(result["snapshot_id"], data_root=tmp_path, vlm_backend="mock")
    run_perceptive_labelers(result["snapshot_id"], data_root=tmp_path)

    report = perceptive_status(result["snapshot_id"], data_root=tmp_path)

    assert report["ready_for_head_to_head"] is False
    assert report["labelers"]["perceptive_masks"]["synthetic_rows"] == 2
    assert report["labelers"]["perceptive_depth"]["synthetic_rows"] == 2
    assert report["labelers"]["perceptive_tracks"]["synthetic_rows"] == 2
