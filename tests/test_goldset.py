from __future__ import annotations

import json
from pathlib import Path

from bridgeengine.goldset import reliability_report, write_gold_template
from bridgeengine.ingest import ingest_bridge_v2
from bridgeengine.orchestrate import run_labelers


def test_gold_template_and_reliability_report(tmp_path: Path) -> None:
    result = ingest_bridge_v2(source="synthetic", episodes=2, data_root=tmp_path)
    run_labelers(result["snapshot_id"], data_root=tmp_path, vlm_backend="mock")
    gold_path = tmp_path / "gold.json"

    template = write_gold_template(result["snapshot_id"], gold_path, data_root=tmp_path)

    assert gold_path.exists()
    assert len(template["episodes"]) == 2
    first = template["episodes"][0]
    for subtask in first["gold"]["subtasks"]:
        subtask["accept_auto"] = True
    first["gold"]["metadata"]["accept_auto"] = True
    first["gold"]["metadata"]["quality"] = first["auto"]["metadata"]["quality"]
    first["gold"]["metadata"]["mistake"] = first["auto"]["metadata"]["mistake"]
    for subgoal, auto_subgoal in zip(first["gold"]["subgoals"], first["auto"]["subgoals"]):
        subgoal["accept_auto"] = True
        subgoal["frame_idx"] = auto_subgoal["frame_idx"]
    gold_path.write_text(json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = reliability_report(result["snapshot_id"], gold_path, data_root=tmp_path)

    assert report["reviewed_episode_count"] == 1
    assert report["subtask_boundary_temporal_iou_mean"] == 1.0
    assert report["quality_exact_agreement"] == 1.0
    assert report["quality_within_one_agreement"] == 1.0
    assert report["subgoal_selection_agreement"] == 1.0
    assert report["labeling_wall_clock_seconds_per_episode"] is not None
    assert "linear and shardable" in report["parallelism_note"]
