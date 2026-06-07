from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bridgeengine.apply_gold import apply_gold_scores_to_snapshot
from bridgeengine.derive_subgoals import DERIVED_SUBGOAL_SOURCE, derive_gold_subgoals_from_boundaries
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


def test_boundary_derived_subgoal_and_gold_snapshot_application(tmp_path: Path) -> None:
    result = ingest_bridge_v2(source="synthetic", episodes=1, data_root=tmp_path)
    source_snapshot = result["snapshot_id"]
    run_labelers(source_snapshot, data_root=tmp_path, vlm_backend="mock")
    gold_path = tmp_path / "gold.json"
    template = write_gold_template(source_snapshot, gold_path, data_root=tmp_path)

    entry = template["episodes"][0]
    auto_segment = entry["auto"]["subtasks"][0]
    auto_end = int(auto_segment["end_step"])
    num_steps = int(entry["num_steps"])
    human_end = min(auto_end + 2, num_steps - 1)
    if human_end == auto_end:
        human_end = max(0, auto_end - 1)

    entry["gold"]["metadata"]["quality"] = 5
    entry["gold"]["metadata"]["curation_quality"] = 5
    entry["gold"]["metadata"]["mistake"] = False
    entry["gold"]["metadata"]["accept_auto"] = False
    entry["gold"]["subtasks"][0]["accept_auto"] = False
    entry["gold"]["subtasks"][0]["start_step"] = int(auto_segment["start_step"])
    entry["gold"]["subtasks"][0]["end_step"] = human_end
    entry["gold"]["subtasks"][0]["subtask_text"] = auto_segment["subtask_text"]
    gold_path.write_text(json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    derive_report = derive_gold_subgoals_from_boundaries(source_snapshot, gold_path, data_root=tmp_path)
    updated = json.loads(gold_path.read_text(encoding="utf-8"))
    gold_subgoal = updated["episodes"][0]["gold"]["subgoals"][0]
    reliability = reliability_report(source_snapshot, gold_path, data_root=tmp_path)

    assert derive_report["updated_subgoals"] == 1
    assert gold_subgoal["frame_idx"] == human_end
    assert gold_subgoal["source"] == DERIVED_SUBGOAL_SOURCE
    assert reliability["subgoal_selection_agreement"] == 0.0

    target_snapshot = f"{source_snapshot}_human_gold"
    apply_report = apply_gold_scores_to_snapshot(
        source_snapshot=source_snapshot,
        target_snapshot=target_snapshot,
        gold_file=gold_path,
        data_root=tmp_path,
    )
    target_path = tmp_path / "snapshots" / target_snapshot
    labels = pd.read_parquet(target_path / "labels.parquet")
    subtask_row = labels[labels["labeler_name"] == "subtask_segmenter"].iloc[0]
    subgoal_row = labels[labels["labeler_name"] == "subgoal_images"].sort_values("segment_idx").iloc[0]
    subtask_payload = json.loads(Path(subtask_row["label_payload_path"]).read_text(encoding="utf-8"))
    subgoal_payload = json.loads(Path(subgoal_row["label_payload_path"]).read_text(encoding="utf-8"))

    assert apply_report["applied_boundary_episode_count"] == 1
    assert subtask_payload["segments"][0]["end_step"] == human_end
    assert subgoal_payload["frame_idx"] == human_end
    assert subgoal_payload["source"] == DERIVED_SUBGOAL_SOURCE
    assert target_snapshot in str(subgoal_row["subgoal_image_path"])
    assert Path(subgoal_row["subgoal_image_path"]).exists()
