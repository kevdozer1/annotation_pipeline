from __future__ import annotations

from pathlib import Path

from bridgeengine.calibration import calibration_reliability, load_or_create_calibration_gold, review_summary, update_episode_review
from bridgeengine.apply_gold import apply_gold_scores_to_snapshot
from bridgeengine.ingest import ingest_bridge_v2
from bridgeengine.orchestrate import run_labelers
from bridgeengine.reliability_review import plan_reliability_review
from bridgeengine.review_gui import ReviewDataset


def test_calibration_gold_review_roundtrip(tmp_path: Path) -> None:
    result = ingest_bridge_v2(source="synthetic", episodes=2, data_root=tmp_path)
    run_labelers(result["snapshot_id"], data_root=tmp_path, vlm_backend="mock")
    gold_path = tmp_path / "calibration_gold.json"

    payload = load_or_create_calibration_gold(result["snapshot_id"], gold_path, data_root=tmp_path)
    episode_id = payload["episodes"][0]["episode_id"]
    auto_quality = payload["episodes"][0]["auto"]["metadata"]["quality"]

    update_episode_review(
        result["snapshot_id"],
        episode_id,
        curation_quality=auto_quality,
        mistake=False,
        reason="Score accepted during calibration.",
        review_notes="Looks usable.",
        accept_auto_metadata=True,
        gold_file=gold_path,
        data_root=tmp_path,
    )

    summary = review_summary(result["snapshot_id"], gold_path, data_root=tmp_path)
    reviewed = summary[summary["episode_id"] == episode_id].iloc[0]
    assert bool(reviewed["reviewed"]) is True
    assert reviewed["gold_score"] == auto_quality
    assert reviewed["notes"] == "Looks usable."

    report = calibration_reliability(result["snapshot_id"], gold_path, data_root=tmp_path)
    assert report["reviewed_episode_count"] == 1
    assert report["quality_exact_agreement"] == 1.0


def test_review_dataset_save_review_advances_to_next(tmp_path: Path) -> None:
    result = ingest_bridge_v2(source="synthetic", episodes=2, data_root=tmp_path)
    run_labelers(result["snapshot_id"], data_root=tmp_path, vlm_backend="mock")
    dataset = ReviewDataset(result["snapshot_id"], data_root=tmp_path)
    first_id = dataset.episode_ids[0]

    payload = dataset.episode_payload(first_id)
    assert payload["episode_id"] == first_id
    assert payload["segments"]

    saved = dataset.save_review(
        {
            "episode_id": first_id,
            "score": 4,
            "mistake": False,
            "reason": "Usable enough.",
            "notes": "Reviewed in browser GUI test.",
            "accept_auto_metadata": False,
            "accept_auto_subtasks": True,
            "accept_auto_subgoals": True,
        }
    )

    assert saved["saved"] is True
    assert saved["next_episode_id"] == dataset.episode_ids[1]
    assert saved["state"]["reviewed_count"] == 1


def test_apply_gold_scores_to_calibrated_snapshot(tmp_path: Path) -> None:
    result = ingest_bridge_v2(source="synthetic", episodes=2, data_root=tmp_path)
    run_labelers(result["snapshot_id"], data_root=tmp_path, vlm_backend="mock")
    gold_path = tmp_path / "calibration_gold.json"
    payload = load_or_create_calibration_gold(result["snapshot_id"], gold_path, data_root=tmp_path)
    episode_id = payload["episodes"][0]["episode_id"]

    update_episode_review(
        result["snapshot_id"],
        episode_id,
        curation_quality=2,
        mistake=True,
        reason="Downgraded by human review.",
        review_notes="The video is not useful enough.",
        gold_file=gold_path,
        data_root=tmp_path,
    )
    target_snapshot = result["snapshot_id"] + "_human"
    report = apply_gold_scores_to_snapshot(
        result["snapshot_id"],
        target_snapshot,
        gold_file=gold_path,
        data_root=tmp_path,
    )

    assert report["target_snapshot_id"] == target_snapshot
    assert report["changed_score_count"] >= 1
    target_labels = (tmp_path / "snapshots" / target_snapshot / "labels.parquet")
    assert target_labels.exists()
    import json
    import pandas as pd

    labels = pd.read_parquet(target_labels)
    row = labels[(labels["episode_id"] == episode_id) & (labels["labeler_name"] == "episode_metadata")].iloc[0]
    metadata = json.loads(row["metadata_payload_json"])
    assert metadata["quality"] == 2
    assert metadata["curation_quality"] == 2
    assert metadata["curation_keep"] is False
    assert metadata["human_calibrated"] is True
    assert row["snapshot_id"] == target_snapshot


def test_boundary_subgoal_review_subset_has_separate_completion(tmp_path: Path) -> None:
    result = ingest_bridge_v2(source="synthetic", episodes=2, data_root=tmp_path)
    run_labelers(result["snapshot_id"], data_root=tmp_path, vlm_backend="mock")
    gold_path = tmp_path / "calibration_gold.json"
    payload = load_or_create_calibration_gold(result["snapshot_id"], gold_path, data_root=tmp_path)
    first_id = payload["episodes"][0]["episode_id"]
    first_score = payload["episodes"][0]["auto"]["metadata"]["quality"]
    update_episode_review(
        result["snapshot_id"],
        first_id,
        curation_quality=first_score,
        mistake=False,
        gold_file=gold_path,
        data_root=tmp_path,
    )

    subset_path = tmp_path / "boundary_subset.json"
    plan = plan_reliability_review(
        result["snapshot_id"],
        subset_path,
        count=1,
        gold_file=gold_path,
        data_root=tmp_path,
    )
    dataset = ReviewDataset(
        result["snapshot_id"],
        data_root=tmp_path,
        gold_file=gold_path,
        episode_file=subset_path,
        review_goal="boundary_subgoal",
    )
    assert plan["episode_count"] == 1
    assert dataset.state()["episode_count"] == 1
    assert dataset.state()["reviewed_count"] == 0

    episode_id = plan["episode_ids"][0]
    score = dataset.episode_payload(episode_id)["review"]["gold_score"]
    saved = dataset.save_review(
        {
            "episode_id": episode_id,
            "score": score,
            "mistake": False,
            "reason": "",
            "notes": "",
            "accept_auto_metadata": False,
            "accept_auto_subtasks": True,
            "accept_auto_subgoals": True,
        }
    )
    assert saved["state"]["reviewed_count"] == 1
