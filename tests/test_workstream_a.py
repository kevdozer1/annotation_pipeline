from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bridgeengine.ingest import ingest_bridge_v2
from bridgeengine.orchestrate import run_labelers
from bridgeengine.quality_gate import evaluate_snapshot_quality


def test_mock_backend_writes_two_stage_provenance(tmp_path: Path) -> None:
    result = ingest_bridge_v2(source="synthetic", episodes=2, data_root=tmp_path)
    run_labelers(result["snapshot_id"], data_root=tmp_path, vlm_backend="mock")

    labels = pd.read_parquet(Path(result["snapshot_path"]) / "labels.parquet")
    segment_row = labels[labels["labeler_name"] == "subtask_segmenter"].iloc[0]
    segment_payload = json.loads(Path(segment_row["label_payload_path"]).read_text(encoding="utf-8"))
    segment_provenance = json.loads(segment_row["provenance_json"])

    assert segment_payload["vlm_backend"] == "mock_vlm"
    assert segment_payload["stage_one_observations"]
    assert Path(segment_payload["raw_observation_output_path"]).exists()
    assert Path(segment_payload["raw_vlm_output_path"]).exists()
    assert segment_provenance["raw_observation_output_path"]

    metadata_row = labels[labels["labeler_name"] == "episode_metadata"].iloc[0]
    metadata_payload = json.loads(Path(metadata_row["label_payload_path"]).read_text(encoding="utf-8"))
    assert metadata_payload["vlm_backend"] == "mock_vlm"
    assert metadata_payload["metadata"]["reason"]


def test_quality_gate_rejects_repeated_subtask_text(tmp_path: Path) -> None:
    result = ingest_bridge_v2(source="synthetic", episodes=2, data_root=tmp_path)
    run_labelers(result["snapshot_id"], data_root=tmp_path, vlm_backend="mock")
    labels = pd.read_parquet(Path(result["snapshot_path"]) / "labels.parquet")
    row = labels[labels["labeler_name"] == "subtask_segmenter"].iloc[0]
    payload_path = Path(row["label_payload_path"])
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["segments"][1]["subtask_text"] = payload["segments"][0]["subtask_text"]
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = evaluate_snapshot_quality(Path(result["snapshot_path"]))

    assert not report.passed
    assert any(issue.check == "repeated_subtask_text" for issue in report.issues)


def test_quality_gate_rejects_metadata_score_reason_contradiction(tmp_path: Path) -> None:
    result = ingest_bridge_v2(source="synthetic", episodes=2, data_root=tmp_path)
    run_labelers(result["snapshot_id"], data_root=tmp_path, vlm_backend="mock")
    labels_path = Path(result["snapshot_path"]) / "labels.parquet"
    labels = pd.read_parquet(labels_path)
    idx = labels[labels["labeler_name"] == "episode_metadata"].index[0]
    labels.at[idx, "metadata_payload_json"] = json.dumps(
        {
            "speed": 10,
            "quality": 1,
            "mistake": False,
            "control_mode": "end_effector",
            "reason": "The robot successfully completed the placement.",
        },
        sort_keys=True,
    )
    labels.to_parquet(labels_path, index=False)

    report = evaluate_snapshot_quality(Path(result["snapshot_path"]))

    assert not report.passed
    assert any(issue.check == "score_reason_consistency" for issue in report.issues)


def test_quality_gate_rejects_ungrounded_object_text(tmp_path: Path) -> None:
    result = ingest_bridge_v2(source="synthetic", episodes=2, data_root=tmp_path)
    run_labelers(result["snapshot_id"], data_root=tmp_path, vlm_backend="mock")
    labels = pd.read_parquet(Path(result["snapshot_path"]) / "labels.parquet")
    row = labels[labels["labeler_name"] == "subtask_segmenter"].iloc[0]
    payload_path = Path(row["label_payload_path"])
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["segments"][0]["subtask_text"] = "grasp purple widget"
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = evaluate_snapshot_quality(Path(result["snapshot_path"]))

    assert not report.passed
    assert any(issue.check == "object_grounding" for issue in report.issues)


def test_quality_gate_ignores_function_words_and_visual_attributes_for_grounding(tmp_path: Path) -> None:
    result = ingest_bridge_v2(source="synthetic", episodes=2, data_root=tmp_path)
    run_labelers(result["snapshot_id"], data_root=tmp_path, vlm_backend="mock")
    labels = pd.read_parquet(Path(result["snapshot_path"]) / "labels.parquet")
    row = labels[labels["labeler_name"] == "subtask_segmenter"].iloc[0]
    payload_path = Path(row["label_payload_path"])
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["stage_one_observations"] = {
        "observations": [
            {
                "objects": ["cup", "sink"],
                "summary": "The gripper moves across the sink, beside the empty rack, and settles near a blue cup.",
            }
        ]
    }
    payload["segments"][0]["subtask_text"] = "move across the empty sink beside the blue cup before release"
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = evaluate_snapshot_quality(Path(result["snapshot_path"]))

    assert not any(issue.check == "object_grounding" for issue in report.issues)


def test_quality_gate_rejects_score_collapse_on_large_enough_snapshot(tmp_path: Path) -> None:
    result = ingest_bridge_v2(source="synthetic", episodes=8, data_root=tmp_path)
    run_labelers(result["snapshot_id"], data_root=tmp_path, vlm_backend="mock")

    report = evaluate_snapshot_quality(Path(result["snapshot_path"]))

    assert not report.passed
    assert any(issue.check == "score_dispersion" for issue in report.issues)


def test_benchmark_gate_error_mentions_specific_quality_issue(tmp_path: Path) -> None:
    result = ingest_bridge_v2(source="synthetic", episodes=8, data_root=tmp_path)
    run_labelers(result["snapshot_id"], data_root=tmp_path, vlm_backend="mock")

    from bridgeengine.benchmark.run_grid import run_grid

    with pytest.raises(RuntimeError, match="score_dispersion"):
        run_grid(result["snapshot_id"], data_root=tmp_path, output_dir=tmp_path / "bench")
