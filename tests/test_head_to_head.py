from __future__ import annotations

import json
from pathlib import Path

from bridgeengine.benchmark.head_to_head import (
    CONDITIONS,
    build_head_to_head_plan,
    compare_snapshot_to_lewm_manifest,
    estimate_runtime,
)
from bridgeengine.ingest import ingest_bridge_v2


def test_head_to_head_confirms_shared_manifest(tmp_path: Path) -> None:
    result = ingest_bridge_v2(source="synthetic", episodes=4, data_root=tmp_path)
    manifest = [{"episode_index": idx} for idx in range(4)]
    manifest_path = tmp_path / "manifest_100.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    shared = compare_snapshot_to_lewm_manifest(
        result["snapshot_id"],
        data_root=tmp_path,
        lewm_manifest=manifest_path,
    )

    assert shared["same_episode_set"] is True
    assert shared["snapshot_episode_count"] == 4
    assert shared["lewm_manifest_episode_count"] == 4
    assert shared["snapshot_only"] == []
    assert shared["lewm_only"] == []


def test_head_to_head_plan_writes_stop_rule_and_splits(tmp_path: Path) -> None:
    result = ingest_bridge_v2(source="synthetic", episodes=6, data_root=tmp_path)
    manifest_path = tmp_path / "manifest_100.json"
    manifest_path.write_text(json.dumps([{"episode_index": idx} for idx in range(6)]), encoding="utf-8")
    h5_path = tmp_path / "bridgedata_v2_100ep.h5"
    h5_path.write_bytes(b"")
    run_dir = tmp_path / "lewm_run"
    run_dir.mkdir()
    (run_dir / "ablation_results.json").write_text(
        json.dumps(
            [
                {"condition": "A", "seed": 42, "status": "ok", "elapsed_sec": 10.0},
                {"condition": "B", "seed": 42, "status": "ok", "elapsed_sec": 12.0},
                {"condition": "D", "seed": 42, "status": "ok", "elapsed_sec": 14.0},
                {"condition": "E", "seed": 42, "status": "ok", "elapsed_sec": 16.0},
            ]
        ),
        encoding="utf-8",
    )
    bridge_csv = tmp_path / "scale_curve_results.csv"
    bridge_csv.write_text(
        "scale_n,family,seed,wall_clock_seconds_training\n"
        "4,baseline,0,1.0\n"
        "4,rich_text,0,2.0\n"
        "4,baseline,1,1.5\n"
        "4,rich_text,1,2.5\n",
        encoding="utf-8",
    )

    plan = build_head_to_head_plan(
        snapshot_id=result["snapshot_id"],
        output_dir=tmp_path / "head_to_head",
        data_root=tmp_path,
        sizes=(4, 6),
        heldout_count=2,
        split_seed=0,
        train_seeds=(42, 137, 256),
        lewm_manifest=manifest_path,
        lewm_h5=h5_path,
        lewm_run_dirs=(run_dir,),
        bridgeengine_scale_csv=bridge_csv,
    )

    assert plan["shared_episode_check"]["same_episode_set"] is True
    assert "annotation-strategy" in plan["mechanism_disclosure"]
    assert "fixed-split evaluator" in " ".join(plan["blocking_notes"])
    assert {condition["name"] for condition in plan["conditions"]} == {condition.name for condition in CONDITIONS}
    assert Path(plan["scale_plan_file"]).exists()
    assert (tmp_path / "head_to_head" / "head_to_head_plan.json").exists()
    assert (tmp_path / "head_to_head" / "runtime_estimate.md").exists()


def test_runtime_estimate_uses_cached_lewm_and_bridgeengine_csv(tmp_path: Path) -> None:
    run_dir = tmp_path / "lewm"
    run_dir.mkdir()
    (run_dir / "ablation_results.json").write_text(
        json.dumps(
            [
                {"condition": "A", "seed": 42, "status": "ok", "elapsed_sec": 100.0},
                {"condition": "B", "seed": 42, "status": "ok", "elapsed_sec": 200.0},
                {"condition": "D", "seed": 42, "status": "ok", "elapsed_sec": 300.0},
                {"condition": "E", "seed": 42, "status": "ok", "elapsed_sec": 400.0},
            ]
        ),
        encoding="utf-8",
    )
    bridge_csv = tmp_path / "bridge.csv"
    bridge_csv.write_text(
        "scale_n,family,seed,wall_clock_seconds_training\n"
        "25,baseline,0,5.0\n"
        "25,baseline,1,7.0\n",
        encoding="utf-8",
    )

    estimate = estimate_runtime(
        sizes=(25, 100),
        train_seeds=(42, 137, 256),
        lewm_run_dirs=(run_dir,),
        bridgeengine_scale_csv=bridge_csv,
    )

    assert estimate["lewm_cv_aux"]["cached_rows_found"] == 4
    assert estimate["lewm_cv_aux"]["cached_100_seconds"] == 1000.0
    assert estimate["bridgeengine_pi07"]["observed_seed_count"] == 2
    assert estimate["bridgeengine_pi07"]["estimated_seconds"] == 18.0
