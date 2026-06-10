from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bridgeengine.benchmark.diagnostics import summarize_diagnostics


def _write_run(run_dir: Path, *, condition: str, seed: int, payload: dict, windows: pd.DataFrame) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "diagnostics.json").write_text(json.dumps(payload), encoding="utf-8")
    windows.to_csv(run_dir / "diagnostics_windows.csv", index=False)


def test_summarize_diagnostics_builds_floor_and_motion(tmp_path: Path) -> None:
    runs = tmp_path / "runs" / "scale_25"
    # Two windows with distinct motion so the quantile binning has a low and high bin.
    p0_windows = pd.DataFrame(
        {
            "episode_id": ["ep_a", "ep_a"],
            "start_idx": [0, 1],
            "trained_mse": [0.10, 0.20],
            "copy_mse": [0.05, 0.40],
            "constant_mse": [1.0, 1.0],
            "motion": [0.05, 0.40],
        }
    )
    e_windows = p0_windows.copy()
    e_windows["trained_mse"] = [0.08, 0.16]  # condition beats P0 on both windows

    _write_run(
        runs / "P0_pi07_baseline_seed42",
        condition="P0_pi07_baseline",
        seed=42,
        payload={
            "condition": "P0_pi07_baseline",
            "scale_n": 25,
            "seed": 42,
            "mean_trained_mse": 0.15,
            "mean_copy_mse": 0.225,
            "mean_constant_mse": 1.0,
            "ratio_trained_over_copy": 0.6667,
            "ratio_trained_over_constant": 0.15,
            "r2_vs_constant": 0.85,
            "train_trained_mse": 0.05,
            "generalization_gap": -0.10,
            "cond_ctx_ratio_mean": 0.0,
            "cond_ctx_ratio_p50": 0.0,
            "cond_ctx_ratio_p90": 0.0,
            "distinct_feature_vectors": 1,
            "subgoal_coverage": 1.0,
        },
        windows=p0_windows,
    )
    _write_run(
        runs / "E_depth_tracks_seed42",
        condition="E_depth_tracks",
        seed=42,
        payload={
            "condition": "E_depth_tracks",
            "scale_n": 25,
            "seed": 42,
            "mean_trained_mse": 0.12,
            "mean_copy_mse": 0.225,
            "mean_constant_mse": 1.0,
            "ratio_trained_over_copy": 0.5333,
            "ratio_trained_over_constant": 0.12,
            "r2_vs_constant": 0.88,
            "train_trained_mse": 0.02,
            "generalization_gap": -0.10,
            "subgoal_coverage": 1.0,
        },
        windows=e_windows,
    )

    out = summarize_diagnostics(tmp_path, scales=(25,))
    floor = pd.read_csv(out["trivial_floor_csv"])
    assert set(floor["condition"]) == {"P0_pi07_baseline", "E_depth_tracks"}
    p0 = floor[floor["condition"] == "P0_pi07_baseline"].iloc[0]
    assert abs(float(p0["ratio_trained_over_copy"]) - 0.6667) < 1e-3

    motion = pd.read_csv(out["motion_bins_csv"])
    # E_depth_tracks should have a negative paired delta vs P0 in each populated bin.
    e_rows = motion[motion["condition"] == "E_depth_tracks"]
    assert (e_rows["mean_paired_delta_vs_p0"] < 0).all()


def test_summarize_diagnostics_conditioning_only_pi07(tmp_path: Path) -> None:
    runs = tmp_path / "runs" / "scale_25"
    windows = pd.DataFrame(
        {
            "episode_id": ["ep_a"],
            "start_idx": [0],
            "trained_mse": [0.1],
            "copy_mse": [0.1],
            "constant_mse": [1.0],
            "motion": [0.1],
        }
    )
    _write_run(
        runs / "B_depth_seed42",
        condition="B_depth",
        seed=42,
        payload={"condition": "B_depth", "scale_n": 25, "seed": 42, "mean_trained_mse": 0.1,
                 "mean_copy_mse": 0.1, "mean_constant_mse": 1.0, "ratio_trained_over_copy": 1.0,
                 "ratio_trained_over_constant": 0.1, "r2_vs_constant": 0.9, "train_trained_mse": 0.02,
                 "generalization_gap": -0.08, "subgoal_coverage": 1.0},
        windows=windows,
    )
    _write_run(
        runs / "P2_pi07_metadata_seed42",
        condition="P2_pi07_metadata",
        seed=42,
        payload={"condition": "P2_pi07_metadata", "scale_n": 25, "seed": 42, "mean_trained_mse": 0.1,
                 "mean_copy_mse": 0.1, "mean_constant_mse": 1.0, "ratio_trained_over_copy": 1.0,
                 "ratio_trained_over_constant": 0.1, "r2_vs_constant": 0.9, "train_trained_mse": 0.02,
                 "generalization_gap": -0.08, "subgoal_coverage": 0.0, "cond_ctx_ratio_mean": 0.05,
                 "cond_ctx_ratio_p50": 0.05, "cond_ctx_ratio_p90": 0.06, "distinct_feature_vectors": 12},
        windows=windows,
    )
    out = summarize_diagnostics(tmp_path, scales=(25,))
    conditioning = pd.read_csv(out["conditioning_csv"])
    # Only the pi0.7 condition should appear in the conditioning table.
    assert set(conditioning["condition"]) == {"P2_pi07_metadata"}
    assert abs(float(conditioning.iloc[0]["cond_ctx_ratio_mean"]) - 0.05) < 1e-9
