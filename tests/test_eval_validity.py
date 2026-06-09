from __future__ import annotations

import numpy as np
import pandas as pd

from bridgeengine.benchmark.leak_power import _paired_deltas, _paired_summary, _subgoal_leak_bins
from bridgeengine.benchmark.train_lewm import WindowRecord
from bridgeengine.benchmark.window_eval import (
    TargetLatentMoments,
    normalized_window_errors,
    window_rows,
)


def test_target_latent_moments_variance_and_norm() -> None:
    moments = TargetLatentMoments()
    moments.update_from_numpy(np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64))
    moments.update_from_numpy(np.array([[0.0, 4.0], [2.0, 4.0]], dtype=np.float64))
    stats = moments.finalize()
    # dim0 var = 1.0, dim1 var = 4.0 -> mean per-dim variance = 2.5
    assert abs(stats["heldout_target_variance"] - 2.5) < 1e-9
    # mean squared L2 norm over the four vectors = (0 + 4 + 16 + 20) / 4 = 10
    assert abs(stats["heldout_target_mean_sq_norm"] - 10.0) < 1e-9


def test_normalized_window_errors_guards_nonpositive_variance() -> None:
    assert normalized_window_errors([1.0, 2.0], 2.0) == [0.5, 1.0]
    assert all(np.isnan(x) for x in normalized_window_errors([1.0, 2.0], 0.0))


def test_window_rows_adds_norm_column_only_when_provided() -> None:
    record = WindowRecord(
        episode_id="ep",
        start_idx=0,
        task="t",
        subtask_text="s",
        segment_idx=0,
        metadata={},
        subgoal_image_path="x.jpg",
        segment_start_step=0,
        segment_end_step=5,
    )
    without = window_rows([record], [0.4], history_size=3)
    assert "norm_sq_err" not in without[0]
    with_norm = window_rows([record], [0.4], history_size=3, norm_sq_err=[0.2])
    assert with_norm[0]["norm_sq_err"] == 0.2


def _paired_frame(with_norm: bool) -> pd.DataFrame:
    rows = []
    for seed in (1, 2):
        for episode in ("ep_a", "ep_b"):
            for start_idx in (0, 1):
                base = {
                    "scale_n": 25,
                    "seed": seed,
                    "episode_id": episode,
                    "start_idx": start_idx,
                    "segment_idx": 0,
                    "subgoal_mask": 1.0,
                    "dist_to_segment_end": 5,
                }
                ref = {**base, "condition": "P0_pi07_baseline", "sq_err": 1.0}
                cond = {**base, "condition": "B_depth", "sq_err": 0.5}
                if with_norm:
                    # Reference variance 1.0, condition variance 0.5: the raw win
                    # halves, but normalized errors are identical -> no normalized gain.
                    ref["norm_sq_err"] = 1.0
                    cond["norm_sq_err"] = 1.0
                rows.extend([ref, cond])
    return pd.DataFrame(rows)


def test_paired_summary_reports_normalized_delta() -> None:
    deltas = _paired_deltas(_paired_frame(with_norm=True), reference="P0_pi07_baseline")
    assert "delta_norm_sq_err" in deltas.columns
    summary = _paired_summary(deltas, bootstrap_reps=50, seed=0)
    row = summary.iloc[0]
    assert row["condition"] == "B_depth"
    # Raw shows a -0.5 win; the variance-normalized delta collapses to ~0.
    assert abs(row["mean_delta_sq_err"] + 0.5) < 1e-9
    assert abs(row["mean_delta_norm_sq_err"]) < 1e-9


def test_paired_deltas_without_norm_stays_backward_compatible() -> None:
    deltas = _paired_deltas(_paired_frame(with_norm=False), reference="P0_pi07_baseline")
    assert "delta_norm_sq_err" not in deltas.columns
    summary = _paired_summary(deltas, bootstrap_reps=50, seed=0)
    assert "mean_delta_norm_sq_err" not in summary.columns


def test_subgoal_leak_bins_emit_relative_columns() -> None:
    rows = []
    for start_idx, dist in enumerate([0, 1, 2, 11, 12, 13]):
        rows.append(
            {
                "scale_n": 25,
                "seed": 42,
                "condition": "P3_pi07_subgoal",
                "reference": "P0_pi07_baseline",
                "episode_id": "ep_a",
                "start_idx": start_idx,
                "segment_idx": 0,
                "subgoal_mask": 1.0,
                "dist_to_segment_end": dist,
                "sq_err": 0.5,
                "ref_sq_err": 1.0,
                "delta_sq_err": -0.5,
                "advantage_sq_err": 0.5,
            }
        )
    leak = _subgoal_leak_bins(pd.DataFrame(rows))
    near_far = leak[leak["bin"] == "near_vs_far"].iloc[0]
    assert "near_advantage_rel" in near_far.index
    # advantage 0.5 over ref 1.0 -> relative advantage 0.5 in both near and far bins.
    assert abs(float(near_far["near_advantage_rel"]) - 0.5) < 1e-9
    assert abs(float(near_far["far_advantage_rel"]) - 0.5) < 1e-9
