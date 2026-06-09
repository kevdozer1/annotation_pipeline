from __future__ import annotations

import pandas as pd

from bridgeengine.benchmark.leak_power import _paired_deltas, _paired_summary
from bridgeengine.benchmark.train_lewm import WindowRecord
from bridgeengine.benchmark.window_eval import window_rows


def test_window_rows_include_segment_distance() -> None:
    rows = window_rows(
        [
            WindowRecord(
                episode_id="episode_a",
                start_idx=4,
                task="task",
                subtask_text="place object",
                segment_idx=2,
                metadata={},
                subgoal_image_path="subgoal.jpg",
                segment_start_step=3,
                segment_end_step=9,
            )
        ],
        [0.25],
        subgoal_mask=[1.0],
        history_size=3,
    )

    assert rows == [
        {
            "episode_id": "episode_a",
            "start_idx": 4,
            "segment_idx": 2,
            "subgoal_mask": 1.0,
            "dist_to_segment_end": 3,
            "sq_err": 0.25,
        }
    ]


def test_paired_summary_uses_episode_seed_bootstrap_units() -> None:
    rows = []
    for seed in (1, 2):
        for episode in ("episode_a", "episode_b"):
            for start_idx in (0, 1):
                rows.append(
                    {
                        "scale_n": 25,
                        "seed": seed,
                        "condition": "P0_pi07_baseline",
                        "episode_id": episode,
                        "start_idx": start_idx,
                        "segment_idx": 0,
                        "subgoal_mask": 1.0,
                        "dist_to_segment_end": 5,
                        "sq_err": 1.0,
                    }
                )
                rows.append(
                    {
                        "scale_n": 25,
                        "seed": seed,
                        "condition": "P4_pi07_full_stack",
                        "episode_id": episode,
                        "start_idx": start_idx,
                        "segment_idx": 0,
                        "subgoal_mask": 1.0,
                        "dist_to_segment_end": 5,
                        "sq_err": 0.75,
                    }
                )

    deltas = _paired_deltas(pd.DataFrame(rows), reference="P0_pi07_baseline")
    summary = _paired_summary(deltas, bootstrap_reps=20, seed=0)
    row = summary.iloc[0]

    assert row["condition"] == "P4_pi07_full_stack"
    assert row["paired_windows"] == 8
    assert row["bootstrap_units"] == 4
    assert row["mean_delta_sq_err"] == -0.25
