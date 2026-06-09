from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from bridgeengine.benchmark.train_lewm import HISTORY_SIZE, WindowRecord


WINDOW_COLUMNS = [
    "episode_id",
    "start_idx",
    "segment_idx",
    "subgoal_mask",
    "dist_to_segment_end",
    "sq_err",
]


def window_rows(
    records: list[WindowRecord],
    sq_err: Iterable[float],
    *,
    subgoal_mask: Iterable[float] | None = None,
    history_size: int = HISTORY_SIZE,
) -> list[dict[str, object]]:
    errors = [float(x) for x in sq_err]
    masks = [float(x) for x in subgoal_mask] if subgoal_mask is not None else None
    if len(records) != len(errors):
        raise ValueError(f"records/errors length mismatch: {len(records)} records, {len(errors)} errors")
    if masks is not None and len(masks) != len(records):
        raise ValueError(f"records/subgoal_mask length mismatch: {len(records)} records, {len(masks)} masks")
    rows: list[dict[str, object]] = []
    for idx, (record, err) in enumerate(zip(records, errors)):
        mask = masks[idx] if masks is not None else (1.0 if record.subgoal_image_path else 0.0)
        rows.append(
            {
                "episode_id": str(record.episode_id),
                "start_idx": int(record.start_idx),
                "segment_idx": "" if record.segment_idx is None else int(record.segment_idx),
                "subgoal_mask": float(mask),
                "dist_to_segment_end": _distance_to_segment_end(record, history_size),
                "sq_err": float(err),
            }
        )
    return rows


def write_fixed_eval_windows(
    run_dir: str | Path,
    records: list[WindowRecord],
    sq_err: Iterable[float],
    *,
    subgoal_mask: Iterable[float] | None = None,
    history_size: int = HISTORY_SIZE,
) -> Path:
    rows = window_rows(records, sq_err, subgoal_mask=subgoal_mask, history_size=history_size)
    path = Path(run_dir) / "fixed_eval_windows.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=WINDOW_COLUMNS).to_csv(path, index=False)
    return path


def mean_or_nan(values: Iterable[float]) -> float:
    arr = np.asarray([float(x) for x in values], dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def weighted_mean_or_nan(values: Iterable[float], weights: Iterable[int]) -> float:
    vals = np.asarray([float(x) for x in values], dtype=np.float64)
    wts = np.asarray([int(x) for x in weights], dtype=np.float64)
    if vals.size == 0 or wts.size == 0 or float(wts.sum()) <= 0.0:
        return float("nan")
    if vals.shape[0] != wts.shape[0]:
        raise ValueError("values and weights must have the same length")
    return float(np.average(vals, weights=wts))


def _distance_to_segment_end(record: WindowRecord, history_size: int) -> int | str:
    if record.segment_end_step is None:
        return ""
    active_step = int(record.start_idx) + int(history_size) - 1
    return int(record.segment_end_step) - active_step
