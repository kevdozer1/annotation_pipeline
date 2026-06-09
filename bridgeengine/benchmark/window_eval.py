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
    norm_sq_err: Iterable[float] | None = None,
) -> list[dict[str, object]]:
    errors = [float(x) for x in sq_err]
    masks = [float(x) for x in subgoal_mask] if subgoal_mask is not None else None
    norms = [float(x) for x in norm_sq_err] if norm_sq_err is not None else None
    if len(records) != len(errors):
        raise ValueError(f"records/errors length mismatch: {len(records)} records, {len(errors)} errors")
    if masks is not None and len(masks) != len(records):
        raise ValueError(f"records/subgoal_mask length mismatch: {len(records)} records, {len(masks)} masks")
    if norms is not None and len(norms) != len(records):
        raise ValueError(f"records/norm_sq_err length mismatch: {len(records)} records, {len(norms)} values")
    rows: list[dict[str, object]] = []
    for idx, (record, err) in enumerate(zip(records, errors)):
        mask = masks[idx] if masks is not None else (1.0 if record.subgoal_image_path else 0.0)
        row: dict[str, object] = {
            "episode_id": str(record.episode_id),
            "start_idx": int(record.start_idx),
            "segment_idx": "" if record.segment_idx is None else int(record.segment_idx),
            "subgoal_mask": float(mask),
            "dist_to_segment_end": _distance_to_segment_end(record, history_size),
            "sq_err": float(err),
        }
        # norm_sq_err is additive: only emitted when target-variance normalization
        # was computed, so legacy callers and the exact-match unit test are unaffected.
        if norms is not None:
            row["norm_sq_err"] = norms[idx]
        rows.append(row)
    return rows


def write_fixed_eval_windows(
    run_dir: str | Path,
    records: list[WindowRecord],
    sq_err: Iterable[float],
    *,
    subgoal_mask: Iterable[float] | None = None,
    history_size: int = HISTORY_SIZE,
    norm_sq_err: Iterable[float] | None = None,
    filename: str = "fixed_eval_windows.csv",
) -> Path:
    rows = window_rows(
        records, sq_err, subgoal_mask=subgoal_mask, history_size=history_size, norm_sq_err=norm_sq_err
    )
    columns = list(WINDOW_COLUMNS)
    if norm_sq_err is not None:
        columns = columns + ["norm_sq_err"]
    path = Path(run_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


class TargetLatentMoments:
    """Streaming accumulator for held-out target-latent statistics.

    The eval-validity concern is that every condition finetunes the full model,
    so each run's target latents ``tgt_emb`` live in their own geometry and a
    condition could lower raw MSE simply by contracting latent variance. To make
    per-window errors comparable across runs we normalize each window's squared
    error by that run's own mean per-dimension target-latent variance.

    Feed every held-out ``tgt_emb`` batch of shape ``(B, T_targets, D)``; the
    accumulator tracks per-dimension first/second moments over all target latent
    vectors (``B * T_targets`` samples), then ``finalize`` returns the scalar
    mean per-dim variance and mean squared L2 norm.
    """

    def __init__(self) -> None:
        self._sum: np.ndarray | None = None
        self._sumsq: np.ndarray | None = None
        self._count: int = 0

    def update_from_numpy(self, flat_targets: np.ndarray) -> None:
        # flat_targets: (n_vectors, D)
        flat = np.asarray(flat_targets, dtype=np.float64)
        if flat.ndim != 2:
            raise ValueError(f"expected 2D (n, D) target array, got shape {flat.shape}")
        col_sum = flat.sum(axis=0)
        col_sumsq = (flat * flat).sum(axis=0)
        if self._sum is None:
            self._sum = col_sum
            self._sumsq = col_sumsq
        else:
            self._sum += col_sum
            self._sumsq += col_sumsq
        self._count += int(flat.shape[0])

    def finalize(self) -> dict[str, float]:
        if self._sum is None or self._count <= 0:
            return {"heldout_target_variance": float("nan"), "heldout_target_mean_sq_norm": float("nan")}
        n = float(self._count)
        mean = self._sum / n
        var_per_dim = self._sumsq / n - mean * mean
        # Numerical guard: tiny negative variances from float roundoff -> 0.
        var_per_dim = np.clip(var_per_dim, 0.0, None)
        return {
            "heldout_target_variance": float(np.mean(var_per_dim)),
            "heldout_target_mean_sq_norm": float(self._sumsq.sum() / n),
        }


def normalized_window_errors(sq_err: Iterable[float], target_variance: float) -> list[float]:
    """Divide per-window squared error by the run's mean target-latent variance."""
    var = float(target_variance)
    if not np.isfinite(var) or var <= 0.0:
        return [float("nan") for _ in sq_err]
    return [float(x) / var for x in sq_err]


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
