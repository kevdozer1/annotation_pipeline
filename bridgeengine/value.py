from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bridgeengine.export.cut import export_cut
from bridgeengine.paths import data_root as resolve_data_root


VALUE_SCORE_VERSION = "bridgeengine-value-v1"
VALUE_COLUMNS = ["value_score", "value_percentile", "value_rank", "value_method", "value_score_version"]
PARQUET_TABLES = ("episodes", "steps", "sensors", "labels")


@dataclass(frozen=True)
class ValueReport:
    snapshot_id: str
    method: str
    episode_count: int
    score_min: float
    score_mean: float
    score_max: float
    top_outliers: list[dict[str, Any]]
    compression: dict[str, Any]
    runtime_seconds: float

    def to_text(self) -> str:
        lines = [
            f"Value report: {self.snapshot_id}",
            f"Method: {self.method}",
            f"Episodes: {self.episode_count}",
            f"Score distribution: min={self.score_min:.6f}, mean={self.score_mean:.6f}, max={self.score_max:.6f}",
            "Top outliers:",
        ]
        for row in self.top_outliers:
            lines.append(
                f"- rank {row['value_rank']}: {row['episode_id']} "
                f"score={row['value_score']:.6f} pct={row['value_percentile']:.3f} task={row.get('language_instruction', '')}"
            )
        if self.compression:
            lines.extend(
                [
                    "Compression:",
                    f"- high-value percentile: {self.compression['high_value_percentile']:.3f}",
                    f"- high-value episodes: {len(self.compression['high_value_episode_ids'])}",
                    f"- source parquet bytes: {self.compression['source_size_bytes']}",
                    f"- uniform zstd bytes: {self.compression['uniform_zstd_size_bytes']}",
                    f"- tiered bytes: {self.compression['tiered_size_bytes']}",
                    f"- tiered vs uniform savings: {self.compression['tiered_vs_uniform_savings_pct']:.2f}%",
                ]
            )
        lines.append(f"Runtime seconds: {self.runtime_seconds:.3f}")
        return "\n".join(lines)


def score_snapshot(
    snapshot_id: str,
    method: str = "auto",
    data_root: str | Path | None = None,
    top_n: int = 10,
    high_value_percentile: float = 0.90,
    write_compression: bool = True,
) -> ValueReport:
    start = time.perf_counter()
    root = resolve_data_root(data_root)
    snapshot_path = root / "snapshots" / snapshot_id
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")
    resolved_method = _resolve_method(method)
    if resolved_method == "prediction-error":
        scores = _prediction_error_scores(snapshot_id, root)
    elif resolved_method == "embedding-distance":
        scores = _embedding_distance_scores(snapshot_path)
    else:
        raise ValueError(f"Unknown value method: {method}")
    episodes = _write_value_columns(snapshot_path, scores, resolved_method)
    compression = (
        write_tiered_compression(snapshot_path, episodes, high_value_percentile, resolved_method)
        if write_compression
        else {}
    )
    values = episodes["value_score"].astype(float)
    report = ValueReport(
        snapshot_id=snapshot_id,
        method=resolved_method,
        episode_count=int(len(episodes)),
        score_min=float(values.min()) if len(values) else 0.0,
        score_mean=float(values.mean()) if len(values) else 0.0,
        score_max=float(values.max()) if len(values) else 0.0,
        top_outliers=_top_outliers(episodes, top_n),
        compression=compression,
        runtime_seconds=time.perf_counter() - start,
    )
    _write_report(snapshot_path, report)
    return report


def write_tiered_compression(
    snapshot_path: Path,
    episodes: pd.DataFrame,
    high_value_percentile: float = 0.90,
    method: str = "embedding-distance",
) -> dict[str, Any]:
    high_value_percentile = float(high_value_percentile)
    if not 0.0 <= high_value_percentile <= 1.0:
        raise ValueError("high_value_percentile must be between 0 and 1")
    threshold = float(episodes["value_score"].quantile(high_value_percentile)) if len(episodes) else 0.0
    high_ids = set(episodes.loc[episodes["value_score"] >= threshold, "episode_id"].astype(str).tolist())
    if not high_ids and len(episodes):
        high_ids.add(str(episodes.sort_values("value_score", ascending=False).iloc[0]["episode_id"]))

    out_dir = snapshot_path / "value_compression" / f"{method}_p{int(high_value_percentile * 100):02d}"
    tiered_dir = out_dir / "tiered"
    uniform_dir = out_dir / "uniform_zstd"
    _clear_directory(tiered_dir)
    _clear_directory(uniform_dir)
    (tiered_dir / "full_fidelity").mkdir(parents=True, exist_ok=True)
    (tiered_dir / "compressed").mkdir(parents=True, exist_ok=True)
    uniform_dir.mkdir(parents=True, exist_ok=True)

    source_size = 0
    for table in PARQUET_TABLES:
        path = snapshot_path / f"{table}.parquet"
        if not path.exists():
            continue
        source_size += path.stat().st_size
        df = pd.read_parquet(path)
        high_df, low_df = _split_by_episode(df, high_ids)
        _write_parquet(high_df, tiered_dir / "full_fidelity" / f"{table}.parquet", compression="snappy")
        _write_parquet(low_df, tiered_dir / "compressed" / f"{table}.parquet", compression="zstd", compression_level=12)
        _write_parquet(df, uniform_dir / f"{table}.parquet", compression="zstd", compression_level=12)

    tiered_size = _directory_size(tiered_dir)
    uniform_size = _directory_size(uniform_dir)
    savings_pct = ((uniform_size - tiered_size) / uniform_size * 100.0) if uniform_size else 0.0
    report = {
        "method": method,
        "high_value_percentile": high_value_percentile,
        "value_threshold": threshold,
        "high_value_episode_ids": sorted(high_ids),
        "source_size_bytes": int(source_size),
        "uniform_zstd_size_bytes": int(uniform_size),
        "tiered_size_bytes": int(tiered_size),
        "tiered_vs_uniform_savings_pct": float(savings_pct),
        "tiered_path": str(tiered_dir.resolve()),
        "uniform_zstd_path": str(uniform_dir.resolve()),
        "note": "Tiering is lossless Parquet compression: high-value episodes use snappy, the rest use zstd level 12.",
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _resolve_method(method: str) -> str:
    method = method.lower().strip()
    if method != "auto":
        return method
    try:
        from bridgeengine.benchmark.train_lewm import _import_torch, _prepare_stable_worldmodel_import

        _import_torch()
        _prepare_stable_worldmodel_import()
        return "prediction-error"
    except Exception:
        return "embedding-distance"


def _prediction_error_scores(snapshot_id: str, root: Path) -> dict[str, float]:
    from bridgeengine.benchmark.train_lewm import (
        DEFAULT_BATCH_SIZE,
        FEATURE_DIM,
        HISTORY_SIZE,
        BridgeWindowDataset,
        PromptConditioner,
        _chunks,
        _condition_features,
        _configure_trainable_parameters,
        _import_torch,
        _keep_frozen_modules_eval,
        _load_lewm,
        _seed_everything,
    )

    torch = _import_torch()
    seed = 0
    _seed_everything(torch, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with tempfile.TemporaryDirectory(prefix="bridgeengine_value_cut_") as tmp:
        cut_root = Path(tmp)
        export_cut(snapshot_id, "TRUE", cut_root, "value_all_episodes", data_root=root)
        cut_path = cut_root / "value_all_episodes"
        episode_ids = [
            line.strip()
            for line in (cut_path / "episode_list.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        dataset = BridgeWindowDataset(cut_path, episode_ids)
        if not dataset.records:
            return {episode_id: 0.0 for episode_id in episode_ids}
        model, latent_dim, _ = _load_lewm(torch, device)
        conditioner = PromptConditioner(FEATURE_DIM, latent_dim).to(device)
        _configure_trainable_parameters(model)
        params = [p for p in model.parameters() if p.requires_grad] + list(conditioner.parameters())
        optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
        epochs = int(__import__("os").environ.get("BRIDGEENGINE_VALUE_LEWM_EPOCHS", "3"))
        rng = random.Random(seed)
        for epoch in range(epochs):
            model.train()
            _keep_frozen_modules_eval(model)
            conditioner.train()
            indices = list(range(len(dataset.records)))
            rng.shuffle(indices)
            for batch_indices in _chunks(indices, DEFAULT_BATCH_SIZE):
                batch = dataset.batch(batch_indices, torch)
                losses = _per_window_losses(model, conditioner, batch, torch, device, _condition_features, HISTORY_SIZE)
                loss = losses.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

        grouped: dict[str, list[float]] = {episode_id: [] for episode_id in episode_ids}
        model.eval()
        conditioner.eval()
        with torch.no_grad():
            for batch_indices in _chunks(list(range(len(dataset.records))), DEFAULT_BATCH_SIZE):
                batch = dataset.batch(batch_indices, torch)
                losses = _per_window_losses(model, conditioner, batch, torch, device, _condition_features, HISTORY_SIZE)
                for record, loss in zip(batch["records"], losses.detach().cpu().tolist()):
                    grouped[record.episode_id].append(float(loss))
        return {episode_id: float(np.mean(values)) if values else 0.0 for episode_id, values in grouped.items()}


def _per_window_losses(model, conditioner, batch: dict[str, Any], torch, device, condition_features_fn, history_size: int):
    pixels = batch["pixels"].to(device, non_blocking=True)
    actions = batch["action"].to(device, non_blocking=True)
    encoded = model.encode({"pixels": pixels, "action": actions})
    emb = encoded["emb"]
    act_emb = encoded["act_emb"]
    features = condition_features_fn(batch["records"], "baseline", random.Random(0), False, torch).to(device)
    condition = conditioner(features, None, torch.zeros(len(batch["records"]), device=device))
    pred = model.predict(emb[:, :history_size] + condition[:, None, :], act_emb[:, :history_size])
    target = emb[:, 1 : history_size + 1].detach()
    return (pred - target).pow(2).mean(dim=(1, 2))


def _embedding_distance_scores(snapshot_path: Path) -> dict[str, float]:
    episodes = pd.read_parquet(snapshot_path / "episodes.parquet").sort_values("episode_id")
    vectors = []
    ids = []
    for row in episodes.to_dict("records"):
        ids.append(str(row["episode_id"]))
        vectors.append(_episode_feature_vector(row))
    if not vectors:
        return {}
    matrix = np.vstack(vectors).astype(np.float32)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    z = (matrix - mean) / std
    centroid = z.mean(axis=0, keepdims=True)
    centroid_dist = np.linalg.norm(z - centroid, axis=1)
    pairwise = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=2)
    k = min(5, max(1, len(ids) - 1))
    knn = np.partition(pairwise + np.eye(len(ids)) * 1e9, kth=k - 1, axis=1)[:, :k].mean(axis=1)
    scores = centroid_dist + 0.5 * knn
    return {episode_id: float(score) for episode_id, score in zip(ids, scores)}


def _episode_feature_vector(row: dict[str, Any]) -> np.ndarray:
    actions = _load_array(row.get("source_path_actions"))
    frames = _load_array(row.get("source_path_frames"))
    parts = [np.asarray([float(row.get("num_steps") or 0.0)], dtype=np.float32)]
    if actions is not None and actions.size:
        arr = np.asarray(actions, dtype=np.float32)
        parts.extend([arr.mean(axis=0), arr.std(axis=0), arr.min(axis=0), arr.max(axis=0)])
    else:
        parts.append(np.zeros(28, dtype=np.float32))
    if frames is not None and frames.size:
        fr = np.asarray(frames, dtype=np.float32)
        sample = fr[np.linspace(0, len(fr) - 1, min(8, len(fr))).astype(int)]
        parts.extend([sample.mean(axis=(0, 1, 2)) / 255.0, sample.std(axis=(0, 1, 2)) / 255.0])
    else:
        parts.append(np.zeros(6, dtype=np.float32))
    return np.concatenate([p.ravel() for p in parts]).astype(np.float32)


def _write_value_columns(snapshot_path: Path, scores: dict[str, float], method: str) -> pd.DataFrame:
    episodes_path = snapshot_path / "episodes.parquet"
    episodes = _ensure_value_columns(pd.read_parquet(episodes_path))
    values = episodes["episode_id"].astype(str).map(scores).fillna(0.0).astype(float)
    rank_desc = values.rank(method="first", ascending=False).astype(int)
    percentile = values.rank(method="average", pct=True).astype(float)
    episodes["value_score"] = values
    episodes["value_percentile"] = percentile
    episodes["value_rank"] = rank_desc
    episodes["value_method"] = method
    episodes["value_score_version"] = VALUE_SCORE_VERSION
    episodes = episodes.sort_values("episode_id")
    episodes.to_parquet(episodes_path, index=False)
    return episodes


def _top_outliers(episodes: pd.DataFrame, top_n: int) -> list[dict[str, Any]]:
    cols = ["episode_id", "language_instruction", "num_steps", "value_score", "value_percentile", "value_rank"]
    existing = [col for col in cols if col in episodes.columns]
    rows = episodes.sort_values("value_score", ascending=False).head(top_n)[existing].to_dict("records")
    for row in rows:
        row["value_score"] = float(row.get("value_score") or 0.0)
        row["value_percentile"] = float(row.get("value_percentile") or 0.0)
        row["value_rank"] = int(row.get("value_rank") or 0)
    return rows


def _write_report(snapshot_path: Path, report: ValueReport) -> None:
    path = snapshot_path / "value_report.json"
    payload = {
        "snapshot_id": report.snapshot_id,
        "method": report.method,
        "episode_count": report.episode_count,
        "score_min": report.score_min,
        "score_mean": report.score_mean,
        "score_max": report.score_max,
        "top_outliers": report.top_outliers,
        "compression": report.compression,
        "runtime_seconds": report.runtime_seconds,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ensure_value_columns(episodes: pd.DataFrame) -> pd.DataFrame:
    episodes = episodes.copy()
    for column in VALUE_COLUMNS:
        if column not in episodes.columns:
            episodes[column] = None
    return episodes


def _split_by_episode(df: pd.DataFrame, high_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "episode_id" not in df.columns:
        return df.copy(), df.iloc[0:0].copy()
    mask = df["episode_id"].astype(str).isin(high_ids)
    return df.loc[mask].copy(), df.loc[~mask].copy()


def _write_parquet(df: pd.DataFrame, path: Path, compression: str, compression_level: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {"compression": compression}
    if compression_level is not None:
        kwargs["compression_level"] = compression_level
    try:
        df.to_parquet(path, index=False, **kwargs)
    except TypeError:
        kwargs.pop("compression_level", None)
        df.to_parquet(path, index=False, **kwargs)


def _clear_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            _clear_directory(child)
            child.rmdir()
        else:
            child.unlink()


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return int(sum(file.stat().st_size for file in path.rglob("*") if file.is_file()))


def _load_array(path_value: Any) -> np.ndarray | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.exists():
        return None
    return np.load(path, allow_pickle=False, mmap_mode="r")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score BridgeEngine episode value and report outliers/compression.")
    sub = parser.add_subparsers(dest="command", required=True)
    report = sub.add_parser("report", help="Score a snapshot and print value/curation report.")
    report.add_argument("--snapshot", required=True)
    report.add_argument("--data-root", default=None)
    report.add_argument("--method", default="auto", choices=["auto", "prediction-error", "embedding-distance"])
    report.add_argument("--top-n", type=int, default=10)
    report.add_argument("--high-value-percentile", type=float, default=0.90)
    report.add_argument("--no-compression", action="store_true")
    args = parser.parse_args()
    if args.command == "report":
        value_report = score_snapshot(
            snapshot_id=args.snapshot,
            method=args.method,
            data_root=Path(args.data_root) if args.data_root else None,
            top_n=args.top_n,
            high_value_percentile=args.high_value_percentile,
            write_compression=not args.no_compression,
        )
        print(value_report.to_text())


if __name__ == "__main__":
    main()
