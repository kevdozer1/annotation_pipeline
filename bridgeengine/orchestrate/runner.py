from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from bridgeengine.ingest.snapshot import write_manifest
from bridgeengine.ingest.schema import LABEL_COLUMNS
from bridgeengine.labelers import EpisodeMetadataLabeler, SubgoalImageLabeler, SubtaskSegmenter
from bridgeengine.labelers.base import LabelResult
from bridgeengine.labelers.perceptive import DepthLabeler, MaskLabeler, TrackLabeler
from bridgeengine.paths import data_root as resolve_data_root


def run_labelers(
    snapshot_id: str,
    data_root: str | Path | None = None,
    labeler_names: list[str] | None = None,
    allow_fallback: bool = False,
    vlm_backend: str | None = None,
    vlm_model: str | None = None,
) -> dict[str, Any]:
    root = resolve_data_root(data_root)
    snapshot_path = root / "snapshots" / snapshot_id
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    episodes = pd.read_parquet(snapshot_path / "episodes.parquet")
    labelers = _build_labelers(root, allow_fallback=allow_fallback, vlm_backend=vlm_backend, vlm_model=vlm_model)
    if labeler_names:
        wanted = set(labeler_names)
        labelers = [l for l in labelers if l.name in wanted]

    rows: list[dict[str, Any]] = []
    runtime_seconds: dict[str, float] = defaultdict(float)
    for _, episode in episodes.sort_values("episode_id").iterrows():
        episode_path = Path(episode["source_path_meta"]).parent
        for labeler in labelers:
            results = _as_results(labeler.label_episode(episode_path, snapshot_id))
            for result in results:
                runtime_seconds[result.labeler_name] += float(result.provenance.get("wall_clock_seconds", 0.0))
                rows.append(_row_from_result(result, snapshot_id))

    labels = pd.DataFrame(rows, columns=LABEL_COLUMNS).sort_values(["episode_id", "labeler_name", "segment_idx"]).reset_index(drop=True)
    labels.to_parquet(snapshot_path / "labels.parquet", index=False)

    manifest = json.loads((snapshot_path / "manifest.json").read_text(encoding="utf-8"))
    write_manifest(
        snapshot_path=snapshot_path,
        snapshot_id=snapshot_id,
        source_episode_count=int(manifest["source_episode_count"]),
        transform_hash=manifest["transform_hash"],
        parent_snapshot_id=manifest.get("parent_snapshot_id"),
        labeler_runtime_seconds=dict(runtime_seconds),
    )
    return {
        "snapshot_id": snapshot_id,
        "label_rows": len(labels),
        "labelers": sorted(labels["labeler_name"].unique().tolist()),
        "runtime_seconds": {k: round(v, 6) for k, v in sorted(runtime_seconds.items())},
    }


def _row_from_result(result: LabelResult, snapshot_id: str) -> dict[str, Any]:
    return {
        "episode_id": result.episode_id,
        "step_idx": None,
        "segment_idx": result.segment_idx,
        "labeler_name": result.labeler_name,
        "labeler_version": result.labeler_version,
        "label_payload_path": str(result.payload_path.resolve()),
        "metadata_payload_json": result.metadata_payload_json,
        "subgoal_image_path": str(result.subgoal_image_path.resolve()) if result.subgoal_image_path else None,
        "confidence": result.confidence,
        "provenance_json": json.dumps(result.provenance, sort_keys=True),
        "snapshot_id": snapshot_id,
    }


def _as_results(value) -> list[LabelResult]:
    if isinstance(value, list):
        return value
    return [value]


def run_perceptive_labelers(
    snapshot_id: str,
    data_root: str | Path | None = None,
    labeler_names: list[str] | None = None,
) -> dict[str, Any]:
    root = resolve_data_root(data_root)
    snapshot_path = root / "snapshots" / snapshot_id
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    episodes = pd.read_parquet(snapshot_path / "episodes.parquet")
    labelers = _build_perceptive_labelers(root)
    if labeler_names:
        wanted = set(labeler_names)
        labelers = [l for l in labelers if l.name in wanted]

    existing = pd.read_parquet(snapshot_path / "labels.parquet")
    rows: list[dict[str, Any]] = existing.to_dict("records")
    runtime_seconds: dict[str, float] = defaultdict(float)
    for _, episode in episodes.sort_values("episode_id").iterrows():
        episode_path = Path(episode["source_path_meta"]).parent
        for labeler in labelers:
            for result in _as_results(labeler.label_episode(episode_path, snapshot_id)):
                runtime_seconds[result.labeler_name] += float(result.provenance.get("wall_clock_seconds", 0.0))
                rows.append(_row_from_result(result, snapshot_id))

    labels = pd.DataFrame(rows, columns=LABEL_COLUMNS).sort_values(["episode_id", "labeler_name", "segment_idx"]).reset_index(drop=True)
    labels.to_parquet(snapshot_path / "labels.parquet", index=False)

    manifest = json.loads((snapshot_path / "manifest.json").read_text(encoding="utf-8"))
    write_manifest(
        snapshot_path=snapshot_path,
        snapshot_id=snapshot_id,
        source_episode_count=int(manifest["source_episode_count"]),
        transform_hash=manifest["transform_hash"],
        parent_snapshot_id=manifest.get("parent_snapshot_id"),
        labeler_runtime_seconds=dict(runtime_seconds),
    )
    return {
        "snapshot_id": snapshot_id,
        "label_rows": len(labels),
        "labelers": sorted(labels["labeler_name"].unique().tolist()),
        "runtime_seconds": {k: round(v, 6) for k, v in sorted(runtime_seconds.items())},
    }


def _build_labelers(root: Path, allow_fallback: bool = False, vlm_backend: str | None = None, vlm_model: str | None = None):
    return [
        SubtaskSegmenter(root, allow_fallback=allow_fallback, backend_name=vlm_backend, backend_model=vlm_model),
        EpisodeMetadataLabeler(root, allow_fallback=allow_fallback, backend_name=vlm_backend, backend_model=vlm_model),
        SubgoalImageLabeler(root),
    ]


def _build_perceptive_labelers(root: Path):
    return [
        MaskLabeler(root),
        DepthLabeler(root),
        TrackLabeler(root),
    ]
