from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from bridgeengine.ingest.snapshot import write_manifest
from bridgeengine.labelers import CaptionLabeler, DepthLabeler, MaskLabeler, TrackLabeler
from bridgeengine.paths import data_root as resolve_data_root


def run_labelers(
    snapshot_id: str,
    data_root: str | Path | None = None,
    labeler_names: list[str] | None = None,
) -> dict[str, Any]:
    root = resolve_data_root(data_root)
    snapshot_path = root / "snapshots" / snapshot_id
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    episodes = pd.read_parquet(snapshot_path / "episodes.parquet")
    labelers = _build_labelers(root)
    if labeler_names:
        wanted = set(labeler_names)
        labelers = [l for l in labelers if l.name in wanted]

    rows: list[dict[str, Any]] = []
    runtime_seconds: dict[str, float] = defaultdict(float)
    for _, episode in episodes.sort_values("episode_id").iterrows():
        episode_path = Path(episode["source_path_meta"]).parent
        for labeler in labelers:
            result = labeler.label_episode(episode_path, snapshot_id)
            runtime_seconds[result.labeler_name] += float(result.provenance.get("wall_clock_seconds", 0.0))
            rows.append(
                {
                    "episode_id": result.episode_id,
                    "step_idx": None,
                    "labeler_name": result.labeler_name,
                    "labeler_version": result.labeler_version,
                    "label_payload_path": str(result.payload_path.resolve()),
                    "confidence": result.confidence,
                    "provenance_json": json.dumps(result.provenance, sort_keys=True),
                    "snapshot_id": snapshot_id,
                }
            )

    labels = pd.DataFrame(rows).sort_values(["episode_id", "labeler_name"]).reset_index(drop=True)
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


def _build_labelers(root: Path):
    return [
        CaptionLabeler(root),
        MaskLabeler(root),
        DepthLabeler(root),
        TrackLabeler(root),
    ]

