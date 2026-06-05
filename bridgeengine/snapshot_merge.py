from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from bridgeengine.ingest.schema import LABEL_COLUMNS
from bridgeengine.ingest.snapshot import write_manifest
from bridgeengine.paths import data_root as resolve_data_root


def merge_labeled_snapshots(
    target_snapshot: str,
    source_snapshots: list[str],
    data_root: str | Path | None = None,
    overwrite_labels: bool = False,
    copy_artifacts: bool = True,
) -> dict[str, Any]:
    root = resolve_data_root(data_root)
    target_path = root / "snapshots" / target_snapshot
    if not target_path.exists():
        raise FileNotFoundError(f"Target snapshot not found: {target_path}")
    if not source_snapshots:
        raise ValueError("At least one source snapshot is required")

    target_episodes = pd.read_parquet(target_path / "episodes.parquet")
    target_ids = set(target_episodes["episode_id"].astype(str).tolist())
    existing_labels = pd.read_parquet(target_path / "labels.parquet")
    if len(existing_labels) and not overwrite_labels:
        raise ValueError(f"Target snapshot already has {len(existing_labels)} label rows. Pass --overwrite-labels to replace them.")

    rows: list[dict[str, Any]] = []
    seen_labeled_episodes: set[str] = set()
    runtime_seconds: dict[str, float] = defaultdict(float)
    for source_snapshot in source_snapshots:
        source_path = root / "snapshots" / source_snapshot
        if not source_path.exists():
            raise FileNotFoundError(f"Source snapshot not found: {source_path}")
        labels = pd.read_parquet(source_path / "labels.parquet")
        source_manifest = _read_json(source_path / "manifest.json")
        for name, seconds in source_manifest.get("labeler_runtime_seconds", {}).items():
            runtime_seconds[str(name)] += float(seconds)
        for row in labels.to_dict("records"):
            episode_id = str(row.get("episode_id"))
            if episode_id not in target_ids:
                continue
            if row.get("labeler_name") == "episode_metadata":
                if episode_id in seen_labeled_episodes:
                    raise ValueError(f"Duplicate metadata label for {episode_id} across source snapshots")
                seen_labeled_episodes.add(episode_id)
            merged = dict(row)
            merged["snapshot_id"] = target_snapshot
            if copy_artifacts:
                merged["label_payload_path"] = _copy_artifact_path(merged.get("label_payload_path"), source_snapshot, target_snapshot)
                merged["subgoal_image_path"] = _copy_artifact_path(merged.get("subgoal_image_path"), source_snapshot, target_snapshot)
            rows.append(merged)
        if copy_artifacts:
            _copy_raw_outputs(source_path, target_path)

    labels_out = pd.DataFrame(rows, columns=LABEL_COLUMNS).sort_values(["episode_id", "labeler_name", "segment_idx"]).reset_index(drop=True)
    labels_out.to_parquet(target_path / "labels.parquet", index=False)

    manifest = _read_json(target_path / "manifest.json")
    manifest["merged_label_source_snapshots"] = source_snapshots
    write_manifest(
        snapshot_path=target_path,
        snapshot_id=target_snapshot,
        source_episode_count=int(manifest["source_episode_count"]),
        transform_hash=manifest["transform_hash"],
        parent_snapshot_id=manifest.get("parent_snapshot_id"),
        labeler_runtime_seconds=dict(runtime_seconds),
    )
    manifest = _read_json(target_path / "manifest.json")
    manifest["merged_label_source_snapshots"] = source_snapshots
    (target_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "target_snapshot_id": target_snapshot,
        "source_snapshot_ids": source_snapshots,
        "target_episode_count": int(len(target_episodes)),
        "merged_label_rows": int(len(labels_out)),
        "merged_metadata_episode_count": int(len(seen_labeled_episodes)),
        "missing_metadata_episode_count": int(len(target_ids - seen_labeled_episodes)),
        "missing_metadata_episode_ids": sorted(target_ids - seen_labeled_episodes),
    }


def _copy_artifact_path(path_value: Any, source_snapshot: str, target_snapshot: str) -> str | None:
    if not path_value:
        return None
    source_path = Path(str(path_value))
    if not source_path.exists():
        return str(source_path)
    target_path = Path(str(source_path).replace(source_snapshot, target_snapshot))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not target_path.exists():
        shutil.copy2(source_path, target_path)
    return str(target_path.resolve())


def _copy_raw_outputs(source_path: Path, target_path: Path) -> None:
    source_raw = source_path / "raw_vlm_outputs"
    if not source_raw.exists():
        return
    target_raw = target_path / "raw_vlm_outputs"
    target_raw.mkdir(parents=True, exist_ok=True)
    for episode_dir in source_raw.iterdir():
        if not episode_dir.is_dir():
            continue
        dst = target_raw / episode_dir.name
        if dst.exists():
            continue
        shutil.copytree(episode_dir, dst)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge labels from source snapshots into an existing target snapshot.")
    parser.add_argument("--target-snapshot", required=True)
    parser.add_argument("--source-snapshot", action="append", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--overwrite-labels", action="store_true")
    parser.add_argument("--no-copy-artifacts", action="store_true")
    args = parser.parse_args()
    result = merge_labeled_snapshots(
        target_snapshot=args.target_snapshot,
        source_snapshots=args.source_snapshot,
        data_root=Path(args.data_root) if args.data_root else None,
        overwrite_labels=args.overwrite_labels,
        copy_artifacts=not args.no_copy_artifacts,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
