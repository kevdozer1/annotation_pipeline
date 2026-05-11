from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"
SNAPSHOT_DATE = "2026_05_11"
DETERMINISTIC_CREATED_AT_UTC = "2026-05-11T00:00:00Z"

LABELER_VERSIONS: dict[str, str] = {
    "captions": "moondream2-adapter@poc-local-caption-v0",
    "masks": "sam3-lewm-port@object_mask.npy-v0",
    "depth": "video-depth-anything-lewm-port@depth.npy-v0",
    "tracks": "cotracker3-lewm-port@tracks.npy-v0",
}


def stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def sha256_json(data: Any) -> str:
    return hashlib.sha256(stable_json(data).encode("utf-8")).hexdigest()


def derive_snapshot_id(source_records: list[dict[str, Any]]) -> tuple[str, str]:
    payload = {
        "source_records": source_records,
        "labeler_versions": LABELER_VERSIONS,
        "schema_version": SCHEMA_VERSION,
    }
    digest = sha256_json(payload)
    return f"snap_{SNAPSHOT_DATE}_{digest[:10]}", f"sha256:{digest}"


def write_manifest(
    snapshot_path: Path,
    snapshot_id: str,
    source_episode_count: int,
    transform_hash: str,
    parent_snapshot_id: str | None = None,
    labeler_runtime_seconds: dict[str, float] | None = None,
) -> dict[str, Any]:
    manifest_path = snapshot_path / "manifest.json"
    created_at = DETERMINISTIC_CREATED_AT_UTC
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            created_at = previous.get("created_at_utc", created_at)
        except json.JSONDecodeError:
            pass

    manifest = {
        "snapshot_id": snapshot_id,
        "created_at_utc": created_at,
        "source_dataset": "bridge_v2",
        "source_episode_count": int(source_episode_count),
        "labeler_versions": LABELER_VERSIONS,
        "schema_version": SCHEMA_VERSION,
        "parent_snapshot_id": parent_snapshot_id,
        "transform_hash": transform_hash,
    }
    if labeler_runtime_seconds is not None:
        manifest["labeler_runtime_seconds"] = {
            k: round(float(v), 6) for k, v in sorted(labeler_runtime_seconds.items())
        }

    snapshot_path.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest

