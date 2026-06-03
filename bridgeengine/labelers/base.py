from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class LabelResult:
    labeler_name: str
    labeler_version: str
    episode_id: str
    payload_path: Path
    confidence: float | None
    provenance: dict
    segment_idx: int | None = None
    metadata_payload_json: str | None = None
    subgoal_image_path: Path | None = None


class Labeler(Protocol):
    name: str
    version: str

    def label_episode(self, episode_path: Path, snapshot_id: str) -> LabelResult | list[LabelResult]:
        ...


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_metadata(episode_path: Path) -> dict:
    for name in ("metadata.json", "meta.json"):
        path = episode_path / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def episode_id_from_path(path: Path) -> str:
    return path.name
