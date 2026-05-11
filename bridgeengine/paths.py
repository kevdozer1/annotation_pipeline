from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_root(default: str | Path | None = None) -> Path:
    raw = os.environ.get("BRIDGEENGINE_DATA_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    if default is not None:
        return Path(default).expanduser().resolve()
    return project_root() / "bridgeengine_data"


def snapshot_dir(snapshot_id: str, root: str | Path | None = None) -> Path:
    return data_root(root) / "snapshots" / snapshot_id


def latest_snapshot_id(root: str | Path | None = None) -> str:
    snap_root = data_root(root) / "snapshots"
    if not snap_root.exists():
        raise FileNotFoundError(f"No snapshot directory found under {snap_root}")
    manifests = sorted(snap_root.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No snapshot manifests found under {snap_root}")
    return manifests[-1].parent.name

