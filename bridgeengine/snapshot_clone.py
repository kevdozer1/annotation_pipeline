from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from bridgeengine.ingest.schema import LABEL_COLUMNS
from bridgeengine.paths import data_root as resolve_data_root


def clone_snapshot(
    source_snapshot: str,
    target_snapshot: str,
    data_root: str | Path | None = None,
    clear_labels: bool = True,
) -> dict[str, Any]:
    root = resolve_data_root(data_root)
    source_path = root / "snapshots" / source_snapshot
    target_path = root / "snapshots" / target_snapshot
    if not source_path.exists():
        raise FileNotFoundError(f"Source snapshot not found: {source_path}")
    if target_path.exists():
        raise FileExistsError(f"Target snapshot already exists: {target_path}")

    target_path.mkdir(parents=True)
    for name in ("episodes.parquet", "steps.parquet", "sensors.parquet", "manifest.json"):
        shutil.copy2(source_path / name, target_path / name)

    labels = pd.DataFrame(columns=LABEL_COLUMNS) if clear_labels else pd.read_parquet(source_path / "labels.parquet")
    for table_name in ("episodes", "steps", "sensors"):
        table_path = target_path / f"{table_name}.parquet"
        table = pd.read_parquet(table_path)
        if "snapshot_id" in table.columns:
            table["snapshot_id"] = target_snapshot
            table.to_parquet(table_path, index=False)
    if "snapshot_id" in labels.columns:
        labels["snapshot_id"] = target_snapshot
    labels.to_parquet(target_path / "labels.parquet", index=False)

    manifest_path = target_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["snapshot_id"] = target_snapshot
    manifest["parent_snapshot_id"] = source_snapshot
    manifest["cloned_from_snapshot_id"] = source_snapshot
    if clear_labels:
        manifest.pop("labeler_runtime_seconds", None)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "source_snapshot_id": source_snapshot,
        "target_snapshot_id": target_snapshot,
        "target_path": str(target_path.resolve()),
        "clear_labels": bool(clear_labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone a BridgeEngine snapshot under a new ID.")
    parser.add_argument("--source-snapshot", required=True)
    parser.add_argument("--target-snapshot", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--keep-labels", action="store_true")
    args = parser.parse_args()
    result = clone_snapshot(
        args.source_snapshot,
        args.target_snapshot,
        data_root=Path(args.data_root) if args.data_root else None,
        clear_labels=not args.keep_labels,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
