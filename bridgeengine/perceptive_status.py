from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from bridgeengine.paths import data_root as resolve_data_root


PERCEPTIVE_LABELERS = ("perceptive_masks", "perceptive_depth", "perceptive_tracks")


def perceptive_status(
    snapshot_id: str,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    root = resolve_data_root(data_root)
    snapshot_path = root / "snapshots" / snapshot_id
    episodes = pd.read_parquet(snapshot_path / "episodes.parquet")
    labels = pd.read_parquet(snapshot_path / "labels.parquet")
    episode_count = int(len(episodes))
    report = {
        "snapshot_id": snapshot_id,
        "episode_count": episode_count,
        "labelers": {},
        "ready_for_head_to_head": True,
        "readiness_rule": "Each perceptive labeler must have one row per episode, existing payloads, and no synthetic adapter provenance.",
    }
    for labeler in PERCEPTIVE_LABELERS:
        rows = labels[labels["labeler_name"] == labeler]
        adapters: Counter[str] = Counter()
        missing_payloads = 0
        for _, row in rows.iterrows():
            payload_path = Path(str(row.get("label_payload_path", "")))
            if not payload_path.exists():
                missing_payloads += 1
            provenance = _parse_json(row.get("provenance_json"))
            config = provenance.get("labeler_config", {}) if isinstance(provenance, dict) else {}
            adapters[str(config.get("adapter", "unknown"))] += 1
        synthetic_rows = sum(count for adapter, count in adapters.items() if adapter.startswith("synthetic"))
        ok = len(rows) >= episode_count and missing_payloads == 0 and synthetic_rows == 0
        report["labelers"][labeler] = {
            "row_count": int(len(rows)),
            "expected_rows": episode_count,
            "missing_payloads": int(missing_payloads),
            "adapter_counts": dict(sorted(adapters.items())),
            "synthetic_rows": int(synthetic_rows),
            "ready": bool(ok),
        }
        if not ok:
            report["ready_for_head_to_head"] = False
    return report


def _parse_json(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether perceptive mask/depth/track labels are real and head-to-head ready.")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--require-real", action="store_true", help="Exit nonzero if perceptive labels are missing or synthetic.")
    args = parser.parse_args()
    report = perceptive_status(args.snapshot, Path(args.data_root) if args.data_root else None)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_real and not report["ready_for_head_to_head"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
