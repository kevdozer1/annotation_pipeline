from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from bridgeengine.calibration import default_gold_path, review_summary
from bridgeengine.goldset import reliability_report
from bridgeengine.paths import data_root as resolve_data_root


HUMAN_CALIBRATION_VERSION = "human_gold_curation_v1"


def apply_gold_scores_to_snapshot(
    source_snapshot: str,
    target_snapshot: str,
    gold_file: str | Path | None = None,
    data_root: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = resolve_data_root(data_root)
    source_path = root / "snapshots" / source_snapshot
    target_path = root / "snapshots" / target_snapshot
    if not source_path.exists():
        raise FileNotFoundError(f"Source snapshot not found: {source_path}")
    if target_path.exists():
        if not overwrite:
            raise FileExistsError(f"Target snapshot already exists: {target_path}")
        shutil.rmtree(target_path)
    shutil.copytree(source_path, target_path)

    gold_path = Path(gold_file) if gold_file else default_gold_path(source_snapshot, root)
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")
    gold = _read_json(gold_path)
    gold_scores = _gold_scores(gold)
    if not gold_scores:
        raise ValueError(f"Gold file has no reviewed curation scores: {gold_path}")

    labels_path = target_path / "labels.parquet"
    labels = pd.read_parquet(labels_path)
    rows = labels.to_dict("records")
    changed = []
    for row in rows:
        row["snapshot_id"] = target_snapshot
        if row.get("label_payload_path"):
            row["label_payload_path"] = _copy_or_rewrite_path(row["label_payload_path"], source_snapshot, target_snapshot)
        if row.get("subgoal_image_path"):
            row["subgoal_image_path"] = _rewrite_existing_snapshot_path(row["subgoal_image_path"], source_snapshot, target_snapshot)
        if row.get("labeler_name") != "episode_metadata":
            continue
        episode_id = str(row.get("episode_id"))
        if episode_id not in gold_scores:
            continue
        metadata = _parse_json(row.get("metadata_payload_json"))
        payload_path = Path(str(row.get("label_payload_path") or ""))
        payload = _read_json(payload_path) if payload_path.exists() else {}
        if not metadata:
            metadata = dict(payload.get("metadata", {}))
        before = _safe_int(metadata.get("curation_quality")) or _safe_int(metadata.get("quality"))
        gold_meta = gold_scores[episode_id]
        quality = int(gold_meta["quality"])
        updated = dict(metadata)
        updated.setdefault("task_success_quality", _safe_int(metadata.get("task_success_quality")) or _safe_int(metadata.get("quality")))
        updated["quality"] = quality
        updated["curation_quality"] = quality
        updated["curation_keep"] = quality >= 4
        updated["mistake"] = bool(gold_meta.get("mistake", metadata.get("mistake", False)))
        updated["human_calibrated"] = True
        updated["human_calibration_version"] = HUMAN_CALIBRATION_VERSION
        updated["human_calibration_gold_file"] = str(gold_path.resolve())
        updated["human_calibration_accept_auto_metadata"] = bool(gold_meta.get("accept_auto", False))
        updated["scoring_basis"] = "human_calibrated_visible_boundary_training_usefulness"
        updated["scoring_version"] = HUMAN_CALIBRATION_VERSION
        updated["scoring_reason"] = f"human calibrated curation score {quality}/5"
        if gold_meta.get("reason"):
            updated["human_calibration_reason"] = str(gold_meta.get("reason"))
        row["metadata_payload_json"] = json.dumps(updated, sort_keys=True)
        if payload_path.exists():
            payload["metadata"] = updated
            _write_json(payload_path, payload)
        changed.append(
            {
                "episode_id": episode_id,
                "auto_quality": before,
                "human_quality": quality,
                "changed": before != quality,
            }
        )

    labels_out = pd.DataFrame(rows, columns=labels.columns)
    labels_out.to_parquet(labels_path, index=False)
    for table_name in ("episodes", "steps", "sensors"):
        table_path = target_path / f"{table_name}.parquet"
        table = pd.read_parquet(table_path)
        if "snapshot_id" in table.columns:
            table["snapshot_id"] = target_snapshot
            table.to_parquet(table_path, index=False)

    manifest_path = target_path / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["snapshot_id"] = target_snapshot
    manifest["parent_snapshot_id"] = source_snapshot
    manifest["human_calibrated_from_snapshot_id"] = source_snapshot
    manifest["human_calibration_version"] = HUMAN_CALIBRATION_VERSION
    manifest["human_calibration_gold_file"] = str(gold_path.resolve())
    _write_json(manifest_path, manifest)

    target_gold_path = target_path / "gold" / Path(gold_path).name
    target_gold_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(gold_path, target_gold_path)
    reliability = reliability_report(target_snapshot, target_gold_path, data_root=root)
    summary = review_summary(source_snapshot, gold_path, data_root=root)
    report = {
        "source_snapshot_id": source_snapshot,
        "target_snapshot_id": target_snapshot,
        "human_calibration_version": HUMAN_CALIBRATION_VERSION,
        "gold_file": str(gold_path.resolve()),
        "reviewed_episode_count": int(summary["reviewed"].sum()) if not summary.empty else 0,
        "episode_count": int(len(summary)),
        "auto_quality_counts": _counts(summary.get("auto_score")),
        "human_quality_counts": _counts(summary.get("gold_score")),
        "changed_score_count": int(sum(bool(item["changed"]) for item in changed)),
        "changed_scores": changed,
        "reliability": reliability,
    }
    _write_json(target_path / "human_calibration_report.json", report)
    return report


def _gold_scores(gold: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for entry in gold.get("episodes", []):
        episode_id = str(entry.get("episode_id"))
        metadata = entry.get("gold", {}).get("metadata", {}) or {}
        quality = _safe_int(metadata.get("curation_quality"))
        if quality is None:
            quality = _safe_int(metadata.get("quality"))
        if quality is None:
            continue
        result[episode_id] = {**metadata, "quality": int(max(1, min(5, quality)))}
    return result


def _copy_or_rewrite_path(path_value: Any, source_snapshot: str, target_snapshot: str) -> str:
    source = Path(str(path_value))
    target = Path(str(path_value).replace(source_snapshot, target_snapshot))
    if target == source:
        return str(source)
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        return str(target.resolve())
    return str(target)


def _rewrite_existing_snapshot_path(path_value: Any, source_snapshot: str, target_snapshot: str) -> str:
    rewritten = Path(str(path_value).replace(source_snapshot, target_snapshot))
    return str(rewritten.resolve()) if rewritten.exists() else str(path_value)


def _counts(values: Any) -> dict[str, int]:
    if values is None:
        return {}
    clean = []
    for value in list(values):
        parsed = _safe_int(value)
        if parsed is not None:
            clean.append(parsed)
    return {str(k): int(v) for k, v in sorted(Counter(clean).items())}


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


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone a snapshot and apply human gold curation scores to metadata labels.")
    parser.add_argument("--source-snapshot", required=True)
    parser.add_argument("--target-snapshot", required=True)
    parser.add_argument("--gold-file", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = apply_gold_scores_to_snapshot(
        source_snapshot=args.source_snapshot,
        target_snapshot=args.target_snapshot,
        gold_file=Path(args.gold_file) if args.gold_file else None,
        data_root=Path(args.data_root) if args.data_root else None,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
