from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from bridgeengine.calibration import default_gold_path, review_summary
from bridgeengine.derive_subgoals import DERIVED_SUBGOAL_SOURCE
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
    gold_entries = _gold_entries(gold)

    labels_path = target_path / "labels.parquet"
    episodes = pd.read_parquet(target_path / "episodes.parquet")
    episode_rows = {str(row["episode_id"]): row for _, row in episodes.iterrows()}
    labels = pd.read_parquet(labels_path)
    rows = labels.to_dict("records")
    changed = []
    boundary_updates = []
    subgoal_updates = []
    for row in rows:
        row["snapshot_id"] = target_snapshot
        if row.get("label_payload_path"):
            row["label_payload_path"] = _copy_or_rewrite_path(row["label_payload_path"], root, target_snapshot)
        if row.get("subgoal_image_path"):
            row["subgoal_image_path"] = _copy_or_rewrite_path(row["subgoal_image_path"], root, target_snapshot)
        labeler_name = str(row.get("labeler_name"))
        episode_id = str(row.get("episode_id"))
        gold_entry = gold_entries.get(episode_id)
        if labeler_name == "subtask_segmenter" and gold_entry:
            payload_path = Path(str(row.get("label_payload_path") or ""))
            update = _apply_gold_subtasks(payload_path, gold_entry, row)
            if update:
                boundary_updates.append({"episode_id": episode_id, **update})
            continue
        if labeler_name == "subgoal_images" and gold_entry:
            update = _apply_gold_subgoal(row, gold_entry, episode_rows.get(episode_id), target_path)
            if update:
                subgoal_updates.append({"episode_id": episode_id, **update})
            continue
        if labeler_name != "episode_metadata":
            continue
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
        "applied_boundary_episode_count": len(boundary_updates),
        "applied_boundary_updates": boundary_updates,
        "applied_subgoal_count": len(subgoal_updates),
        "applied_subgoal_updates": subgoal_updates,
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


def _gold_entries(gold: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(entry.get("episode_id")): entry for entry in gold.get("episodes", []) if entry.get("episode_id") is not None}


def _apply_gold_subtasks(payload_path: Path, gold_entry: dict[str, Any], row: dict[str, Any]) -> dict[str, Any] | None:
    if not payload_path.exists():
        return None
    payload = _read_json(payload_path)
    auto_segments = payload.get("segments") or gold_entry.get("auto", {}).get("subtasks", [])
    segments = _human_segments(gold_entry, auto_segments)
    if not segments:
        return None
    payload["segments"] = segments
    payload["boundary_source"] = "human_gold_boundary_review"
    payload["human_gold_applied"] = True
    payload["human_gold_source"] = str(gold_entry.get("episode_id"))
    payload["prompt_components"] = [
        f"Task: {gold_entry.get('task')}. Subtask: {segment.get('subtask_text')}."
        for segment in segments
    ]
    _write_json(payload_path, payload)
    _add_provenance(row, {"human_gold_boundaries_applied": True, "boundary_source": "human_gold_boundary_review"})
    return {"segment_count": len(segments)}


def _apply_gold_subgoal(
    row: dict[str, Any],
    gold_entry: dict[str, Any],
    episode_row: pd.Series | None,
    target_path: Path,
) -> dict[str, Any] | None:
    if episode_row is None:
        return None
    segment_idx = _safe_int(row.get("segment_idx"))
    if segment_idx is None:
        return None
    gold_subgoal = _gold_item_by_idx(gold_entry, "subgoals").get(segment_idx)
    frame_idx = _safe_int(gold_subgoal.get("frame_idx")) if gold_subgoal else None
    if frame_idx is None and gold_subgoal and gold_subgoal.get("accept_auto"):
        frame_idx = _safe_int(_gold_item_by_idx(gold_entry, "subgoals", auto=True).get(segment_idx, {}).get("frame_idx"))
    if frame_idx is None:
        return None

    frames_path = Path(str(episode_row.get("source_path_frames") or ""))
    if not frames_path.exists():
        episode_dir = Path(str(episode_row.get("source_path_meta") or "")).parent
        frames_path = episode_dir / "frames.npy"
    if not frames_path.exists():
        raise FileNotFoundError(f"frames.npy missing for human-gold subgoal extraction: {frames_path}")

    frames = np.load(frames_path, mmap_mode="r")
    frame_idx = max(0, min(int(frame_idx), int(frames.shape[0]) - 1))
    episode_id = str(row.get("episode_id"))
    out_dir = target_path / "subgoals" / episode_id
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / f"{segment_idx:02d}.jpg"
    Image.fromarray(np.asarray(frames[frame_idx], dtype=np.uint8)).save(image_path, quality=92)

    payload_path = Path(str(row.get("label_payload_path") or out_dir / f"{segment_idx:02d}.json"))
    if not payload_path.exists():
        payload_path = out_dir / f"{segment_idx:02d}.json"
    payload = _read_json(payload_path) if payload_path.exists() else {}
    subtask = _gold_item_by_idx(gold_entry, "subtasks").get(segment_idx, {})
    payload.update(
        {
            "episode_id": episode_id,
            "segment_idx": segment_idx,
            "frame_idx": frame_idx,
            "subtask_text": subtask.get("subtask_text") or payload.get("subtask_text"),
            "subgoal_image_path": str(image_path.resolve()),
            "source": DERIVED_SUBGOAL_SOURCE,
            "human_gold_applied": True,
        }
    )
    _write_json(payload_path, payload)
    row["label_payload_path"] = str(payload_path.resolve())
    row["subgoal_image_path"] = str(image_path.resolve())
    _add_provenance(
        row,
        {
            "human_gold_subgoal_applied": True,
            "source": DERIVED_SUBGOAL_SOURCE,
            "frame_idx": frame_idx,
        },
    )
    return {"segment_idx": segment_idx, "frame_idx": frame_idx}


def _human_segments(gold_entry: dict[str, Any], auto_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    num_steps = _safe_int(gold_entry.get("num_steps"))
    gold_by_idx = _gold_item_by_idx(gold_entry, "subtasks")
    result = []
    for auto in sorted(auto_segments, key=lambda item: _safe_int(item.get("segment_idx")) or 0):
        idx = _safe_int(auto.get("segment_idx"))
        if idx is None:
            continue
        gold = gold_by_idx.get(idx, {})
        if gold.get("accept_auto"):
            start = _safe_int(auto.get("start_step"))
            end = _safe_int(auto.get("end_step"))
            text = auto.get("subtask_text")
        elif gold.get("start_step") is not None or gold.get("end_step") is not None:
            start = _safe_int(gold.get("start_step"))
            end = _safe_int(gold.get("end_step"))
            text = gold.get("subtask_text") or auto.get("subtask_text")
        else:
            start = _safe_int(auto.get("start_step"))
            end = _safe_int(auto.get("end_step"))
            text = auto.get("subtask_text")
        if start is None or end is None:
            continue
        start = _clamp_step(start, num_steps)
        end = _clamp_step(end, num_steps)
        if end < start:
            end = start
        result.append(
            {
                **auto,
                "segment_idx": idx,
                "start_step": start,
                "end_step": end,
                "subtask_text": str(text or ""),
                "source": "human_gold_boundary_review",
                "human_gold_applied": True,
            }
        )
    return result


def _gold_item_by_idx(entry: dict[str, Any], key: str, auto: bool = False) -> dict[int, dict[str, Any]]:
    source = "auto" if auto else "gold"
    items = entry.get(source, {}).get(key, [])
    result = {}
    for item in items:
        idx = _safe_int(item.get("segment_idx"))
        if idx is not None:
            result[idx] = item
    return result


def _copy_or_rewrite_path(path_value: Any, root: Path, target_snapshot: str) -> str:
    if not _has_path_value(path_value):
        return str(path_value)
    source = Path(str(path_value))
    target = _target_snapshot_path(source, root, target_snapshot)
    if target == source:
        return str(source)
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        return str(target.resolve())
    return str(target)


def _target_snapshot_path(source: Path, root: Path, target_snapshot: str) -> Path:
    parts = list(source.parts)
    for idx, part in enumerate(parts):
        if part.startswith("snap_"):
            parts[idx] = target_snapshot
            return Path(*parts)
    try:
        rel = source.resolve().relative_to(root.resolve())
    except ValueError:
        return root / "snapshots" / target_snapshot / "copied_artifacts" / source.name
    return root / "snapshots" / target_snapshot / "copied_artifacts" / rel.name


def _has_path_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() not in {"", "None", "nan"}


def _add_provenance(row: dict[str, Any], update: dict[str, Any]) -> None:
    provenance = _parse_json(row.get("provenance_json"))
    provenance.update(update)
    row["provenance_json"] = json.dumps(provenance, sort_keys=True)


def _clamp_step(value: int, num_steps: int | None) -> int:
    if num_steps is not None and num_steps > 0:
        return max(0, min(int(value), int(num_steps) - 1))
    return max(0, int(value))


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
