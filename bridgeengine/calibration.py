from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from bridgeengine.goldset import reliability_report, write_gold_template
from bridgeengine.paths import data_root as resolve_data_root


def default_gold_path(snapshot_id: str, data_root: str | Path | None = None) -> Path:
    root = resolve_data_root(data_root)
    return root / "snapshots" / snapshot_id / "gold" / "calibration_gold.json"


def load_or_create_calibration_gold(
    snapshot_id: str,
    gold_file: str | Path | None = None,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(gold_file) if gold_file else default_gold_path(snapshot_id, data_root)
    if not path.exists():
        return write_gold_template(snapshot_id, path, data_root=data_root)

    existing = _read_json(path)
    template_path = path.with_suffix(".template.tmp.json")
    template = write_gold_template(snapshot_id, template_path, data_root=data_root)
    try:
        template_path.unlink(missing_ok=True)
    except OSError:
        pass

    by_episode = {str(entry.get("episode_id")): entry for entry in existing.get("episodes", [])}
    synced = []
    changed = False
    for template_entry in template.get("episodes", []):
        episode_id = str(template_entry.get("episode_id"))
        if episode_id in by_episode:
            merged = _merge_entry(template_entry, by_episode[episode_id])
            synced.append(merged)
        else:
            synced.append(template_entry)
            changed = True
    if len(synced) != len(existing.get("episodes", [])):
        changed = True

    payload = {
        **template,
        "episodes": synced,
        "calibration_note": "Generated or updated by BridgeEngine calibration UI. Auto fields are retained for reliability reports.",
    }
    if changed or payload != existing:
        _write_json(path, payload)
    return payload


def update_episode_review(
    snapshot_id: str,
    episode_id: str,
    curation_quality: int,
    mistake: bool | None = None,
    reason: str | None = None,
    review_notes: str | None = None,
    accept_auto_metadata: bool = False,
    accept_auto_subtasks: bool | None = None,
    accept_auto_subgoals: bool | None = None,
    gold_file: str | Path | None = None,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(gold_file) if gold_file else default_gold_path(snapshot_id, data_root)
    gold = load_or_create_calibration_gold(snapshot_id, path, data_root=data_root)
    entry = _entry_for(gold, episode_id)
    auto_metadata = entry.get("auto", {}).get("metadata", {}) or {}
    quality = int(max(1, min(5, curation_quality)))
    gold_metadata = entry.setdefault("gold", {}).setdefault("metadata", {})
    gold_metadata["quality"] = quality
    gold_metadata["curation_quality"] = quality
    gold_metadata["curation_keep"] = quality >= 4
    gold_metadata["mistake"] = bool(mistake) if mistake is not None else bool(auto_metadata.get("mistake", False))
    gold_metadata["control_mode"] = auto_metadata.get("control_mode", "end_effector")
    gold_metadata["reason"] = reason or ""
    gold_metadata["accept_auto"] = bool(accept_auto_metadata)

    if accept_auto_subtasks is not None:
        for subtask in entry.get("gold", {}).get("subtasks", []):
            subtask["accept_auto"] = bool(accept_auto_subtasks)
    if accept_auto_subgoals is not None:
        auto_subgoals = entry.get("auto", {}).get("subgoals", [])
        auto_frames = {s.get("segment_idx"): s.get("frame_idx") for s in auto_subgoals}
        for subgoal in entry.get("gold", {}).get("subgoals", []):
            subgoal["accept_auto"] = bool(accept_auto_subgoals)
            if accept_auto_subgoals:
                subgoal["frame_idx"] = auto_frames.get(subgoal.get("segment_idx"))
            else:
                subgoal["frame_idx"] = None

    entry["review_notes"] = review_notes or ""
    _write_json(path, gold)
    return entry


def review_summary(
    snapshot_id: str,
    gold_file: str | Path | None = None,
    data_root: str | Path | None = None,
) -> pd.DataFrame:
    path = Path(gold_file) if gold_file else default_gold_path(snapshot_id, data_root)
    gold = load_or_create_calibration_gold(snapshot_id, path, data_root=data_root)
    rows = []
    for entry in gold.get("episodes", []):
        auto_metadata = entry.get("auto", {}).get("metadata", {}) or {}
        gold_metadata = entry.get("gold", {}).get("metadata", {}) or {}
        gold_quality = _safe_int(gold_metadata.get("curation_quality"))
        if gold_quality is None:
            gold_quality = _safe_int(gold_metadata.get("quality"))
        auto_quality = _safe_int(auto_metadata.get("curation_quality"))
        if auto_quality is None:
            auto_quality = _safe_int(auto_metadata.get("quality"))
        reviewed = gold_quality is not None
        rows.append(
            {
                "reviewed": reviewed,
                "episode_id": entry.get("episode_id"),
                "task": entry.get("task"),
                "num_steps": entry.get("num_steps"),
                "auto_score": auto_quality,
                "gold_score": gold_quality,
                "auto_keep": _safe_bool(auto_metadata.get("curation_keep")),
                "gold_keep": gold_quality >= 4 if gold_quality is not None else None,
                "boundary_clarity": auto_metadata.get("boundary_clarity"),
                "auto_mistake": _safe_bool(auto_metadata.get("mistake")),
                "gold_mistake": _safe_bool(gold_metadata.get("mistake")),
                "notes": entry.get("review_notes", ""),
            }
        )
    return pd.DataFrame(rows)


def calibration_reliability(
    snapshot_id: str,
    gold_file: str | Path | None = None,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(gold_file) if gold_file else default_gold_path(snapshot_id, data_root)
    load_or_create_calibration_gold(snapshot_id, path, data_root=data_root)
    return reliability_report(snapshot_id, path, data_root=data_root)


def _merge_entry(template_entry: dict[str, Any], existing_entry: dict[str, Any]) -> dict[str, Any]:
    merged = {**template_entry, "gold": template_entry.get("gold", {})}
    merged["gold"] = _merge_gold(template_entry.get("gold", {}), existing_entry.get("gold", {}))
    merged["review_notes"] = existing_entry.get("review_notes", template_entry.get("review_notes", ""))
    return merged


def _merge_gold(template_gold: dict[str, Any], existing_gold: dict[str, Any]) -> dict[str, Any]:
    merged = dict(template_gold)
    merged["metadata"] = {**template_gold.get("metadata", {}), **existing_gold.get("metadata", {})}
    merged["subtasks"] = _merge_indexed(template_gold.get("subtasks", []), existing_gold.get("subtasks", []), "segment_idx")
    merged["subgoals"] = _merge_indexed(template_gold.get("subgoals", []), existing_gold.get("subgoals", []), "segment_idx")
    return merged


def _merge_indexed(template_items: list[dict[str, Any]], existing_items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    existing_by_key = {item.get(key): item for item in existing_items}
    merged = []
    for item in template_items:
        idx = item.get(key)
        merged.append({**item, **existing_by_key.get(idx, {})})
    return merged


def _entry_for(gold: dict[str, Any], episode_id: str) -> dict[str, Any]:
    for entry in gold.get("episodes", []):
        if str(entry.get("episode_id")) == str(episode_id):
            return entry
    raise KeyError(f"Episode not present in gold file: {episode_id}")


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in {"true", "yes", "1"}:
            return True
        if value.lower() in {"false", "no", "0"}:
            return False
    return bool(value)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
