from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from bridgeengine.paths import data_root as resolve_data_root


SCORING_VERSION = "boundary-usefulness-v3"


@dataclass(frozen=True)
class EpisodeScore:
    episode_id: str
    task_success_quality: int | None
    curation_quality: int
    curation_keep: bool
    boundary_clarity: str
    structure_score: int
    scoring_reason: str


def rescore_snapshot(
    snapshot_id: str,
    data_root: str | Path | None = None,
    threshold: int = 4,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = resolve_data_root(data_root)
    snapshot_path = root / "snapshots" / snapshot_id
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")
    labels_path = snapshot_path / "labels.parquet"
    labels = pd.read_parquet(labels_path)
    before = _quality_counts(labels, prefer_curation=False)
    task_success_before = _task_success_counts(labels)
    rows = labels.to_dict("records")
    segment_payloads = _segment_payloads(rows)
    scores: list[EpisodeScore] = []

    for idx, row in enumerate(rows):
        if row.get("labeler_name") != "episode_metadata":
            continue
        episode_id = str(row.get("episode_id"))
        payload_path = Path(str(row.get("label_payload_path") or ""))
        payload = _read_json(payload_path)
        metadata = _parse_json(row.get("metadata_payload_json")) or dict(payload.get("metadata", {}))
        score = score_metadata_for_curation(
            episode_id=episode_id,
            metadata=metadata,
            segments=segment_payloads.get(episode_id, []),
            threshold=threshold,
        )
        scores.append(score)
        updated = dict(metadata)
        updated.setdefault("task_success_quality", _safe_int(metadata.get("task_success_quality")) or _safe_int(metadata.get("quality")))
        updated["quality"] = int(score.curation_quality)
        updated["curation_quality"] = int(score.curation_quality)
        updated["curation_keep"] = bool(score.curation_keep)
        updated["boundary_clarity"] = score.boundary_clarity
        updated["interaction_structure_score"] = int(score.structure_score)
        updated["scoring_basis"] = "visible_boundary_training_usefulness"
        updated["scoring_version"] = SCORING_VERSION
        updated["scoring_reason"] = score.scoring_reason
        rows[idx]["metadata_payload_json"] = json.dumps(updated, sort_keys=True)

        if payload_path.exists() and payload:
            payload["metadata"] = updated
            if not dry_run:
                _write_json(payload_path, payload)

    after_df = pd.DataFrame(rows, columns=labels.columns)
    after = _quality_counts(after_df, prefer_curation=True)
    changed = [
        {
            "episode_id": score.episode_id,
            "task_success_quality": score.task_success_quality,
            "curation_quality": score.curation_quality,
            "curation_keep": score.curation_keep,
            "boundary_clarity": score.boundary_clarity,
            "structure_score": score.structure_score,
            "scoring_reason": score.scoring_reason,
        }
        for score in scores
        if score.task_success_quality is not None and score.task_success_quality != score.curation_quality
    ]
    report = {
        "snapshot_id": snapshot_id,
        "scoring_version": SCORING_VERSION,
        "threshold": int(threshold),
        "dry_run": bool(dry_run),
        "episode_count": len(scores),
        "before_quality_counts": before,
        "task_success_quality_counts": task_success_before,
        "after_quality_counts": after,
        "changed_episode_count": len(changed),
        "changed_episodes": changed,
    }
    if not dry_run:
        after_df.to_parquet(labels_path, index=False)
        _write_json(snapshot_path / "curation_scoring_report.json", report)
    return report


def score_metadata_for_curation(
    episode_id: str,
    metadata: dict[str, Any],
    segments: list[dict[str, Any]],
    threshold: int = 4,
) -> EpisodeScore:
    task_success_quality = _safe_int(metadata.get("task_success_quality"))
    if task_success_quality is None:
        task_success_quality = _safe_int(metadata.get("quality"))
    reason = str(metadata.get("reason", "")).lower()
    mistake = bool(metadata.get("mistake"))
    structure = _interaction_structure(segments)
    structure_score = sum(
        [
            structure["has_approach"],
            structure["has_grasp"],
            structure["has_transport"],
            structure["has_release_or_place"],
        ]
    )
    strong_reject = _strong_reject_reason(reason)
    wrong_object = _has_unnegated_phrase(reason, ["wrong object", "wrong target", "not the requested object"]) or bool(
        re.search(r"\bno (?:target object|requested object|eggplant)\b", reason)
    )
    weak_outcome = _weak_outcome_reason(reason)
    destination_absent = _destination_absent_reason(reason)
    localized_attempt = _localized_end_state_attempt(segments, structure_score)
    no_visible_transfer = (
        "does not visibly grasp" in reason
        or re.search(r"\bno .{0,48}visibly grasp", reason) is not None
        or "never visibly" in reason
        or "pot appears empty" in reason
    )

    if strong_reject and destination_absent and len(segments) <= 2:
        quality = 1
        boundary = "weak"
        scoring_reason = "clear reject: short unfinished interaction with no reached target state"
    elif strong_reject and no_visible_transfer and localized_attempt:
        quality = 3
        boundary = "partial"
        scoring_reason = "near reject: localized target contact/end-state attempt is visible but not reliable"
    elif strong_reject and no_visible_transfer:
        quality = 1
        boundary = "weak"
        scoring_reason = "reject: no visible completed grasp/release cycle"
    elif wrong_object:
        quality = 2
        boundary = "partial" if structure_score >= 3 else "weak"
        scoring_reason = "reject: visible structure is tied to the wrong or missing target object"
    elif (
        task_success_quality is not None
        and task_success_quality >= 5
        and not mistake
        and not strong_reject
        and (structure_score >= 2 or _pickup_success_reason(reason))
    ):
        quality = 5
        boundary = "clear"
        scoring_reason = "clear keep: successful task with visible contact/transport structure"
    elif structure_score >= 4 and len(segments) >= 2:
        boundary = "clear"
        if task_success_quality is not None and task_success_quality >= 4 and not mistake and len(segments) >= 4:
            quality = 5
            scoring_reason = "clear keep: long stacked interaction with visible subtask boundaries"
        elif task_success_quality is not None and task_success_quality >= 5 and not mistake and not weak_outcome:
            quality = 5
            scoring_reason = "clear keep: clean interaction cycle with low ambiguity"
        else:
            quality = 4
            scoring_reason = "keep: visible approach/grasp/transport/release boundaries"
    elif structure_score >= 3 and len(segments) >= 2:
        if task_success_quality is not None and task_success_quality >= 5 and not mistake and not strong_reject and not weak_outcome:
            quality = 5
            boundary = "clear"
            scoring_reason = "clear keep: clean visible interaction cycle for the requested task"
        else:
            quality = 4 if not strong_reject else 3
            boundary = "clear" if quality >= 4 else "partial"
            scoring_reason = "keep: interaction boundaries are visible even though outcome is imperfect" if quality >= 4 else "near reject: partial interaction with major visible uncertainty"
    elif structure_score >= 2:
        quality = 3
        boundary = "partial"
        scoring_reason = "near reject: some interaction is visible, but the cycle is incomplete"
    else:
        quality = 2 if task_success_quality and task_success_quality >= 3 else 1
        boundary = "weak"
        scoring_reason = "reject: weak or missing segmentable manipulation structure"

    return EpisodeScore(
        episode_id=episode_id,
        task_success_quality=task_success_quality,
        curation_quality=int(max(1, min(5, quality))),
        curation_keep=int(max(1, min(5, quality))) >= threshold,
        boundary_clarity=boundary,
        structure_score=int(structure_score),
        scoring_reason=scoring_reason,
    )


def metadata_quality(metadata: dict[str, Any]) -> int | None:
    return _safe_int(metadata.get("curation_quality")) or _safe_int(metadata.get("quality"))


def task_success_quality(metadata: dict[str, Any]) -> int | None:
    return _safe_int(metadata.get("task_success_quality")) or _safe_int(metadata.get("quality"))


def _interaction_structure(segments: list[dict[str, Any]]) -> dict[str, bool]:
    text = " ".join(str(segment.get("subtask_text", "")).lower() for segment in segments)
    return {
        "has_approach": bool(re.search(r"\b(move|approach|align|position|reach|go)\b", text)),
        "has_grasp": bool(re.search(r"\b(grasp|grip|pick|lift|hold)\b", text)),
        "has_transport": bool(re.search(r"\b(carry|transport|move|toward|over|into|onto|across|to)\b", text)),
        "has_release_or_place": bool(re.search(r"\b(release|place|put|lower|set|settle|drop|position)\b", text)),
    }


def _strong_reject_reason(reason: str) -> bool:
    if _has_unnegated_phrase(reason, ["wrong destination", "wrong object", "wrong target", "not the requested object"]):
        return True
    patterns = [
        r"does not visibly grasp",
        r"\bno .{0,48}visibly grasp",
        r"does not visibly .*release",
        r"never visibly",
        r"task is unfinished",
        r"major failure",
        r"pot appears empty",
        r"appears empty",
    ]
    return any(re.search(pattern, reason) for pattern in patterns)


def _has_unnegated_phrase(text: str, phrases: list[str]) -> bool:
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    negators = {"no", "not", "without", "none", "never"}
    for phrase in phrases:
        phrase_tokens = re.findall(r"[a-z0-9_]+", phrase.lower())
        if not phrase_tokens:
            continue
        n = len(phrase_tokens)
        for idx in range(0, len(tokens) - n + 1):
            if tokens[idx : idx + n] != phrase_tokens:
                continue
            window = tokens[max(0, idx - 8) : idx]
            if any(token in negators for token in window):
                continue
            return True
    return False


def _weak_outcome_reason(reason: str) -> bool:
    patterns = [
        r"not clearly",
        r"not fully clear",
        r"obscured",
        r"somewhat",
        r"not perfectly",
        r"partially",
    ]
    return any(re.search(pattern, reason) for pattern in patterns)


def _destination_absent_reason(reason: str) -> bool:
    patterns = [
        r"destination is not reached",
        r"never transports",
        r"never .{0,32}places?",
        r"never .{0,32}placed",
        r"never reaches?",
        r"does not .{0,32}places?",
        r"does not .{0,32}reach",
        r"task is unfinished",
    ]
    return any(re.search(pattern, reason) for pattern in patterns)


def _localized_end_state_attempt(segments: list[dict[str, Any]], structure_score: int) -> bool:
    if len(segments) < 3 or structure_score < 3:
        return False
    text = " ".join(str(segment.get("subtask_text", "")).lower() for segment in segments)
    return bool(re.search(r"\b(grip|grasp|stabilize|lower|position|contact|hold)\b", text))


def _pickup_success_reason(reason: str) -> bool:
    return bool(re.search(r"\b(lift|lifted|pick up|picked up|pickup|picked)\b", reason))


def _segment_payloads(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("labeler_name") != "subtask_segmenter":
            continue
        payload = _read_json(Path(str(row.get("label_payload_path") or "")))
        result[str(row.get("episode_id"))] = list(payload.get("segments", []))
    return result


def _quality_counts(labels: pd.DataFrame, prefer_curation: bool) -> dict[int, int]:
    values = []
    if labels.empty or "metadata_payload_json" not in labels.columns:
        return {}
    for raw in labels.loc[labels["labeler_name"] == "episode_metadata", "metadata_payload_json"].dropna().tolist():
        metadata = _parse_json(raw)
        quality = metadata_quality(metadata) if prefer_curation else _safe_int(metadata.get("quality"))
        if quality is not None:
            values.append(int(quality))
    return {int(k): int(v) for k, v in pd.Series(values, dtype="int64").value_counts().sort_index().to_dict().items()} if values else {}


def _task_success_counts(labels: pd.DataFrame) -> dict[int, int]:
    values = []
    if labels.empty or "metadata_payload_json" not in labels.columns:
        return {}
    for raw in labels.loc[labels["labeler_name"] == "episode_metadata", "metadata_payload_json"].dropna().tolist():
        metadata = _parse_json(raw)
        quality = _safe_int(metadata.get("task_success_quality")) or _safe_int(metadata.get("quality"))
        if quality is not None:
            values.append(int(quality))
    return {int(k): int(v) for k, v in pd.Series(values, dtype="int64").value_counts().sort_index().to_dict().items()} if values else {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"Curation scoring: {report['snapshot_id']}",
        f"Version: {report['scoring_version']}",
        f"Threshold: quality >= {report['threshold']} keep",
        f"Current/pre-rescore quality counts: {report['before_quality_counts']}",
        f"Original task-success quality counts: {report['task_success_quality_counts']}",
        f"Curation quality counts: {report['after_quality_counts']}",
        f"Changed episodes: {report['changed_episode_count']} / {report['episode_count']}",
    ]
    for item in report["changed_episodes"][:20]:
        lines.append(
            "- "
            f"{item['episode_id']}: task_success={item['task_success_quality']} "
            f"-> curation={item['curation_quality']} keep={item['curation_keep']} "
            f"({item['scoring_reason']})"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore BridgeEngine metadata for boundary-usefulness curation.")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--threshold", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = rescore_snapshot(
        args.snapshot,
        data_root=Path(args.data_root) if args.data_root else None,
        threshold=args.threshold,
        dry_run=args.dry_run,
    )
    print(format_report(report))


if __name__ == "__main__":
    main()
