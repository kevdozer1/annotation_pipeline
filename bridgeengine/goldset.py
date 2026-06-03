from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from bridgeengine.paths import data_root as resolve_data_root


def write_gold_template(
    snapshot_id: str,
    output_path: str | Path,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    root = resolve_data_root(data_root)
    snapshot_path = root / "snapshots" / snapshot_id
    episodes = pd.read_parquet(snapshot_path / "episodes.parquet")
    labels = pd.read_parquet(snapshot_path / "labels.parquet")
    entries = []
    for _, episode in episodes.sort_values("episode_id").iterrows():
        episode_id = str(episode["episode_id"])
        ep_labels = labels[labels["episode_id"] == episode_id]
        auto_subtasks = _payload_for(ep_labels, "subtask_segmenter").get("segments", [])
        auto_metadata = _payload_for(ep_labels, "episode_metadata").get("metadata", {})
        auto_subgoals = _subgoal_payloads(ep_labels)
        entries.append(
            {
                "episode_id": episode_id,
                "task": episode.get("language_instruction"),
                "num_steps": int(episode.get("num_steps", 0)),
                "auto": {
                    "subtasks": auto_subtasks,
                    "metadata": auto_metadata,
                    "subgoals": auto_subgoals,
                },
                "gold": {
                    "subtasks": [
                        {
                            "segment_idx": s.get("segment_idx"),
                            "start_step": None,
                            "end_step": None,
                            "subtask_text": None,
                            "accept_auto": None,
                        }
                        for s in auto_subtasks
                    ],
                    "metadata": {
                        "quality": None,
                        "mistake": None,
                        "control_mode": "end_effector",
                        "reason": None,
                        "accept_auto": None,
                    },
                    "subgoals": [
                        {
                            "segment_idx": s.get("segment_idx"),
                            "frame_idx": None,
                            "accept_auto": None,
                        }
                        for s in auto_subgoals
                    ],
                },
                "review_notes": "",
            }
        )
    payload = {
        "schema_version": "gold_set_v1",
        "snapshot_id": snapshot_id,
        "instructions": (
            "Fill the gold fields by confirming or correcting the auto labels. "
            "Leave auto fields unchanged; BridgeEngine uses them for reliability comparisons."
        ),
        "episodes": entries,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def reliability_report(
    snapshot_id: str,
    gold_file: str | Path,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    root = resolve_data_root(data_root)
    snapshot_path = root / "snapshots" / snapshot_id
    labels = pd.read_parquet(snapshot_path / "labels.parquet")
    manifest = _read_json(snapshot_path / "manifest.json")
    gold = _read_json(Path(gold_file))
    boundary_ious: list[float] = []
    exact_quality: list[bool] = []
    within_one_quality: list[bool] = []
    subgoal_matches: list[bool] = []
    reviewed = 0
    per_episode: list[dict[str, Any]] = []
    for entry in gold.get("episodes", []):
        episode_id = str(entry.get("episode_id"))
        ep_labels = labels[labels["episode_id"] == episode_id]
        auto_subtasks = _payload_for(ep_labels, "subtask_segmenter").get("segments", [])
        auto_metadata = _payload_for(ep_labels, "episode_metadata").get("metadata", {})
        auto_subgoals = _subgoal_payloads(ep_labels)
        gold_subtasks = _gold_subtasks(entry)
        gold_metadata = _gold_metadata(entry)
        gold_subgoals = _gold_subgoals(entry)
        if gold_subtasks or gold_metadata or gold_subgoals:
            reviewed += 1

        episode_ious = [_temporal_iou(a, g) for a, g in zip(auto_subtasks, gold_subtasks)]
        episode_ious = [x for x in episode_ious if x is not None]
        boundary_ious.extend(episode_ious)

        q_auto = _safe_int(auto_metadata.get("quality"))
        q_gold = _safe_int(gold_metadata.get("quality"))
        q_exact = q_auto is not None and q_gold is not None and q_auto == q_gold
        q_within = q_auto is not None and q_gold is not None and abs(q_auto - q_gold) <= 1
        if q_gold is not None:
            exact_quality.append(q_exact)
            within_one_quality.append(q_within)

        auto_frames = {int(s.get("segment_idx")): _safe_int(s.get("frame_idx")) for s in auto_subgoals if s.get("segment_idx") is not None}
        for gold_subgoal in gold_subgoals:
            idx = _safe_int(gold_subgoal.get("segment_idx"))
            frame = _safe_int(gold_subgoal.get("frame_idx"))
            if idx is not None and frame is not None:
                subgoal_matches.append(auto_frames.get(idx) == frame)

        per_episode.append(
            {
                "episode_id": episode_id,
                "boundary_iou_mean": _mean_or_none(episode_ious),
                "quality_exact": q_exact if q_gold is not None else None,
                "quality_within_one": q_within if q_gold is not None else None,
                "wall_clock_seconds_labeling": _episode_label_seconds(ep_labels),
                "estimated_cost_usd_labeling": _episode_label_cost(ep_labels),
            }
        )
    episode_count = len(gold.get("episodes", []))
    total_runtime = sum(float(x) for x in manifest.get("labeler_runtime_seconds", {}).values())
    report = {
        "snapshot_id": snapshot_id,
        "gold_file": str(Path(gold_file).resolve()),
        "episode_count": episode_count,
        "reviewed_episode_count": reviewed,
        "subtask_boundary_temporal_iou_mean": _mean_or_none(boundary_ious),
        "quality_exact_agreement": _bool_mean(exact_quality),
        "quality_within_one_agreement": _bool_mean(within_one_quality),
        "subgoal_selection_agreement": _bool_mean(subgoal_matches),
        "labeling_wall_clock_seconds_total": total_runtime,
        "labeling_wall_clock_seconds_per_episode": total_runtime / episode_count if episode_count else None,
        "estimated_cost_usd_total": _snapshot_cost(labels),
        "estimated_cost_usd_per_episode": _snapshot_cost(labels) / episode_count if episode_count and _snapshot_cost(labels) is not None else None,
        "parallelism_note": "Episodes are labeled independently; labeling cost is linear and shardable across episodes.",
        "per_episode": per_episode,
    }
    return report


def _payload_for(labels: pd.DataFrame, labeler_name: str) -> dict[str, Any]:
    rows = labels[labels["labeler_name"] == labeler_name]
    if rows.empty:
        return {}
    path = Path(str(rows.iloc[0]["label_payload_path"]))
    if not path.exists():
        return {}
    return _read_json(path)


def _subgoal_payloads(labels: pd.DataFrame) -> list[dict[str, Any]]:
    rows = labels[labels["labeler_name"] == "subgoal_images"].sort_values("segment_idx")
    payloads = []
    for _, row in rows.iterrows():
        path = Path(str(row["label_payload_path"]))
        if path.exists():
            payloads.append(_read_json(path))
    return payloads


def _gold_subtasks(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in entry.get("gold", {}).get("subtasks", [])
        if item.get("accept_auto") or item.get("start_step") is not None or item.get("end_step") is not None
    ]


def _gold_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    metadata = entry.get("gold", {}).get("metadata", {})
    if metadata.get("accept_auto") or metadata.get("quality") is not None:
        return metadata
    return {}


def _gold_subgoals(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in entry.get("gold", {}).get("subgoals", [])
        if item.get("accept_auto") or item.get("frame_idx") is not None
    ]


def _temporal_iou(auto: dict[str, Any], gold: dict[str, Any]) -> float | None:
    if gold.get("accept_auto"):
        gold = {**gold, "start_step": auto.get("start_step"), "end_step": auto.get("end_step")}
    a0 = _safe_int(auto.get("start_step"))
    a1 = _safe_int(auto.get("end_step"))
    g0 = _safe_int(gold.get("start_step"))
    g1 = _safe_int(gold.get("end_step"))
    if None in {a0, a1, g0, g1}:
        return None
    intersection = max(0, min(a1, g1) - max(a0, g0) + 1)
    union = max(a1, g1) - min(a0, g0) + 1
    return intersection / union if union > 0 else None


def _episode_label_seconds(labels: pd.DataFrame) -> float:
    total = 0.0
    for value in labels["provenance_json"].dropna().tolist():
        data = _parse_json(value)
        total += float(data.get("wall_clock_seconds", 0.0))
    return round(total, 6)


def _episode_label_cost(labels: pd.DataFrame) -> float | None:
    costs = []
    for value in labels["provenance_json"].dropna().tolist():
        data = _parse_json(value)
        cost = data.get("estimated_cost_usd")
        if cost is not None:
            costs.append(float(cost))
    return round(sum(costs), 8) if costs else None


def _snapshot_cost(labels: pd.DataFrame) -> float | None:
    costs = [_episode_label_cost(group) for _, group in labels.groupby("episode_id")]
    present = [cost for cost in costs if cost is not None]
    return round(sum(present), 8) if present else None


def _bool_mean(values: list[bool]) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    parser = argparse.ArgumentParser(description="Create gold-label templates and report reliability against them.")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Write a human-editable gold-set template from auto-labels.")
    init.add_argument("--snapshot", required=True)
    init.add_argument("--output", required=True)
    init.add_argument("--data-root", default=None)
    report = sub.add_parser("report", help="Compute reliability metrics against a filled gold set.")
    report.add_argument("--snapshot", required=True)
    report.add_argument("--gold-file", required=True)
    report.add_argument("--data-root", default=None)
    args = parser.parse_args()
    if args.command == "init":
        payload = write_gold_template(args.snapshot, args.output, Path(args.data_root) if args.data_root else None)
        print(json.dumps({"output": str(Path(args.output).resolve()), "episodes": len(payload["episodes"])}, indent=2))
    else:
        payload = reliability_report(args.snapshot, args.gold_file, Path(args.data_root) if args.data_root else None)
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
