from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from bridgeengine.calibration import default_gold_path, load_or_create_calibration_gold, review_summary
from bridgeengine.paths import data_root as resolve_data_root


DEFAULT_SNAPSHOT = "snap_2026_05_11_1dde3edf5d"


def plan_reliability_review(
    snapshot_id: str,
    output_path: str | Path,
    *,
    count: int = 50,
    gold_file: str | Path | None = None,
    data_root: str | Path | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    root = resolve_data_root(data_root)
    gold_path = Path(gold_file) if gold_file else default_gold_path(snapshot_id, root)
    load_or_create_calibration_gold(snapshot_id, gold_path, data_root=root)
    summary = review_summary(snapshot_id, gold_path, data_root=root)
    if summary.empty:
        raise ValueError(f"No episodes available for reliability review in snapshot {snapshot_id!r}")
    rows = []
    for row in summary.to_dict("records"):
        episode_id = str(row["episode_id"])
        gold_score = _safe_int(row.get("gold_score"))
        auto_score = _safe_int(row.get("auto_score"))
        score = gold_score if gold_score is not None else auto_score
        rows.append(
            {
                "episode_id": episode_id,
                "task": row.get("task"),
                "auto_score": auto_score,
                "gold_score": gold_score,
                "review_score": score,
                "score_changed": gold_score is not None and auto_score is not None and gold_score != auto_score,
                "sort_key": _stable_key(f"{seed}:{episode_id}"),
            }
        )
    selected = _select_subset(rows, count)
    payload = {
        "snapshot_id": snapshot_id,
        "gold_file": _portable_path(gold_path),
        "selection_policy": (
            "Deterministic boundary/subgoal reliability subset. Prefer score-changed episodes, "
            "but preserve unchanged examples and quality-score coverage."
        ),
        "requested_count": int(count),
        "episode_count": len(selected),
        "episode_ids": [row["episode_id"] for row in selected],
        "quality_counts": _counts(row["review_score"] for row in selected),
        "score_changed_counts": _counts(row["score_changed"] for row in selected),
        "episodes": selected,
        "review_instruction": (
            "Open the review GUI with --review-goal boundary_subgoal. Leave the score alone. "
            "For each episode, check the subtask-boundary and subgoal boxes if the auto labels look acceptable; "
            "leave them unchecked only when you genuinely disagree, then Save review and next."
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _select_subset(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    if count >= len(rows):
        return sorted(rows, key=lambda row: (str(row.get("review_score")), row["sort_key"]))
    changed = [row for row in rows if row["score_changed"]]
    unchanged = [row for row in rows if not row["score_changed"]]
    changed_target = min(len(changed), max(count // 2, int(round(count * 0.6))))
    unchanged_target = count - changed_target
    if unchanged_target > len(unchanged):
        changed_target = min(len(changed), count - len(unchanged))
        unchanged_target = count - changed_target
    selected = _stratified_pick(changed, changed_target) + _stratified_pick(unchanged, unchanged_target)
    if len(selected) < count:
        selected_ids = {row["episode_id"] for row in selected}
        remaining = [row for row in rows if row["episode_id"] not in selected_ids]
        selected.extend(sorted(remaining, key=lambda row: row["sort_key"])[: count - len(selected)])
    return sorted(selected[:count], key=lambda row: row["episode_id"])


def _stratified_pick(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("review_score"))].append(row)
    for key in buckets:
        buckets[key].sort(key=lambda row: row["sort_key"])
    selected = []
    keys = sorted(buckets)
    while len(selected) < count and any(buckets.values()):
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop(0))
    return selected


def _counts(values) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(Counter(values).items(), key=lambda item: str(item[0]))}


def _stable_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan a focused boundary/subgoal human reliability review subset.")
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--output", default="gold_sets/boundary_subgoal_review_50.json")
    parser.add_argument("--gold-file", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    payload = plan_reliability_review(
        args.snapshot,
        args.output,
        count=args.count,
        gold_file=Path(args.gold_file) if args.gold_file else None,
        data_root=Path(args.data_root) if args.data_root else None,
        seed=args.seed,
    )
    output = Path(args.output)
    gold_path = Path(args.gold_file) if args.gold_file else default_gold_path(args.snapshot, Path(args.data_root) if args.data_root else None)
    command = (
        ".\\.venv\\Scripts\\python.exe -m bridgeengine.review_gui "
        f"--snapshot {args.snapshot} "
        f"--gold-file {gold_path.resolve()} "
        f"--episode-file {output} "
        "--review-goal boundary_subgoal "
        "--port 8787"
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "episode_count": payload["episode_count"],
                "quality_counts": payload["quality_counts"],
                "score_changed_counts": payload["score_changed_counts"],
                "review_gui_command": command,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
